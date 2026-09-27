"""
AegisBurn AI — Anomaly Detection Evaluation Harness
====================================================

Solves the evaluation gap in the SIH problem statement:

    "Anomaly Detection Score: a False Negative (missing a
    defective part) is catastrophic, penalizing teams that let
    bad parts escape."

This script:

1. Generates (or reuses) a HELD-OUT labeled evaluation dataset,
   using a different random seed than the production training
   data (src/data/raw/component_data.csv), so it is genuinely
   unseen by the trained models.

2. Loads the trained anomaly model registry
   (models/anomaly/anomaly_models.pkl).

3. Runs inference using ONLY value_0h/value_24h/value_96h (plus
   component_type/parameter_name/lot_id for grouping and lot
   baselines) — defect_type/is_defective are NEVER passed to the
   model. They are used afterwards, purely to score predictions
   against ground truth.

4. Reports, per component family and overall:
       TP, TN, FP, FN
       precision, recall, specificity
       false negative rate  (FN / (FN + TP))  <-- the metric that
                                                   matters most here
       F1
       ROC-AUC, PR-AUC (using the continuous anomaly_index)

5. Sweeps the flagging_threshold (0-100) to find the lowest
   threshold that hits a target recall (default 98%, i.e. at
   most ~2% of truly defective parts are missed), per family.

6. With --apply, writes the recommended thresholds back into the
   saved model registry, so the live system actually uses the
   FN-minimized threshold instead of the model's naive default.

USAGE:

    python evaluate_anomaly.py
    python evaluate_anomaly.py --target-recall 0.99
    python evaluate_anomaly.py --target-recall 0.98 --apply
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.anomaly_detection import (
    DEFAULT_MODEL_PATH,
    ParameterModelRegistry,
)
from src.data_generator import generate_dataset


# ============================================================
# CONFIG
# ============================================================

# Deliberately different from the production dataset's seed (42)
# so this evaluation set is genuinely held-out, not just a
# re-shuffle of data the model may have influenced feature
# statistics from.
EVAL_SEED = 20260913

EVAL_DATA_PATH = Path(
    "src/data/eval/held_out_labeled.csv"
)

MODEL_INPUT_COLUMNS = [
    "component_id",
    "component_type",
    "parameter_name",
    "unit",
    "lot_id",
    "temperature_c",
    "value_0h",
    "value_24h",
    "value_96h",
    "value_168h",
]


# ============================================================
# HELD-OUT DATASET
# ============================================================

def get_eval_dataset(
    force_regenerate: bool = False,
) -> pd.DataFrame:
    """
    Load the cached held-out labeled evaluation set, generating
    it once (with a seed distinct from production) if it does
    not already exist.
    """

    if (
        EVAL_DATA_PATH.exists()
        and not force_regenerate
    ):

        print(
            f"Loading cached held-out evaluation set: "
            f"{EVAL_DATA_PATH}"
        )

        return pd.read_csv(
            EVAL_DATA_PATH
        )

    print(
        "Generating held-out evaluation set "
        f"(seed={EVAL_SEED}, distinct from production seed 42)..."
    )

    df = generate_dataset(
        seed=EVAL_SEED
    )

    EVAL_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        EVAL_DATA_PATH,
        index=False,
    )

    print(
        f"Saved held-out evaluation set: {EVAL_DATA_PATH}"
    )

    return df


# ============================================================
# METRICS
# ============================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
) -> Dict[str, float]:

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    precision = (
        tp / (tp + fp) if (tp + fp) > 0 else 0.0
    )

    recall = (
        tp / (tp + fn) if (tp + fn) > 0 else 0.0
    )

    specificity = (
        tn / (tn + fp) if (tn + fp) > 0 else 0.0
    )

    false_negative_rate = (
        fn / (fn + tp) if (fn + tp) > 0 else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    try:
        roc_auc = roc_auc_score(y_true, scores)
    except ValueError:
        roc_auc = float("nan")

    try:
        pr_auc = average_precision_score(y_true, scores)
    except ValueError:
        pr_auc = float("nan")

    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "false_negative_rate": false_negative_rate,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


def find_threshold_for_recall(
    y_true: np.ndarray,
    scores: np.ndarray,
    target_recall: float,
) -> float:
    """
    Sweep flagging thresholds (0-100) and return the LOWEST
    threshold whose recall is still >= target_recall. Lower
    threshold = flag more components = catch more true
    defectives, at the cost of more false positives.

    If no threshold reaches the target (e.g. the model simply
    cannot separate the classes well enough), returns 0.0 (flag
    everything) as the most conservative fallback for a
    FN-penalized objective.
    """

    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)

    candidate_thresholds = np.arange(
        0.0, 100.5, 0.5
    )

    best_threshold = 0.0

    # Sweep from high threshold to low. Recall only ever
    # increases (or stays the same) as the threshold drops, so
    # the first threshold (scanning high->low) that reaches the
    # target is the highest threshold that still hits it.
    for threshold in sorted(
        candidate_thresholds,
        reverse=True,
    ):

        y_pred = (scores >= threshold).astype(int)

        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))

        recall = (
            tp / (tp + fn) if (tp + fn) > 0 else 0.0
        )

        if recall >= target_recall:
            best_threshold = threshold
            break

    return float(best_threshold)


# ============================================================
# MAIN EVALUATION
# ============================================================

def run_evaluation(
    target_recall: float,
    apply_thresholds: bool,
    regenerate: bool,
) -> None:

    eval_df = get_eval_dataset(
        force_regenerate=regenerate
    )

    print()
    print("=" * 78)
    print(
        f"HELD-OUT EVALUATION SET: {len(eval_df):,} components "
        f"(seed={EVAL_SEED})"
    )

    defect_rate = eval_df["is_defective"].mean()

    print(
        f"True defect rate in this set: {defect_rate:.1%} "
        "(ground truth — never shown to the model)"
    )
    print("=" * 78)

    if not Path(DEFAULT_MODEL_PATH).exists():

        print(
            f"\nERROR: trained model not found at "
            f"{DEFAULT_MODEL_PATH}. Run "
            "'python src/anomaly_detection.py' first."
        )

        return

    registry = ParameterModelRegistry.load(
        DEFAULT_MODEL_PATH
    )

    # ------------------------------------------------------------
    # Inference: only the columns a real early-screening pipeline
    # would have. defect_type/is_defective are set aside now and
    # reattached only for scoring, never passed into the model.
    # ------------------------------------------------------------

    model_input = eval_df[
        [
            column
            for column in MODEL_INPUT_COLUMNS
            if column in eval_df.columns
        ]
    ].copy()

    ground_truth = eval_df[
        "is_defective"
    ].astype(int).to_numpy()

    analyzed = registry.analyze(
        model_input
    )

    analyzed["is_defective"] = ground_truth

    print()
    print("=" * 78)
    print(
        f"RESULTS AT DEFAULT (MODEL-NATIVE) THRESHOLDS"
    )
    print("=" * 78)

    recommended: Dict[Any, float] = {}

    families = sorted(
        registry.models.keys()
    )

    overall_y_true = []
    overall_y_pred = []
    overall_scores = []

    for family in families:

        component_type, parameter_name = family

        mask = (
            (eval_df["component_type"] == component_type)
            & (eval_df["parameter_name"] == parameter_name)
        )

        family_df = analyzed.loc[
            mask.to_numpy()
        ]

        y_true = family_df["is_defective"].to_numpy()
        y_pred = family_df["anomaly_flag"].to_numpy()
        scores = family_df["anomaly_index"].to_numpy()

        overall_y_true.append(y_true)
        overall_y_pred.append(y_pred)
        overall_scores.append(scores)

        metrics = compute_metrics(
            y_true,
            y_pred,
            scores,
        )

        print(
            f"\n{component_type} / {parameter_name} "
            f"(n={len(family_df)}, "
            f"true defective={int(y_true.sum())})"
        )

        print(
            f"  TP={metrics['tp']}  FN={metrics['fn']}  "
            f"FP={metrics['fp']}  TN={metrics['tn']}"
        )

        print(
            f"  Recall (catch rate):     {metrics['recall']:.1%}"
        )

        print(
            f"  False negative rate:     "
            f"{metrics['false_negative_rate']:.1%}  "
            "<-- missed defective parts"
        )

        print(
            f"  Precision:               {metrics['precision']:.1%}"
        )

        print(
            f"  Specificity:             {metrics['specificity']:.1%}"
        )

        print(
            f"  F1:                      {metrics['f1']:.3f}"
        )

        print(
            f"  ROC-AUC:                 {metrics['roc_auc']:.3f}"
        )

        print(
            f"  PR-AUC:                  {metrics['pr_auc']:.3f}"
        )

        # ---------------------------------------------------------
        # Threshold calibration for this family.
        # ---------------------------------------------------------

        calibrated_threshold = find_threshold_for_recall(
            y_true,
            scores,
            target_recall,
        )

        recommended[family] = calibrated_threshold

        calibrated_pred = (
            scores >= calibrated_threshold
        ).astype(int)

        calibrated_metrics = compute_metrics(
            y_true,
            calibrated_pred,
            scores,
        )

        current_threshold = (
            registry.models[family]
            .calibration
            .flagging_threshold
        )

        print(
            f"  Current flagging_threshold: "
            f"{current_threshold:.1f}"
        )

        print(
            f"  Calibrated for >= {target_recall:.0%} recall: "
            f"threshold={calibrated_threshold:.1f}  "
            f"-> recall={calibrated_metrics['recall']:.1%}  "
            f"FN rate={calibrated_metrics['false_negative_rate']:.1%}  "
            f"precision={calibrated_metrics['precision']:.1%}"
        )

    # ------------------------------------------------------------
    # Overall (all families combined)
    # ------------------------------------------------------------

    overall_y_true = np.concatenate(overall_y_true)
    overall_y_pred = np.concatenate(overall_y_pred)
    overall_scores = np.concatenate(overall_scores)

    overall_metrics = compute_metrics(
        overall_y_true,
        overall_y_pred,
        overall_scores,
    )

    print()
    print("=" * 78)
    print("OVERALL (ALL FAMILIES, DEFAULT THRESHOLDS)")
    print("=" * 78)

    print(
        f"Recall (catch rate):     {overall_metrics['recall']:.1%}"
    )

    print(
        f"False negative rate:     "
        f"{overall_metrics['false_negative_rate']:.1%}"
    )

    print(
        f"Precision:               {overall_metrics['precision']:.1%}"
    )

    print(
        f"F1:                      {overall_metrics['f1']:.3f}"
    )

    print(
        f"ROC-AUC:                 {overall_metrics['roc_auc']:.3f}"
    )

    print(
        f"PR-AUC:                  {overall_metrics['pr_auc']:.3f}"
    )

    # ------------------------------------------------------------
    # Breakdown by defect_type. LATENT_DEFECT is the core
    # motivating case in the problem statement -- a component
    # that passes absolute limits but drifts subtly over time --
    # so its recall is reported explicitly, not just folded into
    # the overall "defective" recall above.
    # ------------------------------------------------------------

    if "defect_type" in eval_df.columns:

        print()
        print("=" * 78)
        print("RECALL BY DEFECT TYPE (all families combined)")
        print("=" * 78)

        analyzed_all = registry.analyze(
            model_input
        )

        analyzed_all["defect_type"] = eval_df[
            "defect_type"
        ].to_numpy()

        for defect_type in [
            "GRADUAL_DRIFT",
            "LATENT_DEFECT",
            "SUDDEN_ANOMALY",
        ]:

            subset = analyzed_all[
                analyzed_all["defect_type"]
                == defect_type
            ]

            if len(subset) == 0:
                continue

            caught = int(
                subset["anomaly_flag"].sum()
            )

            total = len(subset)

            recall = caught / total

            missed = total - caught

            print(
                f"{defect_type}: {caught}/{total} caught "
                f"({recall:.1%})  -- {missed} missed"
            )

        normal_subset = analyzed_all[
            analyzed_all["defect_type"] == "NORMAL"
        ]

        if len(normal_subset) > 0:

            false_alarm_rate = normal_subset[
                "anomaly_flag"
            ].mean()

            print(
                f"\nFalse alarm rate on truly NORMAL parts: "
                f"{false_alarm_rate:.1%}"
            )

    # ------------------------------------------------------------
    # Apply calibrated thresholds and re-save, if requested.
    # ------------------------------------------------------------

    if apply_thresholds:

        print()
        print("=" * 78)
        print(
            f"APPLYING CALIBRATED THRESHOLDS "
            f"(target recall >= {target_recall:.0%})"
        )
        print("=" * 78)

        for family, threshold in recommended.items():

            registry.models[
                family
            ].calibration.flagging_threshold = threshold

            print(
                f"{family[0]} / {family[1]}: "
                f"flagging_threshold -> {threshold:.1f}"
            )

        saved_path = registry.save(
            DEFAULT_MODEL_PATH
        )

        print(
            f"\nSaved recalibrated model registry: {saved_path}"
        )

    else:

        print()
        print(
            "Run again with --apply to write these calibrated "
            "thresholds into the live model."
        )


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate AegisBurn AI's anomaly detector against a "
            "held-out labeled dataset, with false-negative-aware "
            "threshold calibration."
        )
    )

    parser.add_argument(
        "--target-recall",
        type=float,
        default=0.98,
        help=(
            "Minimum fraction of truly defective parts that must "
            "be caught (default: 0.98, i.e. at most 2%% missed)."
        ),
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Write the calibrated thresholds into "
            "models/anomaly/anomaly_models.pkl."
        ),
    )

    parser.add_argument(
        "--regenerate",
        action="store_true",
        help=(
            "Force regeneration of the held-out evaluation "
            "dataset, even if a cached copy exists."
        ),
    )

    args = parser.parse_args()

    run_evaluation(
        target_recall=args.target_recall,
        apply_thresholds=args.apply,
        regenerate=args.regenerate,
    )