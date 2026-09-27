"""
AegisBurn AI - Unsupervised Anomaly Detection
================================================

Detects abnormal burn-in behavior in electronic components.

Important design rules
----------------------
1. No defect_type column is required.
2. No is_defective column is required.
3. Anomaly detection uses only measurements available before 168h:
       value_0h
       value_24h
       value_96h
4. value_168h is NOT used by the anomaly detector.
5. Each component family gets its own Isolation Forest model.
6. Anomaly scores are calibrated continuously from the training
   distribution instead of using a direct 0/100 mapping.
7. The module remains compatible with the AegisBurn backend registry.
"""

from __future__ import annotations

import argparse
import os
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATASET_PATH = (
    PROJECT_ROOT
    / "src"
    / "data"
    / "raw"
    / "component_data.csv"
)

DEFAULT_MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "anomaly"
)

DEFAULT_MODEL_PATH = (
    DEFAULT_MODEL_DIR
    / "anomaly_models.pkl"
)


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

GROUP_COLUMNS = [
    "component_type",
    "parameter_name",
]

IDENTIFIER_COLUMNS = [
    "component_id",
    "component_type",
    "parameter_name",
    "unit",
    "lot_id",
]

MEASUREMENT_COLUMNS = [
    "value_0h",
    "value_24h",
    "value_96h",
    "value_168h",
]


# ---------------------------------------------------------------------
# IMPORTANT:
# These are the only features used by the anomaly detector.
#
# value_168h is intentionally excluded.
# The goal is to detect abnormal behavior before the final 168h
# burn-in measurement is known.
# ---------------------------------------------------------------------

TRAJECTORY_FEATURES = [
    "value_0h",
    "value_24h",
    "value_96h",

    "drift_0_24",
    "drift_24_96",
    "drift_0_96",

    "relative_drift_0_24",
    "relative_drift_24_96",
    "relative_drift_0_96",

    "slope_early",
    "slope_mid",

    "ratio_24_0",
    "ratio_96_24",
    "ratio_96_0",

    "acceleration",
]


# ---------------------------------------------------------------------
# LOT-RELATIVE FEATURES
#
# Static/global comparisons (this component vs. the whole parameter
# population across every lot) miss the case called out in the SIH
# problem statement: a lot averaging 10µA where one part reads 45µA
# is a severe anomaly relative to ITS OWN LOT, even though 45µA may
# be well under the datasheet's absolute limit and unremarkable
# compared to the global population.
#
# These features express each measurement as a z-score against its
# own lot's mean/std, learned from training data and looked up by
# lot_id at inference time (with fallback to the parameter-family
# baseline for lots that are new/unseen or too small to trust).
# ---------------------------------------------------------------------

LOT_RELATIVE_FEATURES = [
    "lot_zscore_0h",
    "lot_zscore_24h",
    "lot_zscore_96h",
]

# A lot needs at least this many rows before its own mean/std are
# trusted over the parameter-family-wide fallback baseline.
MIN_LOT_SIZE = 3


# ---------------------------------------------------------------------
# EARLY-ONLY (0h/24h) anomaly features
#
# Used automatically for components that are still mid burn-in and
# only have value_0h/value_24h measured so far. This is the 96h-
# independent subset of TRAJECTORY_FEATURES/LOT_RELATIVE_FEATURES
# above. Without this, a missing value_96h would leave every
# 96h-derived feature as NaN, which previously got silently
# zero-filled -- making an ordinary in-progress component look like
# an extreme statistical outlier to Isolation Forest for no real
# reason.
# ---------------------------------------------------------------------

EARLY_TRAJECTORY_FEATURES = [
    "value_0h",
    "value_24h",

    "drift_0_24",
    "relative_drift_0_24",
    "slope_early",
    "ratio_24_0",
]

EARLY_LOT_RELATIVE_FEATURES = [
    "lot_zscore_0h",
    "lot_zscore_24h",
]


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to float safely."""
    try:
        result = float(value)

        if not np.isfinite(result):
            return default

        return result

    except (TypeError, ValueError):
        return default


def _safe_divide(
    numerator: Any,
    denominator: Any,
    epsilon: float = 1e-9,
) -> float:
    """Safe scalar division."""
    num = _safe_float(numerator)
    den = _safe_float(denominator)

    if abs(den) < epsilon:
        den = epsilon if den >= 0 else -epsilon

    result = num / den

    if not np.isfinite(result):
        return 0.0

    return float(result)


def _clip_float(
    value: float,
    low: float,
    high: float,
) -> float:
    """Finite numeric clipping."""
    if not np.isfinite(value):
        return low

    return float(max(low, min(high, value)))


def _sigmoid(value: float) -> float:
    """
    Numerically stable sigmoid.
    """
    value = _safe_float(value)

    if value >= 40:
        return 1.0

    if value <= -40:
        return 0.0

    return 1.0 / (1.0 + np.exp(-value))


def _robust_scale(values: np.ndarray) -> float:
    """
    Robust scale based on MAD.

    Falls back to standard deviation and finally 1.0.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return 1.0

    median = float(np.median(values))

    mad = float(
        np.median(
            np.abs(values - median)
        )
    )

    scale = 1.4826 * mad

    if np.isfinite(scale) and scale > 1e-9:
        return float(scale)

    std = float(np.std(values))

    if np.isfinite(std) and std > 1e-9:
        return float(std)

    return 1.0


def _percentile(values: np.ndarray, q: float) -> float:
    """
    Safe percentile.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return 0.0

    return float(np.percentile(values, q))


# ---------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------

def validate_dataset(df: pd.DataFrame) -> None:
    """
    Validate the minimum dataset structure.

    defect_type and is_defective are deliberately NOT required.
    """
    required = [
        "component_id",
        "component_type",
        "parameter_name",
        "unit",
        "lot_id",
        "value_0h",
        "value_24h",
        "value_96h",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(missing)
        )


def load_dataset(
    dataset_path: str | os.PathLike[str],
) -> pd.DataFrame:
    """
    Load and validate the burn-in dataset.
    """
    path = Path(dataset_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    df = pd.read_csv(path)

    validate_dataset(df)

    numeric_columns = [
        "value_0h",
        "value_24h",
        "value_96h",
    ]

    if "value_168h" in df.columns:
        numeric_columns.append("value_168h")

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "component_type",
            "parameter_name",
            "value_0h",
            "value_24h",
            "value_96h",
        ]
    ).copy()

    return df


# ---------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------

def create_anomaly_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create anomaly-detection features using only 0h, 24h and 96h.

    168h is intentionally ignored.
    """
    result = df.copy()

    v0 = pd.to_numeric(
        result["value_0h"],
        errors="coerce",
    )

    v24 = pd.to_numeric(
        result["value_24h"],
        errors="coerce",
    )

    v96 = pd.to_numeric(
        result["value_96h"],
        errors="coerce",
    )

    eps = 1e-9

    # Capture this BEFORE the fillna(0.0) cleanup below erases the
    # NaN signal -- value_96h is itself one of TRAJECTORY_FEATURES,
    # so it gets zero-filled along with the derived columns. This
    # marker is what lets downstream code detect "still mid
    # burn-in, no 96h yet" instead of misreading a filled 0.0 as
    # if it were a real measurement.
    result["_has_96h"] = v96.notna()

    # -------------------------------------------------------------
    # Absolute drift
    # -------------------------------------------------------------

    result["drift_0_24"] = v24 - v0
    result["drift_24_96"] = v96 - v24
    result["drift_0_96"] = v96 - v0

    # -------------------------------------------------------------
    # Relative drift
    # -------------------------------------------------------------

    result["relative_drift_0_24"] = (
        result["drift_0_24"]
        / v0.abs().clip(lower=eps)
    )

    result["relative_drift_24_96"] = (
        result["drift_24_96"]
        / v24.abs().clip(lower=eps)
    )

    result["relative_drift_0_96"] = (
        result["drift_0_96"]
        / v0.abs().clip(lower=eps)
    )

    # -------------------------------------------------------------
    # Slopes
    #
    # 0 -> 24h = 24 hours
    # 24 -> 96h = 72 hours
    # -------------------------------------------------------------

    result["slope_early"] = (
        result["drift_0_24"]
        / 24.0
    )

    result["slope_mid"] = (
        result["drift_24_96"]
        / 72.0
    )

    # -------------------------------------------------------------
    # Ratios
    # -------------------------------------------------------------

    result["ratio_24_0"] = (
        v24
        / v0.abs().clip(lower=eps)
    )

    result["ratio_96_24"] = (
        v96
        / v24.abs().clip(lower=eps)
    )

    result["ratio_96_0"] = (
        v96
        / v0.abs().clip(lower=eps)
    )

    # -------------------------------------------------------------
    # Acceleration
    #
    # Difference between early and mid slopes.
    # -------------------------------------------------------------

    result["acceleration"] = (
        result["slope_mid"]
        - result["slope_early"]
    )

    # -------------------------------------------------------------
    # Clean numerical values
    # -------------------------------------------------------------

    for column in TRAJECTORY_FEATURES:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

        result[column] = (
            result[column]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .fillna(0.0)
        )

    return result


# ---------------------------------------------------------------------
# Unsupervised normal-core selection
# ---------------------------------------------------------------------

def _build_stability_index(
    feature_df: pd.DataFrame,
) -> np.ndarray:
    """
    Build an unsupervised stability/degradation magnitude index.

    This does NOT use labels.

    The index is only used to identify the stable core of the
    training distribution. Isolation Forest is then fitted on that
    core.

    This is useful because the supplied prototype dataset contains
    many abnormal trajectories. Training directly on all rows can
    cause a large fraction of abnormal trajectories to be treated
    as normal.
    """
    early = np.abs(
        feature_df["relative_drift_0_24"].to_numpy(
            dtype=float
        )
    )

    mid = np.abs(
        feature_df["relative_drift_24_96"].to_numpy(
            dtype=float
        )
    )

    cumulative = np.abs(
        feature_df["relative_drift_0_96"].to_numpy(
            dtype=float
        )
    )

    acceleration = np.abs(
        feature_df["acceleration"].to_numpy(
            dtype=float
        )
    )

    # Robustly scale each dimension.
    early_scale = _robust_scale(early)
    mid_scale = _robust_scale(mid)
    cumulative_scale = _robust_scale(cumulative)
    acceleration_scale = _robust_scale(acceleration)

    score = (
        0.25 * (early / early_scale)
        + 0.30 * (mid / mid_scale)
        + 0.30 * (cumulative / cumulative_scale)
        + 0.15 * (acceleration / acceleration_scale)
    )

    score = np.asarray(
        score,
        dtype=float,
    )

    score[
        ~np.isfinite(score)
    ] = 0.0

    return score


def select_unsupervised_core(
    feature_df: pd.DataFrame,
    core_fraction: float = 0.55,
) -> pd.DataFrame:
    """
    Select a stable unsupervised core from the training data.

    No defect labels are used.

    The default 55% is intentionally close to the expected normal
    majority in the prototype data, but it is not based on the
    removed is_defective column.
    """
    if len(feature_df) <= 20:
        return feature_df.copy()

    core_fraction = _clip_float(
        core_fraction,
        0.40,
        0.80,
    )

    stability = _build_stability_index(
        feature_df
    )

    cutoff = np.quantile(
        stability,
        core_fraction,
    )

    mask = stability <= cutoff

    core = feature_df.loc[mask].copy()

    minimum_core = max(
        20,
        int(len(feature_df) * 0.30),
    )

    if len(core) < minimum_core:
        order = np.argsort(stability)

        selected = order[
            :minimum_core
        ]

        core = feature_df.iloc[
            selected
        ].copy()

    return core


# ---------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------

@dataclass
class ScoreCalibration:
    """
    Statistics required to turn Isolation Forest raw scores into
    continuous 0-100 anomaly scores.
    """

    median: float = 0.0
    scale: float = 1.0
    low_percentile: float = 0.0
    high_percentile: float = 0.0
    threshold_raw: float = 0.0

    # Binary anomaly_flag is decided by anomaly_index >= this
    # value (0-100 scale), NOT by Isolation Forest's own internal
    # contamination-driven cutoff. This makes the flagging
    # threshold something that can be calibrated against a
    # labeled evaluation set to hit a target recall / false-
    # negative rate, instead of being an opaque fixed guess.
    flagging_threshold: float = 50.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "median": float(self.median),
            "scale": float(self.scale),
            "low_percentile": float(
                self.low_percentile
            ),
            "high_percentile": float(
                self.high_percentile
            ),
            "threshold_raw": float(
                self.threshold_raw
            ),
            "flagging_threshold": float(
                self.flagging_threshold
            ),
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
    ) -> "ScoreCalibration":
        return cls(
            median=_safe_float(
                data.get("median"),
                0.0,
            ),
            scale=max(
                _safe_float(
                    data.get("scale"),
                    1.0,
                ),
                1e-9,
            ),
            low_percentile=_safe_float(
                data.get("low_percentile"),
                0.0,
            ),
            high_percentile=_safe_float(
                data.get("high_percentile"),
                0.0,
            ),
            threshold_raw=_safe_float(
                data.get("threshold_raw"),
                0.0,
            ),
            flagging_threshold=_clip_float(
                _safe_float(
                    data.get(
                        "flagging_threshold"
                    ),
                    50.0,
                ),
                0.0,
                100.0,
            ),
        )


def fit_score_calibration(
    raw_scores: np.ndarray,
    threshold_raw: float,
) -> ScoreCalibration:
    """
    Fit robust score calibration.
    """
    raw_scores = np.asarray(
        raw_scores,
        dtype=float,
    )

    raw_scores = raw_scores[
        np.isfinite(raw_scores)
    ]

    if len(raw_scores) == 0:
        return ScoreCalibration(
            median=0.0,
            scale=1.0,
            low_percentile=0.0,
            high_percentile=0.0,
            threshold_raw=threshold_raw,
        )

    median = float(
        np.median(raw_scores)
    )

    scale = _robust_scale(
        raw_scores
    )

    return ScoreCalibration(
        median=median,
        scale=scale,
        low_percentile=_percentile(
            raw_scores,
            5,
        ),
        high_percentile=_percentile(
            raw_scores,
            95,
        ),
        threshold_raw=float(
            threshold_raw
        ),
    )


def raw_to_anomaly_score(
    raw_score: float,
    calibration: ScoreCalibration,
) -> float:
    """
    Convert Isolation Forest's raw normality score into a
    continuous anomaly score.

    Isolation Forest:
        higher raw score = more normal
        lower raw score = more anomalous

    Therefore:
        median - raw_score
    increases as the component becomes more anomalous.

    The sigmoid keeps the output continuous and avoids the old
    direct mapping that produced excessive 0/100 saturation.
    """
    raw_score = _safe_float(
        raw_score
    )

    median = _safe_float(
        calibration.median,
        0.0,
    )

    scale = max(
        _safe_float(
            calibration.scale,
            1.0,
        ),
        1e-9,
    )

    z = (
        median
        - raw_score
    ) / scale

    # Moderate the curve so normal components remain around the
    # middle of the scale and abnormal components move upward
    # progressively.
    probability = _sigmoid(
        z
    )

    score = (
        probability
        * 100.0
    )

    # Do not return exact 0 or 100.
    score = _clip_float(
        score,
        1.0,
        99.0,
    )

    return float(score)


# ---------------------------------------------------------------------
# Parameter anomaly detector
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Lot-relative baselines
# ---------------------------------------------------------------------

@dataclass
class LotBaselineStore:
    """
    Per-lot mean/std baselines for value_0h/24h/96h, learned from
    training data, with fallback to the parameter-family-wide
    baseline for lots that are unseen at inference time or too
    small (fewer than MIN_LOT_SIZE rows) to trust.

    This is what makes the anomaly detector "dynamic": a component
    is judged against what is typical for its OWN lot, not just
    against the global population spanning every lot.
    """

    lot_stats: Dict[str, Dict[str, float]]
    family_stats: Dict[str, float]
    min_lot_size: int = MIN_LOT_SIZE

    VALUE_COLUMNS = (
        "value_0h",
        "value_24h",
        "value_96h",
    )

    @classmethod
    def fit(
        cls,
        df: pd.DataFrame,
        min_lot_size: int = MIN_LOT_SIZE,
    ) -> "LotBaselineStore":
        """
        Learn family-wide and per-lot mean/std baselines from a
        training dataframe. df is expected to already be scoped to
        a single component_type/parameter_name family.
        """

        family_stats: Dict[str, float] = {}

        for column in cls.VALUE_COLUMNS:

            values = pd.to_numeric(
                df[column],
                errors="coerce",
            )

            family_stats[f"{column}_mean"] = _safe_float(
                values.mean(),
                0.0,
            )

            std = _safe_float(
                values.std(),
                0.0,
            )

            family_stats[f"{column}_std"] = (
                std if std > 1e-9 else 1.0
            )

        lot_stats: Dict[str, Dict[str, float]] = {}

        if "lot_id" in df.columns:

            grouped = df.groupby(
                df["lot_id"].astype(str)
            )

            for lot_id, group in grouped:

                if len(group) < min_lot_size:
                    # Too few rows to trust this lot's own
                    # statistics; it will fall back to the
                    # family baseline at lookup time.
                    continue

                stats: Dict[str, float] = {}

                for column in cls.VALUE_COLUMNS:

                    values = pd.to_numeric(
                        group[column],
                        errors="coerce",
                    )

                    mean = _safe_float(
                        values.mean(),
                        family_stats[
                            f"{column}_mean"
                        ],
                    )

                    std = _safe_float(
                        values.std(),
                        0.0,
                    )

                    if std <= 1e-9:
                        std = family_stats[
                            f"{column}_std"
                        ]

                    stats[f"{column}_mean"] = mean
                    stats[f"{column}_std"] = std

                lot_stats[str(lot_id)] = stats

        return cls(
            lot_stats=lot_stats,
            family_stats=family_stats,
            min_lot_size=min_lot_size,
        )

    def _baseline_for_lot(
        self,
        lot_id: Any,
    ) -> Dict[str, float]:

        return self.lot_stats.get(
            str(lot_id),
            self.family_stats,
        )

    def add_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Return a copy of df with lot_zscore_0h/24h/96h columns
        added, computed against each row's own lot baseline (or
        the family fallback baseline when the lot is unseen).
        """

        result = df.copy()

        if "lot_id" in result.columns:
            lot_ids = result["lot_id"]
        else:
            lot_ids = pd.Series(
                [""] * len(result),
                index=result.index,
            )

        for column in self.VALUE_COLUMNS:

            suffix = column.split("_")[1]

            zscore_column = (
                f"lot_zscore_{suffix}"
            )

            mean_column = (
                f"lot_mean_{suffix}"
            )

            std_column = (
                f"lot_std_{suffix}"
            )

            values = pd.to_numeric(
                result[column],
                errors="coerce",
            )

            zscores = np.zeros(
                len(result),
                dtype=float,
            )

            means = np.zeros(
                len(result),
                dtype=float,
            )

            stds = np.zeros(
                len(result),
                dtype=float,
            )

            for position, (
                _,
                lot_id,
            ) in enumerate(
                lot_ids.items()
            ):

                baseline = (
                    self._baseline_for_lot(
                        lot_id
                    )
                )

                mean = baseline.get(
                    f"{column}_mean",
                    0.0,
                )

                std = baseline.get(
                    f"{column}_std",
                    1.0,
                )

                means[position] = mean
                stds[position] = std

                value = values.iloc[
                    position
                ]

                if (
                    pd.isna(value)
                    or std <= 1e-9
                ):
                    zscores[position] = 0.0
                else:
                    zscores[position] = (
                        (value - mean)
                        / std
                    )

            result[mean_column] = means
            result[std_column] = stds

            result[zscore_column] = np.clip(
                zscores,
                -20.0,
                20.0,
            )

        return result

    def to_dict(self) -> Dict[str, Any]:

        return {
            "lot_stats": self.lot_stats,
            "family_stats": self.family_stats,
            "min_lot_size": self.min_lot_size,
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
    ) -> "LotBaselineStore":

        return cls(
            lot_stats=(
                data.get("lot_stats", {})
                or {}
            ),
            family_stats=(
                data.get("family_stats", {})
                or {}
            ),
            min_lot_size=int(
                data.get(
                    "min_lot_size",
                    MIN_LOT_SIZE,
                )
            ),
        )


class ParameterAnomalyDetector:
    """
    Isolation Forest detector for one component_type /
    parameter_name family.
    """

    def __init__(
        self,
        component_type: Optional[str] = None,
        parameter_name: Optional[str] = None,
        contamination: float = 0.10,
        n_estimators: int = 400,
        random_state: int = 42,
    ):
        self.component_type = (
            component_type
        )

        self.parameter_name = (
            parameter_name
        )

        self.contamination = _clip_float(
            contamination,
            0.01,
            0.49,
        )

        self.n_estimators = int(
            max(
                100,
                n_estimators,
            )
        )

        self.random_state = int(
            random_state
        )

        self.model: Optional[
            IsolationForest
        ] = None

        self.feature_columns = (
            list(TRAJECTORY_FEATURES)
            + list(LOT_RELATIVE_FEATURES)
        )

        self.lot_baselines: Optional[
            LotBaselineStore
        ] = None

        self.calibration = (
            ScoreCalibration()
        )

        self.training_rows = 0
        self.core_training_rows = 0

        self.fitted = False

        # -------------------------------------------------------
        # Early-only (0h/24h) model -- used automatically when
        # value_96h is not yet available for a component (still
        # mid burn-in).
        # -------------------------------------------------------

        self.early_feature_columns = (
            list(EARLY_TRAJECTORY_FEATURES)
            + list(EARLY_LOT_RELATIVE_FEATURES)
        )

        self.early_model: Optional[
            IsolationForest
        ] = None

        self.early_calibration = (
            ScoreCalibration()
        )

        self.early_training_rows = 0
        self.early_core_training_rows = 0

    # -----------------------------------------------------------------
    # Matrix creation
    # -----------------------------------------------------------------

    def _matrix(
        self,
        feature_df: pd.DataFrame,
    ) -> np.ndarray:
        """
        Convert feature dataframe into model matrix (full
        0h/24h/96h feature set).
        """
        return self._matrix_for_columns(
            feature_df,
            self.feature_columns,
        )

    def _early_matrix(
        self,
        feature_df: pd.DataFrame,
    ) -> np.ndarray:
        """
        Convert feature dataframe into model matrix (early-only
        0h/24h feature set).
        """
        return self._matrix_for_columns(
            feature_df,
            self.early_feature_columns,
        )

    def _matrix_for_columns(
        self,
        feature_df: pd.DataFrame,
        columns: list,
    ) -> np.ndarray:

        matrix = feature_df[
            columns
        ].copy()

        matrix = matrix.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        matrix = matrix.fillna(
            0.0
        )

        matrix = matrix.astype(
            float
        )

        return matrix.to_numpy(
            dtype=np.float64
        )

    # -----------------------------------------------------------------
    # Lot-relative feature enrichment
    # -----------------------------------------------------------------

    def _enrich_with_lot_features(
        self,
        feature_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add lot_zscore_0h/24h/96h columns using baselines learned
        during training. Falls back to zero-filled columns if this
        detector predates lot baselines (e.g. an older saved model
        loaded before this feature existed).
        """

        if self.lot_baselines is None:

            result = feature_df.copy()

            for column in LOT_RELATIVE_FEATURES:

                result[column] = 0.0

            return result

        return self.lot_baselines.add_features(
            feature_df
        )

    def _has_96h(
        self,
        df: pd.DataFrame,
    ) -> pd.Series:
        """
        Per-row boolean: is value_96h actually available for this
        component? False for components still mid burn-in.

        Prefers the _has_96h marker set by create_anomaly_features
        (captured before value_96h itself gets zero-filled as part
        of TRAJECTORY_FEATURES cleanup). Falls back to checking
        value_96h directly for dataframes that didn't go through
        that function.
        """

        if "_has_96h" in df.columns:

            return df["_has_96h"].astype(
                bool
            )

        if "value_96h" not in df.columns:

            return pd.Series(
                False,
                index=df.index,
            )

        return pd.to_numeric(
            df["value_96h"],
            errors="coerce",
        ).notna()

    # -----------------------------------------------------------------
    # Fit
    # -----------------------------------------------------------------

    def fit(
        self,
        feature_df: pd.DataFrame,
    ) -> "ParameterAnomalyDetector":
        """
        Fit the detector without using ground-truth labels.
        """
        if feature_df.empty:
            raise ValueError(
                "Cannot train anomaly detector "
                "with an empty dataframe."
            )

        self.training_rows = len(
            feature_df
        )

        # -------------------------------------------------------------
        # Learn per-lot baselines from this family's training data,
        # then enrich with lot-relative z-score features.
        # -------------------------------------------------------------

        self.lot_baselines = (
            LotBaselineStore.fit(
                feature_df
            )
        )

        enriched_df = (
            self._enrich_with_lot_features(
                feature_df
            )
        )

        matrix = self._matrix(
            enriched_df
        )

        # -------------------------------------------------------------
        # Select a stable unsupervised core.
        # -------------------------------------------------------------

        core_df = select_unsupervised_core(
            enriched_df
        )

        core_matrix = self._matrix(
            core_df
        )

        self.core_training_rows = len(
            core_df
        )

        # -------------------------------------------------------------
        # Train Isolation Forest.
        # -------------------------------------------------------------

        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
            bootstrap=False,
        )

        self.model.fit(
            core_matrix
        )

        # -------------------------------------------------------------
        # Calibrate against the stable core.
        # -------------------------------------------------------------

        core_raw_scores = (
            self.model.decision_function(
                core_matrix
            )
        )

        # The model's own prediction threshold is retained as the
        # binary anomaly boundary.
        threshold_raw = (
            _safe_float(
                getattr(
                    self.model,
                    "offset_",
                    0.0,
                ),
                0.0,
            )
        )

        self.calibration = (
            fit_score_calibration(
                core_raw_scores,
                threshold_raw,
            )
        )

        # Default the flagging threshold to whatever anomaly_index
        # corresponds to Isolation Forest's own natural boundary,
        # so behavior is unchanged until it is deliberately
        # recalibrated against a labeled evaluation set (see
        # evaluate_anomaly.py) to target a specific recall / false
        # negative rate.
        self.calibration.flagging_threshold = (
            raw_to_anomaly_score(
                threshold_raw,
                self.calibration,
            )
        )

        # -------------------------------------------------------------
        # Early-only (0h/24h) model -- used automatically for
        # components still mid burn-in that don't have value_96h
        # yet. Trained on the same stable core rows, but only the
        # 96h-independent features, so it never sees NaN/zero-
        # filled 96h-derived columns.
        # -------------------------------------------------------------

        self.early_training_rows = len(
            feature_df
        )

        self.early_core_training_rows = len(
            core_df
        )

        early_core_matrix = self._early_matrix(
            core_df
        )

        self.early_model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
            bootstrap=False,
        )

        self.early_model.fit(
            early_core_matrix
        )

        early_core_raw_scores = (
            self.early_model.decision_function(
                early_core_matrix
            )
        )

        early_threshold_raw = (
            _safe_float(
                getattr(
                    self.early_model,
                    "offset_",
                    0.0,
                ),
                0.0,
            )
        )

        self.early_calibration = (
            fit_score_calibration(
                early_core_raw_scores,
                early_threshold_raw,
            )
        )

        self.early_calibration.flagging_threshold = (
            raw_to_anomaly_score(
                early_threshold_raw,
                self.early_calibration,
            )
        )

        self.fitted = True

        return self

    # -----------------------------------------------------------------
    # Raw prediction
    # -----------------------------------------------------------------

    def predict_raw(
        self,
        feature_df: pd.DataFrame,
    ) -> np.ndarray:
        """
        Return Isolation Forest raw normality scores.
        """
        if not self.fitted or self.model is None:
            raise RuntimeError(
                "Anomaly detector has not been fitted."
            )

        enriched_df = (
            self._enrich_with_lot_features(
                feature_df
            )
        )

        matrix = self._matrix(
            enriched_df
        )

        return self.model.decision_function(
            matrix
        )

    # -----------------------------------------------------------------
    # Continuous anomaly score
    # -----------------------------------------------------------------

    def predict_score(
        self,
        feature_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Return continuous anomaly scores. Rows with a real
        value_96h use the full (0h/24h/96h) model; rows still mid
        burn-in (no value_96h yet) automatically use the
        early-only (0h/24h) model instead of zero-filling missing
        96h-derived features into the full model.
        """
        if not self.fitted or self.model is None:
            raise RuntimeError(
                "Anomaly detector has not been fitted."
            )

        enriched_df = (
            self._enrich_with_lot_features(
                feature_df
            )
        )

        has_96h = self._has_96h(
            feature_df
        )

        anomaly_indices = np.full(
            len(feature_df),
            0.0,
            dtype=float,
        )

        raw_scores = np.full(
            len(feature_df),
            0.0,
            dtype=float,
        )

        if has_96h.any():

            full_rows = enriched_df.loc[
                has_96h
            ]

            full_matrix = self._matrix(
                full_rows
            )

            full_raw = (
                self.model.decision_function(
                    full_matrix
                )
            )

            full_index = np.array(
                [
                    raw_to_anomaly_score(
                        raw,
                        self.calibration,
                    )
                    for raw in full_raw
                ],
                dtype=float,
            )

            mask = has_96h.to_numpy()
            raw_scores[mask] = full_raw
            anomaly_indices[mask] = full_index

        missing_96h = ~has_96h

        if missing_96h.any():

            early_rows = enriched_df.loc[
                missing_96h
            ]

            early_matrix = self._early_matrix(
                early_rows
            )

            early_raw = (
                self.early_model.decision_function(
                    early_matrix
                )
            )

            early_index = np.array(
                [
                    raw_to_anomaly_score(
                        raw,
                        self.early_calibration,
                    )
                    for raw in early_raw
                ],
                dtype=float,
            )

            mask = missing_96h.to_numpy()
            raw_scores[mask] = early_raw
            anomaly_indices[mask] = early_index

        # Each row is flagged against ITS OWN model's calibrated
        # threshold (full or early), not a single shared value.
        thresholds = np.where(
            has_96h.to_numpy(),
            self.calibration.flagging_threshold,
            self.early_calibration.flagging_threshold,
        )

        flags = (
            anomaly_indices >= thresholds
        ).astype(int)

        return pd.DataFrame(
            {
                "anomaly_flag": flags,

                "anomaly_score": (
                    anomaly_indices
                ),

                "anomaly_index": (
                    anomaly_indices
                ),

                "anomaly_raw_score": (
                    raw_scores
                ),

                "input_stage": np.where(
                    has_96h.to_numpy(),
                    "0h_24h_96h",
                    "0h_24h",
                ),
            },
            index=feature_df.index,
        )

    # -----------------------------------------------------------------
    # Full analysis
    # -----------------------------------------------------------------

    def analyze(
        self,
        feature_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Analyze trajectories and return the original feature data
        plus anomaly outputs.
        """
        result = self._enrich_with_lot_features(
            feature_df
        )

        scores = self.predict_score(
            feature_df
        )

        for column in scores.columns:
            result[column] = scores[
                column
            ]

        result["anomaly_label"] = np.where(
            result["anomaly_flag"] == 1,
            "ANOMALY",
            "NORMAL",
        )

        # Rounded display-friendly fields.
        result["anomaly_score"] = (
            result["anomaly_score"]
            .astype(float)
            .round(4)
        )

        result["anomaly_index"] = (
            result["anomaly_index"]
            .astype(float)
            .round(4)
        )

        return result

    # -----------------------------------------------------------------
    # Serialization
    # -----------------------------------------------------------------

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        """
        Serialize detector metadata.
        """
        return {
            "component_type": self.component_type,
            "parameter_name": self.parameter_name,
            "contamination": self.contamination,
            "n_estimators": self.n_estimators,
            "random_state": self.random_state,
            "feature_columns": list(
                self.feature_columns
            ),
            "calibration": (
                self.calibration.to_dict()
            ),
            "training_rows": (
                self.training_rows
            ),
            "core_training_rows": (
                self.core_training_rows
            ),
            "fitted": self.fitted,
            "model": self.model,
            "lot_baselines": (
                self.lot_baselines.to_dict()
                if self.lot_baselines
                is not None
                else None
            ),
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
    ) -> "ParameterAnomalyDetector":
        """
        Restore detector from serialized dictionary.
        """
        detector = cls(
            component_type=data.get(
                "component_type"
            ),
            parameter_name=data.get(
                "parameter_name"
            ),
            contamination=_safe_float(
                data.get(
                    "contamination",
                    0.10,
                ),
                0.10,
            ),
            n_estimators=int(
                data.get(
                    "n_estimators",
                    400,
                )
            ),
            random_state=int(
                data.get(
                    "random_state",
                    42,
                )
            ),
        )

        detector.feature_columns = list(
            data.get(
                "feature_columns",
                list(TRAJECTORY_FEATURES)
                + list(LOT_RELATIVE_FEATURES),
            )
        )

        detector.calibration = (
            ScoreCalibration.from_dict(
                data.get(
                    "calibration",
                    {},
                )
            )
        )

        detector.training_rows = int(
            data.get(
                "training_rows",
                0,
            )
        )

        detector.core_training_rows = int(
            data.get(
                "core_training_rows",
                0,
            )
        )

        detector.model = data.get(
            "model"
        )

        lot_baselines_data = data.get(
            "lot_baselines"
        )

        detector.lot_baselines = (
            LotBaselineStore.from_dict(
                lot_baselines_data
            )
            if lot_baselines_data
            else None
        )

        detector.fitted = (
            detector.model is not None
        )

        return detector


# ---------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------

class ParameterModelRegistry:
    """
    Registry containing one anomaly detector per component family.

    Example groups:

        Logic IC / Iddq
        Memory IC / Standby Current
        ADC / Leakage Current
        Driver IC / Propagation Delay
    """

    VERSION = 5

    def __init__(self):
        self.models: Dict[
            Tuple[str, str],
            ParameterAnomalyDetector,
        ] = {}

    # -----------------------------------------------------------------
    # Group key
    # -----------------------------------------------------------------

    @staticmethod
    def _key(
        component_type: Any,
        parameter_name: Any,
    ) -> Tuple[str, str]:
        return (
            str(component_type),
            str(parameter_name),
        )

    # -----------------------------------------------------------------
    # Fit all groups
    # -----------------------------------------------------------------

    def fit_all(
        self,
        df: pd.DataFrame,
        contamination: float = 0.10,
        n_estimators: int = 400,
        random_state: int = 42,
        verbose: bool = True,
    ) -> "ParameterModelRegistry":
        """
        Train one detector per component family.

        No defect labels are accessed.
        """
        validate_dataset(
            df
        )

        features = create_anomaly_features(
            df
        )

        self.models = {}

        grouped = features.groupby(
            GROUP_COLUMNS,
            dropna=False,
        )

        for (
            component_type,
            parameter_name,
        ), group in grouped:

            key = self._key(
                component_type,
                parameter_name,
            )

            detector = (
                ParameterAnomalyDetector(
                    component_type=str(
                        component_type
                    ),
                    parameter_name=str(
                        parameter_name
                    ),
                    contamination=contamination,
                    n_estimators=n_estimators,
                    random_state=random_state,
                )
            )

            detector.fit(
                group
            )

            self.models[key] = (
                detector
            )

            if verbose:
                print(
                    "Trained anomaly model:",
                    f"{component_type} / "
                    f"{parameter_name}",
                    f"| rows={len(group)}",
                    f"| core={detector.core_training_rows}",
                )

        return self

    # -----------------------------------------------------------------
    # Get model
    # -----------------------------------------------------------------

    def get_model(
        self,
        component_type: str,
        parameter_name: str,
    ) -> Optional[
        ParameterAnomalyDetector
    ]:
        """
        Retrieve detector for a component family.
        """
        key = self._key(
            component_type,
            parameter_name,
        )

        return self.models.get(
            key
        )

    # -----------------------------------------------------------------
    # Analyze one dataframe
    # -----------------------------------------------------------------

    def analyze(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Analyze all component groups.
        """
        if df.empty:
            return df.copy()

        validate_dataset(
            df
        )

        features = create_anomaly_features(
            df
        )

        outputs: List[
            pd.DataFrame
        ] = []

        for (
            component_type,
            parameter_name,
        ), group in features.groupby(
            GROUP_COLUMNS,
            dropna=False,
        ):
            key = self._key(
                component_type,
                parameter_name,
            )

            detector = self.models.get(
                key
            )

            if detector is None:
                raise KeyError(
                    "No anomaly model exists for "
                    f"{component_type} / "
                    f"{parameter_name}"
                )

            analyzed = detector.analyze(
                group
            )

            outputs.append(
                analyzed
            )

        if not outputs:
            return features.copy()

        result = pd.concat(
            outputs,
            axis=0,
        )

        return result.sort_index()

    # -----------------------------------------------------------------
    # Analyze one component
    # -----------------------------------------------------------------

    def analyze_component(
        self,
        row: pd.DataFrame | Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Analyze one component trajectory.

        Accepts either a one-row dataframe or a dictionary.
        """
        if isinstance(
            row,
            dict,
        ):
            input_df = pd.DataFrame(
                [row]
            )
        else:
            input_df = row.copy()

        if input_df.empty:
            raise ValueError(
                "Component data is empty."
            )

        validate_dataset(
            input_df
        )

        result = self.analyze(
            input_df
        )

        if result.empty:
            raise ValueError(
                "No anomaly result generated."
            )

        return result.iloc[
            0
        ].to_dict()

    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------

    def save(
        self,
        path: str | os.PathLike[str],
    ) -> str:
        """
        Save the entire registry to a pickle file.
        """
        output_path = Path(
            path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "version": self.VERSION,
            "models": self.models,
        }

        with open(
            output_path,
            "wb",
        ) as handle:
            pickle.dump(
                payload,
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        return str(
            output_path
        )

    # -----------------------------------------------------------------
    # Load
    # -----------------------------------------------------------------

    @classmethod
    def load(
        cls,
        path: str | os.PathLike[str],
    ) -> "ParameterModelRegistry":
        """
        Load a saved model registry.
        """
        model_path = Path(
            path
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"Anomaly model registry not found: "
                f"{model_path}"
            )

        with open(
            model_path,
            "rb",
        ) as handle:
            payload = pickle.load(
                handle
            )

        registry = cls()

        # -------------------------------------------------------------
        # Current registry format
        # -------------------------------------------------------------

        if isinstance(
            payload,
            dict,
        ) and "models" in payload:

            models = payload.get(
                "models",
                {},
            )

            for key, detector in models.items():

                if isinstance(
                    detector,
                    ParameterAnomalyDetector,
                ):
                    registry.models[
                        tuple(key)
                    ] = detector

                elif isinstance(
                    detector,
                    dict,
                ):
                    registry.models[
                        tuple(key)
                    ] = (
                        ParameterAnomalyDetector
                        .from_dict(
                            detector
                        )
                    )

            return registry

        # -------------------------------------------------------------
        # Backward compatibility:
        # dictionary directly containing models
        # -------------------------------------------------------------

        if isinstance(
            payload,
            dict,
        ):
            for key, detector in payload.items():

                if isinstance(
                    key,
                    tuple,
                ) and len(key) == 2:

                    if isinstance(
                        detector,
                        ParameterAnomalyDetector,
                    ):
                        registry.models[
                            tuple(key)
                        ] = detector

                    elif isinstance(
                        detector,
                        dict,
                    ):
                        registry.models[
                            tuple(key)
                        ] = (
                            ParameterAnomalyDetector
                            .from_dict(
                                detector
                            )
                        )

            if registry.models:
                return registry

        raise ValueError(
            "Unsupported anomaly model registry format."
        )

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------

    def summary(self) -> pd.DataFrame:
        """
        Return a compact model registry summary.
        """
        rows = []

        for (
            component_type,
            parameter_name,
        ), detector in sorted(
            self.models.items()
        ):
            rows.append(
                {
                    "component_type": component_type,
                    "parameter_name": parameter_name,
                    "training_rows": (
                        detector.training_rows
                    ),
                    "core_training_rows": (
                        detector.core_training_rows
                    ),
                    "contamination": (
                        detector.contamination
                    ),
                    "n_estimators": (
                        detector.n_estimators
                    ),
                    "score_median": (
                        detector.calibration.median
                    ),
                    "score_scale": (
                        detector.calibration.scale
                    ),
                    "threshold_raw": (
                        detector.calibration.threshold_raw
                    ),
                    "fitted": (
                        detector.fitted
                    ),
                }
            )

        return pd.DataFrame(
            rows
        )


# ---------------------------------------------------------------------
# Public training API
# ---------------------------------------------------------------------

def train_anomaly_models(
    dataset_path: str | os.PathLike[str] = DEFAULT_DATASET_PATH,
    model_path: str | os.PathLike[str] = DEFAULT_MODEL_PATH,
    contamination: float = 0.10,
    n_estimators: int = 400,
    random_state: int = 42,
    verbose: bool = True,
) -> ParameterModelRegistry:
    """
    Train and save the complete anomaly model registry.
    """
    dataset_path = Path(
        dataset_path
    )

    model_path = Path(
        model_path
    )

    print()
    print("=" * 70)
    print("AEGISBURN AI - ANOMALY MODEL TRAINING")
    print("=" * 70)
    print(
        "Dataset:",
        dataset_path,
    )

    df = load_dataset(
        dataset_path
    )

    print(
        "Rows loaded:",
        len(df),
    )

    print(
        "Columns:",
        list(df.columns),
    )

    # Explicitly report that labels are not being used.
    print(
        "Training mode: UNSUPERVISED"
    )

    print(
        "Ground-truth labels used: NO"
    )

    print(
        "168h measurement used by anomaly detector: NO"
    )

    registry = (
        ParameterModelRegistry()
        .fit_all(
            df,
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            verbose=verbose,
        )
    )

    saved_path = registry.save(
        model_path
    )

    print()
    print(
        "Anomaly model registry saved to:",
        saved_path,
    )

    print()
    print(
        registry.summary().to_string(
            index=False
        )
    )

    print()
    print("=" * 70)
    print("ANOMALY TRAINING COMPLETE")
    print("=" * 70)
    print()

    return registry


# ---------------------------------------------------------------------
# Public loading API
# ---------------------------------------------------------------------

def load_anomaly_models(
    model_path: str | os.PathLike[str] = DEFAULT_MODEL_PATH,
) -> ParameterModelRegistry:
    """
    Load the consolidated anomaly model registry.
    """
    return ParameterModelRegistry.load(
        model_path
    )


# ---------------------------------------------------------------------
# Convenience prediction API
# ---------------------------------------------------------------------

def analyze_dataframe(
    df: pd.DataFrame,
    registry: ParameterModelRegistry,
) -> pd.DataFrame:
    """
    Analyze a dataframe with an already-loaded registry.
    """
    return registry.analyze(
        df
    )


def analyze_component(
    component: Dict[str, Any],
    registry: ParameterModelRegistry,
) -> Dict[str, Any]:
    """
    Analyze one component dictionary.
    """
    return registry.analyze_component(
        component
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def build_argument_parser() -> argparse.ArgumentParser:
    """
    Build command-line argument parser.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Train AegisBurn AI unsupervised "
            "burn-in anomaly detection models."
        )
    )

    parser.add_argument(
        "--data",
        default=str(
            DEFAULT_DATASET_PATH
        ),
        help=(
            "Path to component_data.csv"
        ),
    )

    parser.add_argument(
        "--model",
        default=str(
            DEFAULT_MODEL_PATH
        ),
        help=(
            "Output anomaly_models.pkl path"
        ),
    )

    parser.add_argument(
        "--contamination",
        type=float,
        default=0.10,
        help=(
            "Isolation Forest contamination "
            "parameter."
        ),
    )

    parser.add_argument(
        "--estimators",
        type=int,
        default=400,
        help=(
            "Number of Isolation Forest trees."
        ),
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help=(
            "Random seed."
        ),
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help=(
            "Reduce training output."
        ),
    )

    return parser


def main() -> None:
    """
    CLI entry point.
    """
    parser = (
        build_argument_parser()
    )

    args = parser.parse_args()

    train_anomaly_models(
        dataset_path=args.data,
        model_path=args.model,
        contamination=args.contamination,
        n_estimators=args.estimators,
        random_state=args.random_state,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()