"""Tests for the 24h dynamic anomaly detector."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.anomaly_detection import DynamicAnomalyDetector
from src.features import EARLY_FEATURE_COLUMNS, LEAKAGE_COLUMNS, create_early_features


def _lot_with_outlier(n_normal: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    value_0h = rng.normal(10.0, 0.25, size=n_normal + 1)
    value_24h = value_0h + rng.normal(0.15, 0.05, size=n_normal + 1)
    # Last unit: sudden 24h jump, still below a hypothetical absolute limit.
    value_24h[-1] = value_0h[-1] + 4.5
    is_defective = np.zeros(n_normal + 1, dtype=int)
    is_defective[-1] = 1
    defect = np.array(["NORMAL"] * n_normal + ["SUDDEN_ANOMALY"])
    return pd.DataFrame(
        {
            "component_id": [f"C{i:03d}" for i in range(n_normal + 1)],
            "lot_id": ["LOT-X"] * (n_normal + 1),
            "temperature_c": np.full(n_normal + 1, 125.0),
            "value_0h": value_0h,
            "value_24h": value_24h,
            "value_96h": value_24h + 0.2,
            "value_168h": value_24h + 0.4,
            "defect_type": defect,
            "is_defective": is_defective,
        }
    )


def test_prediction_output_shape_and_labels() -> None:
    df = create_early_features(_lot_with_outlier())
    detector = DynamicAnomalyDetector(
        n_estimators=80,
        contamination=0.05,
        random_state=0,
        z_score_flag_threshold=2.4,
    )
    detector.fit(df)
    flags = detector.predict(df)
    scores = detector.predict_score(df)

    assert flags.shape == (len(df),)
    assert scores.shape == (len(df),)
    assert set(np.unique(flags)).issubset({0, 1})
    assert np.isfinite(scores).all()
    assert flags[-1] == 1
    assert scores[-1] >= np.median(scores)


def test_early_model_features_have_no_future_leakage() -> None:
    leaked = set(EARLY_FEATURE_COLUMNS) & set(LEAKAGE_COLUMNS)
    assert not leaked
    detector = DynamicAnomalyDetector(feature_names=list(EARLY_FEATURE_COLUMNS))
    detector._assert_no_leakage()


def test_evaluate_uses_labels_only_after_predict() -> None:
    df = create_early_features(_lot_with_outlier())
    detector = DynamicAnomalyDetector(n_estimators=80, contamination=0.08, random_state=0)
    detector.fit(df)
    metrics = detector.evaluate(df)
    assert {"precision", "recall", "f1", "false_negative_rate", "confusion_matrix"} <= set(
        metrics
    )
    cm = metrics["confusion_matrix"]
    assert cm["tp"] + cm["fn"] + cm["tn"] + cm["fp"] == len(df)


def test_save_and_load_roundtrip(tmp_path) -> None:
    df = create_early_features(_lot_with_outlier())
    detector = DynamicAnomalyDetector(n_estimators=50, contamination=0.08, random_state=1)
    detector.fit(df)
    path = tmp_path / "anomaly_model.joblib"
    detector.save(path)
    loaded = DynamicAnomalyDetector.load(path)
    np.testing.assert_array_equal(loaded.predict(df), detector.predict(df))
    np.testing.assert_allclose(loaded.predict_score(df), detector.predict_score(df))
