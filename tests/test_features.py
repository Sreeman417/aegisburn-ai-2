"""Tests for early/full feature engineering and leakage guards."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import (
    EARLY_FEATURE_COLUMNS,
    EARLY_HOURS,
    FULL_ONLY_FEATURE_COLUMNS,
    LEAKAGE_COLUMNS,
    _safe_divide,
    create_early_features,
    create_full_features,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "component_id": ["A", "B", "C", "D"],
            "lot_id": ["L1", "L1", "L1", "L2"],
            "value_0h": [10.0, 12.0, 11.0, 20.0],
            "value_24h": [10.5, 12.6, 11.2, 20.4],
            "value_96h": [11.0, 18.0, 11.5, 21.0],
            "value_168h": [11.2, 22.0, 11.8, 21.5],
            "defect_type": ["NORMAL", "LATENT_DEFECT", "NORMAL", "NORMAL"],
            "is_defective": [0, 1, 0, 0],
        }
    )


def test_drift_calculation() -> None:
    out = create_early_features(_frame())
    expected = out["value_24h"] - out["value_0h"]
    pd.testing.assert_series_equal(out["drift_0_24"], expected, check_names=False)


def test_slope_calculation() -> None:
    out = create_early_features(_frame())
    expected = (out["value_24h"] - out["value_0h"]) / EARLY_HOURS
    pd.testing.assert_series_equal(out["slope_early"], expected, check_names=False)


def test_lot_z_score_calculation() -> None:
    out = create_early_features(_frame())
    lot1 = out[out["lot_id"] == "L1"]
    mean_0 = lot1["value_0h"].mean()
    std_0 = lot1["value_0h"].std(ddof=0)
    expected_z = (lot1["value_0h"] - mean_0) / std_0
    np.testing.assert_allclose(lot1["z_score_0h"].to_numpy(), expected_z.to_numpy())
    np.testing.assert_allclose(lot1["z_score_0h"].mean(), 0.0, atol=1e-12)


def test_zero_division_protection() -> None:
    protected = _safe_divide(pd.Series([1.0, 2.0, 3.0]), pd.Series([0.0, 0.0, 2.0]))
    assert protected.iloc[0] == 0.0
    assert protected.iloc[1] == 0.0
    assert protected.iloc[2] == pytest.approx(1.5)

    zero_0h = _frame()
    zero_0h.loc[0, "value_0h"] = 0.0
    out = create_early_features(zero_0h)
    assert np.isfinite(out["ratio_24_0"]).all()
    assert out.loc[0, "ratio_24_0"] == 1.0

    constant_lot = pd.DataFrame(
        {
            "lot_id": ["L", "L"],
            "value_0h": [5.0, 5.0],
            "value_24h": [5.0, 5.0],
        }
    )
    z = create_early_features(constant_lot)
    assert (z["lot_std_0h"] == 0.0).all()
    assert (z["z_score_0h"] == 0.0).all()


def test_early_features_have_no_future_leakage() -> None:
    leaked = set(EARLY_FEATURE_COLUMNS) & set(LEAKAGE_COLUMNS)
    assert not leaked
    for col in ("value_96h", "value_168h", "defect_type", "is_defective"):
        assert col not in EARLY_FEATURE_COLUMNS
    for col in FULL_ONLY_FEATURE_COLUMNS:
        assert col not in EARLY_FEATURE_COLUMNS

    out = create_early_features(_frame())
    for col in EARLY_FEATURE_COLUMNS:
        assert col in out.columns
    for col in FULL_ONLY_FEATURE_COLUMNS:
        assert col not in out.columns


def test_full_features_include_later_windows() -> None:
    out = create_full_features(_frame())
    assert out.loc[0, "drift_24_96"] == pytest.approx(0.5)
    assert out.loc[0, "total_drift"] == pytest.approx(1.2)
    for col in FULL_ONLY_FEATURE_COLUMNS:
        assert col in out.columns
