from __future__ import annotations

import numpy as np
import pandas as pd


def create_early_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features available at the 24-hour screening point.

    Features are calculated within a component/parameter/lot
    reference population where appropriate.

    IMPORTANT:
    value_96h and value_168h are never used in early features.
    """

    result = df.copy()

    # ---------------------------------------------------------
    # Basic early behaviour
    # ---------------------------------------------------------

    result["drift_0_24"] = (
        result["value_24h"]
        - result["value_0h"]
    )

    result["slope_early"] = (
        result["drift_0_24"] / 24.0
    )

    result["ratio_24_0"] = np.where(
        result["value_0h"] != 0,
        result["value_24h"] / result["value_0h"],
        np.nan,
    )

    # ---------------------------------------------------------
    # Reference population
    #
    # Do not compare unrelated parameters such as:
    # µA values against ns values.
    #
    # We therefore group by component_type,
    # parameter_name and lot_id.
    # ---------------------------------------------------------

    group_columns = [
        "component_type",
        "parameter_name",
        "lot_id",
    ]

    # 0h statistics
    result["reference_mean_0h"] = (
        result.groupby(group_columns)["value_0h"]
        .transform("mean")
    )

    result["reference_std_0h"] = (
        result.groupby(group_columns)["value_0h"]
        .transform("std")
        .fillna(0.0)
    )

    result["z_score_0h"] = np.where(
        result["reference_std_0h"] > 0,
        (
            result["value_0h"]
            - result["reference_mean_0h"]
        )
        / result["reference_std_0h"],
        0.0,
    )

    # 24h statistics
    result["reference_mean_24h"] = (
        result.groupby(group_columns)["value_24h"]
        .transform("mean")
    )

    result["reference_std_24h"] = (
        result.groupby(group_columns)["value_24h"]
        .transform("std")
        .fillna(0.0)
    )

    result["z_score_24h"] = np.where(
        result["reference_std_24h"] > 0,
        (
            result["value_24h"]
            - result["reference_mean_24h"]
        )
        / result["reference_std_24h"],
        0.0,
    )

    # Early slope statistics
    result["reference_mean_slope_early"] = (
        result.groupby(group_columns)["slope_early"]
        .transform("mean")
    )

    result["reference_std_slope_early"] = (
        result.groupby(group_columns)["slope_early"]
        .transform("std")
        .fillna(0.0)
    )

    result["z_score_slope_early"] = np.where(
        result["reference_std_slope_early"] > 0,
        (
            result["slope_early"]
            - result["reference_mean_slope_early"]
        )
        / result["reference_std_slope_early"],
        0.0,
    )

    return result


def create_full_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features requiring measurements beyond 24 hours."""

    result = df.copy()

    result["drift_24_96"] = (
        result["value_96h"]
        - result["value_24h"]
    )

    result["drift_96_168"] = (
        result["value_168h"]
        - result["value_96h"]
    )

    result["total_drift"] = (
        result["value_168h"]
        - result["value_0h"]
    )

    result["slope_mid"] = (
        result["drift_24_96"] / 72.0
    )

    result["slope_late"] = (
        result["drift_96_168"] / 72.0
    )

    result["slope_total"] = (
        result["total_drift"] / 168.0
    )

    return result


def create_all_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create early and full burn-in features."""

    result = create_early_features(df)

    result = create_full_features(result)

    return result
