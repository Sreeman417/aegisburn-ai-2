from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
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
    "defect_type",
    "is_defective",
]

NUMERIC_COLUMNS = [
    "temperature_c",
    "value_0h",
    "value_24h",
    "value_96h",
    "value_168h",
]


def load_and_validate_data(
    path: str | Path,
) -> pd.DataFrame:
    """Load and validate the multi-parameter burn-in dataset."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(
            "Dataset is empty."
        )

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Preserve ALL columns from the CSV.
    df = df.copy()

    # ---------------------------------------------------------
    # Identifier / categorical validation
    # ---------------------------------------------------------

    for column in [
        "component_id",
        "component_type",
        "parameter_name",
        "unit",
        "lot_id",
        "defect_type",
    ]:
        df[column] = df[column].astype(str)

    if df["component_id"].duplicated().any():
        duplicates = (
            df.loc[
                df["component_id"].duplicated(),
                "component_id",
            ]
            .head(10)
            .tolist()
        )

        raise ValueError(
            "Duplicate component_id values found: "
            f"{duplicates}"
        )

    # ---------------------------------------------------------
    # Numeric validation
    # ---------------------------------------------------------

    for column in NUMERIC_COLUMNS:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        if df[column].isna().any():
            raise ValueError(
                f"Missing or invalid numeric values "
                f"found in '{column}'."
            )

        if (df[column] <= 0).any():
            raise ValueError(
                f"Non-positive values found in '{column}'."
            )

    # ---------------------------------------------------------
    # Ground-truth validation
    # ---------------------------------------------------------

    df["is_defective"] = pd.to_numeric(
        df["is_defective"],
        errors="coerce",
    )

    if df["is_defective"].isna().any():
        raise ValueError(
            "Invalid values found in 'is_defective'."
        )

    if not set(
        df["is_defective"].unique()
    ).issubset({0, 1}):
        raise ValueError(
            "'is_defective' must contain only 0 or 1."
        )

    # ---------------------------------------------------------
    # Basic categorical validation
    # ---------------------------------------------------------

    for column in [
        "component_type",
        "parameter_name",
        "unit",
        "lot_id",
    ]:
        if df[column].str.strip().eq("").any():
            raise ValueError(
                f"Empty values found in '{column}'."
            )

    return df


def preprocess_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Perform safe preprocessing while preserving the complete
    multi-parameter schema.
    """

    result = df.copy()

    # Remove accidental whitespace from categorical values.
    for column in [
        "component_id",
        "component_type",
        "parameter_name",
        "unit",
        "lot_id",
        "defect_type",
    ]:
        result[column] = (
            result[column]
            .astype(str)
            .str.strip()
        )

    # Ensure numeric columns are numeric.
    for column in NUMERIC_COLUMNS:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    # Sort consistently.
    result = result.sort_values(
        [
            "component_type",
            "parameter_name",
            "lot_id",
            "component_id",
        ]
    ).reset_index(drop=True)

    return result


def get_parameter_groups(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return the unique component-type / parameter / unit
    combinations in the dataset.
    """

    return (
        df[
            [
                "component_type",
                "parameter_name",
                "unit",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "component_type",
                "parameter_name",
            ]
        )
        .reset_index(drop=True)
    )
