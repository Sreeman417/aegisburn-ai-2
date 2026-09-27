from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_OUTPUT = Path("data/raw/component_data.csv")
DEFAULT_COMPONENTS = 10_000
DEFAULT_LOTS = 20
DEFAULT_SEED = 42


# =============================================================================
# PARAMETER PROFILES
# =============================================================================

PARAMETER_PROFILES = [
    {
        "component_type": "Logic IC",
        "parameter_name": "Iddq",
        "unit": "µA",
        "base_mean": 10.0,
        "base_std": 0.25,
        "normal_step": 0.25,
        "engineering_limit": 50.0,
    },
    {
        "component_type": "Memory IC",
        "parameter_name": "Standby Current",
        "unit": "µA",
        "base_mean": 25.0,
        "base_std": 0.50,
        "normal_step": 0.40,
        "engineering_limit": 80.0,
    },
    {
        "component_type": "ADC",
        "parameter_name": "Leakage Current",
        "unit": "µA",
        "base_mean": 5.0,
        "base_std": 0.15,
        "normal_step": 0.15,
        "engineering_limit": 25.0,
    },
    {
        "component_type": "Driver IC",
        "parameter_name": "Propagation Delay",
        "unit": "ns",
        "base_mean": 18.0,
        "base_std": 0.40,
        "normal_step": 0.20,
        "engineering_limit": 40.0,
    },
]


# =============================================================================
# DEFECT DISTRIBUTION
#
# Increased defective population from 15% to 45%.
# =============================================================================

DEFECT_TYPES = [
    "NORMAL",
    "GRADUAL_DRIFT",
    "LATENT_DEFECT",
    "SUDDEN_ANOMALY",
]

DEFECT_PROBABILITIES = [
    0.55,  # NORMAL
    0.20,  # GRADUAL_DRIFT
    0.15,  # LATENT_DEFECT
    0.10,  # SUDDEN_ANOMALY
]


# =============================================================================
# LOT GENERATION
# =============================================================================

def generate_lot_offsets(
    rng: np.random.Generator,
    num_lots: int,
) -> dict[str, float]:

    offsets = {}

    for lot_number in range(1, num_lots + 1):

        lot_id = f"LOT-{lot_number:02d}"

        offsets[lot_id] = float(
            rng.normal(
                0.0,
                0.20,
            )
        )

    return offsets


# =============================================================================
# NOISE
# =============================================================================

def add_noise(
    rng: np.random.Generator,
    value: float,
    noise_std: float,
) -> float:

    result = value + rng.normal(
        0.0,
        noise_std,
    )

    return float(
        max(
            result,
            0.001,
        )
    )


# =============================================================================
# COMPONENT GENERATION
# =============================================================================

def generate_component(
    rng: np.random.Generator,
    component_number: int,
    lot_id: str,
    lot_offset: float,
    profile: dict[str, object],
) -> dict[str, object]:

    component_type = str(
        profile["component_type"]
    )

    parameter_name = str(
        profile["parameter_name"]
    )

    unit = str(
        profile["unit"]
    )

    base_mean = float(
        profile["base_mean"]
    )

    base_std = float(
        profile["base_std"]
    )

    normal_step = float(
        profile["normal_step"]
    )

    component_id = (
        f"CMP-{component_number:06d}"
    )

    # -------------------------------------------------------------------------
    # COMPONENT BASE VALUE
    # -------------------------------------------------------------------------

    base = (
        base_mean
        + lot_offset
        + rng.normal(
            0.0,
            base_std,
        )
    )

    # -------------------------------------------------------------------------
    # DEFECT TYPE SELECTION
    # -------------------------------------------------------------------------

    defect_type = str(
        rng.choice(
            DEFECT_TYPES,
            p=DEFECT_PROBABILITIES,
        )
    )

    # -------------------------------------------------------------------------
    # TEMPERATURE
    # -------------------------------------------------------------------------

    temperature = float(
        rng.normal(
            125.0,
            1.5,
        )
    )

    # =========================================================================
    # NORMAL
    # =========================================================================

    if defect_type == "NORMAL":

        v0 = base

        v24 = (
            v0
            + rng.normal(
                normal_step,
                normal_step * 0.35,
            )
        )

        v96 = (
            v24
            + rng.normal(
                normal_step,
                normal_step * 0.35,
            )
        )

        v168 = (
            v96
            + rng.normal(
                normal_step,
                normal_step * 0.35,
            )
        )

        is_defective = 0


    # =========================================================================
    # GRADUAL DRIFT
    #
    # Increased drift severity so risk engine can identify more components.
    # =========================================================================

    elif defect_type == "GRADUAL_DRIFT":

        v0 = base

        v24 = (
            v0
            + rng.uniform(
                normal_step * 6,
                normal_step * 12,
            )
        )

        v96 = (
            v24
            + rng.uniform(
                normal_step * 20,
                normal_step * 40,
            )
        )

        v168 = (
            v96
            + rng.uniform(
                normal_step * 40,
                normal_step * 80,
            )
        )

        is_defective = 1


    # =========================================================================
    # LATENT DEFECT
    #
    # Mild early drift but strong later deterioration.
    # =========================================================================

    elif defect_type == "LATENT_DEFECT":

        v0 = base

        v24 = (
            v0
            + rng.uniform(
                normal_step * 3,
                normal_step * 8,
            )
        )

        v96 = (
            v24
            + rng.uniform(
                normal_step * 25,
                normal_step * 50,
            )
        )

        v168 = (
            v96
            + rng.uniform(
                normal_step * 50,
                normal_step * 100,
            )
        )

        is_defective = 1


    # =========================================================================
    # SUDDEN ANOMALY
    #
    # Very large abnormal change during early burn-in.
    # =========================================================================

    elif defect_type == "SUDDEN_ANOMALY":

        v0 = base

        v24 = (
            v0
            + rng.uniform(
                normal_step * 30,
                normal_step * 70,
            )
        )

        # Sudden anomaly may partially recover later,
        # but early behavior remains abnormal.

        v96 = (
            base
            + rng.uniform(
                normal_step * 2,
                normal_step * 10,
            )
        )

        v168 = (
            v96
            + rng.uniform(
                normal_step * 2,
                normal_step * 15,
            )
        )

        is_defective = 1


    else:

        raise ValueError(
            f"Unknown defect type: {defect_type}"
        )


    # =========================================================================
    # MEASUREMENT NOISE
    # =========================================================================

    measurement_noise = max(
        abs(base) * 0.01,
        0.01,
    )

    v0 = add_noise(
        rng,
        v0,
        measurement_noise,
    )

    v24 = add_noise(
        rng,
        v24,
        measurement_noise,
    )

    v96 = add_noise(
        rng,
        v96,
        measurement_noise,
    )

    v168 = add_noise(
        rng,
        v168,
        measurement_noise,
    )


    # =========================================================================
    # RETURN COMPONENT
    # =========================================================================

    return {

        "component_id": component_id,

        "component_type": component_type,

        "parameter_name": parameter_name,

        "unit": unit,

        "lot_id": lot_id,

        "temperature_c": round(
            temperature,
            2,
        ),

        "value_0h": round(
            v0,
            4,
        ),

        "value_24h": round(
            v24,
            4,
        ),

        "value_96h": round(
            v96,
            4,
        ),

        "value_168h": round(
            v168,
            4,
        ),

        "defect_type": defect_type,

        "is_defective": is_defective,
    }


# =============================================================================
# DATASET GENERATION
# =============================================================================

def generate_dataset(
    num_components: int = DEFAULT_COMPONENTS,
    num_lots: int = DEFAULT_LOTS,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:

    if num_components <= 0:

        raise ValueError(
            "num_components must be greater than zero."
        )

    if num_lots <= 0:

        raise ValueError(
            "num_lots must be greater than zero."
        )

    rng = np.random.default_rng(
        seed
    )

    lot_offsets = generate_lot_offsets(
        rng,
        num_lots,
    )

    lot_ids = list(
        lot_offsets.keys()
    )

    rows = []

    print()
    print("=" * 70)
    print("GENERATING AEGISBURN COMPONENT DATASET")
    print("=" * 70)
    print(
        f"Components to generate: {num_components:,}"
    )
    print(
        f"Lots: {num_lots}"
    )
    print()
    print("Defect distribution:")
    print("  NORMAL          : 55%")
    print("  GRADUAL_DRIFT   : 20%")
    print("  LATENT_DEFECT   : 15%")
    print("  SUDDEN_ANOMALY  : 10%")
    print("=" * 70)
    print()

    for component_number in range(
        1,
        num_components + 1,
    ):

        profile = PARAMETER_PROFILES[
            (component_number - 1)
            % len(PARAMETER_PROFILES)
        ]

        lot_id = str(
            rng.choice(
                lot_ids
            )
        )

        row = generate_component(
            rng=rng,
            component_number=component_number,
            lot_id=lot_id,
            lot_offset=lot_offsets[
                lot_id
            ],
            profile=profile,
        )

        rows.append(
            row
        )

        # Progress indicator.

        if (
            component_number % 1000 == 0
            or component_number == num_components
        ):

            print(
                f"Generated: "
                f"{component_number:,}"
                f"/{num_components:,}"
            )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# VALIDATION
# =============================================================================

def validate_dataset(
    df: pd.DataFrame,
) -> None:

    required_columns = {

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
    }

    missing = (
        required_columns
        - set(
            df.columns
        )
    )

    if missing:

        raise ValueError(
            f"Missing columns: "
            f"{sorted(missing)}"
        )

    if df[
        "component_id"
    ].duplicated().any():

        raise ValueError(
            "Duplicate component IDs detected."
        )

    numeric_columns = [

        "temperature_c",

        "value_0h",

        "value_24h",

        "value_96h",

        "value_168h",
    ]

    for column in numeric_columns:

        if df[
            column
        ].isna().any():

            raise ValueError(
                f"Missing values in {column}."
            )

        if not pd.api.types.is_numeric_dtype(
            df[column]
        ):

            raise ValueError(
                f"{column} must be numeric."
            )

        if (
            df[column] <= 0
        ).any():

            raise ValueError(
                f"Non-positive values in {column}."
            )

    if not set(
        df[
            "is_defective"
        ].unique()
    ).issubset(
        {0, 1}
    ):

        raise ValueError(
            "is_defective must contain only 0 and 1."
        )


# =============================================================================
# DATASET REPORT
# =============================================================================

def print_dataset_report(
    df: pd.DataFrame,
) -> None:

    print()

    print("=" * 70)
    print(
        "AEGISBURN AI - DATASET REPORT"
    )
    print("=" * 70)

    print(
        f"Total components : "
        f"{len(df):,}"
    )

    print(
        f"Component types  : "
        f"{df['component_type'].nunique()}"
    )

    print(
        f"Parameters       : "
        f"{df['parameter_name'].nunique()}"
    )

    print(
        f"Lots             : "
        f"{df['lot_id'].nunique()}"
    )

    defective = int(
        df[
            "is_defective"
        ].sum()
    )

    defect_percentage = (
        defective
        / len(df)
        * 100
    )

    print(
        f"Defective units  : "
        f"{defective:,} "
        f"({defect_percentage:.2f}%)"
    )

    print()

    print(
        "Defect distribution"
    )

    print("-" * 70)

    print(
        df[
            "defect_type"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()

    print(
        "Defective component examples"
    )

    print("-" * 70)

    print(
        df[
            df["is_defective"] == 1
        ]
        .head(20)
        .to_string(
            index=False
        )
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    parser = argparse.ArgumentParser(

        description=(
            "Generate synthetic multi-parameter "
            "component burn-in data."
        )
    )

    parser.add_argument(

        "--components",

        type=int,

        default=DEFAULT_COMPONENTS,
    )

    parser.add_argument(

        "--lots",

        type=int,

        default=DEFAULT_LOTS,
    )

    parser.add_argument(

        "--seed",

        type=int,

        default=DEFAULT_SEED,
    )

    parser.add_argument(

        "--output",

        type=Path,

        default=DEFAULT_OUTPUT,
    )

    args = parser.parse_args()


    # -------------------------------------------------------------------------
    # GENERATE
    # -------------------------------------------------------------------------

    df = generate_dataset(

        num_components=args.components,

        num_lots=args.lots,

        seed=args.seed,
    )


    # -------------------------------------------------------------------------
    # VALIDATE
    # -------------------------------------------------------------------------

    print()
    print(
        "Validating dataset..."
    )

    validate_dataset(
        df
    )

    print(
        "Dataset validation successful."
    )


    # -------------------------------------------------------------------------
    # SAVE
    # -------------------------------------------------------------------------

    args.output.parent.mkdir(

        parents=True,

        exist_ok=True,
    )

    df.to_csv(

        args.output,

        index=False,
    )


    # -------------------------------------------------------------------------
    # REPORT
    # -------------------------------------------------------------------------

    print_dataset_report(
        df
    )


    print()

    print("=" * 70)

    print(
        "DATASET SAVED SUCCESSFULLY"
    )

    print("=" * 70)

    print(
        args.output.resolve()
    )


if __name__ == "__main__":

    main()
