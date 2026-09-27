"""
AEGISBURN AI — Complete Model Training

Trains:
    Module A — parameter-specific anomaly detection
    Module B — parameter-specific 168h prediction
"""

from __future__ import annotations

from pathlib import Path

from src.preprocessing import load_and_validate_data
from src.features import create_early_features
from src.anomaly_detection import ParameterModelRegistry
from src.drift_prediction import PredictionModelRegistry


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "component_data.csv"
)

ANOMALY_OUTPUT_DIR = (
    BASE_DIR
    / "models"
    / "anomaly"
)

PREDICTION_OUTPUT_DIR = (
    BASE_DIR
    / "models"
    / "prediction"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("AEGISBURN AI — PARAMETER-AWARE MODEL TRAINING")
    print("=" * 70)

    # ========================================================
    # LOAD DATA
    # ========================================================

    print()
    print("Loading dataset...")

    df = load_and_validate_data(
        DATA_PATH
    )

    print(
        f"Components: {len(df):,}"
    )

    print(
        "Component types:",
        df["component_type"].nunique()
    )

    print(
        "Parameters:",
        df["parameter_name"].nunique()
    )

    # ========================================================
    # FEATURES
    # ========================================================

    print()
    print("Creating early features...")

    feature_df = create_early_features(
        df
    )

    # ========================================================
    # MODULE A
    # ========================================================

    print()
    print("=" * 70)
    print("MODULE A — PARAMETER-AWARE ANOMALY DETECTION")
    print("=" * 70)

    anomaly_registry = (
        ParameterModelRegistry()
    )

    anomaly_registry.fit_all(
        feature_df,
        contamination=0.10,
        n_estimators=300,
        random_state=42,
    )

    anomaly_registry.save(
        ANOMALY_OUTPUT_DIR
    )

    print(
        f"Created "
        f"{len(anomaly_registry.models)} "
        f"anomaly models."
    )

    # ========================================================
    # MODULE B
    # ========================================================

    print()
    print("=" * 70)
    print("MODULE B — PARAMETER-AWARE 168H PREDICTION")
    print("=" * 70)

    prediction_registry = (
        PredictionModelRegistry()
    )

    prediction_registry.fit_all(
        feature_df,
        random_state=42,
    )

    # --------------------------------------------------------
    # Display model metrics
    # --------------------------------------------------------

    for key in sorted(
        prediction_registry.models.keys()
    ):

        predictor = (
            prediction_registry.models[key]
        )

        print()
        print(key)

        print(
            f"Selected model : "
            f"{predictor.model_name}"
        )

        print(
            f"MAE            : "
            f"{predictor.mae:.6f}"
        )

        print(
            f"RMSE           : "
            f"{predictor.rmse:.6f}"
        )

        print(
            f"R²             : "
            f"{predictor.r2:.6f}"
        )

    # --------------------------------------------------------
    # Save prediction registry
    # --------------------------------------------------------

    prediction_registry.save(
        PREDICTION_OUTPUT_DIR
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Anomaly models saved to: "
        f"{ANOMALY_OUTPUT_DIR}"
    )

    print(
        f"Prediction models saved to: "
        f"{PREDICTION_OUTPUT_DIR}"
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()