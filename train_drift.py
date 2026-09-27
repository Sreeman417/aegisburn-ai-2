"""
AegisBurn AI -- Drift Prediction Model Training Entrypoint
=============================================================

Train (or retrain) the 168h drift-prediction regression models by
running THIS script:

    python train_drift.py

WHY THIS SCRIPT EXISTS:

There was previously no way to actually train and save these models
at all -- src/drift_prediction.py's __main__ block only printed a
message. As a result, backend/main.py's load_prediction_models()
always found no saved model directory and silently fell back to a
simple linear extrapolation (linear_prediction_fallback) for every
168h prediction shown on the live dashboard, regardless of any
improvements made to the real regression models.

This script fits PredictionModelRegistry on the production dataset
and saves it to models/prediction/, which is exactly where
backend/main.py's load_prediction_models() looks first.

Like train_anomaly.py, this script IMPORTS drift_prediction.py as a
module rather than ever executing it directly, so ParameterDriftPredictor
keeps its correct, portable module path (src.drift_prediction) in the
saved files instead of being tagged "__main__" (which would make it
unloadable from any other script, exactly like the anomaly detector
bug found earlier).
"""

from pathlib import Path

import pandas as pd

from src.drift_prediction import PredictionModelRegistry


PROJECT_ROOT = Path(__file__).resolve().parent

DATA_PATH = (
    PROJECT_ROOT
    / "src"
    / "data"
    / "raw"
    / "component_data.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "prediction"
)


def main() -> None:

    print("=" * 70)
    print("AEGISBURN AI - DRIFT PREDICTION MODEL TRAINING")
    print("=" * 70)

    print(f"Dataset: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    print(f"Rows loaded: {len(df)}")

    print(
        "Inputs used: value_0h, value_24h, value_96h "
        "(+ derived drift/ratio/acceleration features)"
    )

    print("Target: value_168h")
    print()

    registry = PredictionModelRegistry()

    registry.fit_all(df)

    for key, predictor in registry.models.items():

        component_type, parameter_name = key.split(
            "__",
            1,
        )

        print(
            f"{component_type} / {parameter_name}: "
            f"{predictor.model_name}  "
            f"MAE={predictor.mae:.4f}  "
            f"RMSE={predictor.rmse:.4f}  "
            f"R2={predictor.r2:.4f}"
        )

    registry.save(MODEL_DIR)

    print()
    print(f"Saved prediction model registry to: {MODEL_DIR}")
    print("=" * 70)
    print("DRIFT PREDICTION TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()