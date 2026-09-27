from src.preprocessing import (
    load_and_validate_data,
    preprocess_data,
)
from src.drift_prediction import DriftPredictor


def main():
    df = load_and_validate_data(
        "data/raw/component_data.csv"
    )

    df = preprocess_data(df)

    predictor = DriftPredictor()

    results = predictor.fit(df)

    print("\nMODULE B — 168H PREDICTION")
    print("=" * 70)

    print(
        results["comparison"].to_string(
            index=False
        )
    )

    predictor.save(
        "models/drift_model.joblib",
        "models/drift_metadata.json",
    )

    print("\nSelected model:")
    print(predictor.model_name)

    print("\nSaved:")
    print("models/drift_model.joblib")
    print("models/drift_metadata.json")


if __name__ == "__main__":
    main()
