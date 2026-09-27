from src.preprocessing import (
    load_and_validate_data,
    preprocess_data,
)
from src.features import create_early_features
from src.anomaly_detection import DynamicAnomalyDetector


def main():
    df = load_and_validate_data(
        "data/raw/component_data.csv"
    )

    df = preprocess_data(df)
    df = create_early_features(df)

    values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    print("\nIsolation Forest comparison")
    print("=" * 70)

    for contamination in values:

        model = DynamicAnomalyDetector(
            contamination=contamination,
            n_estimators=300,
            random_state=42,
        )

        model.fit(df)

        metrics = model.evaluate(df)

        print(
            f"Contamination: {contamination:.2f} | "
            f"Precision: {metrics['precision']:.3f} | "
            f"Recall: {metrics['recall']:.3f} | "
            f"F1: {metrics['f1']:.3f} | "
            f"FNR: {metrics['false_negative_rate']:.3f}"
        )


if __name__ == "__main__":
    main()
