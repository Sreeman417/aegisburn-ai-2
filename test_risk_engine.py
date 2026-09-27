from src.preprocessing import (
    load_and_validate_data,
    preprocess_data,
)

from src.features import create_early_features

from src.anomaly_detection import (
    DynamicAnomalyDetector,
)

from src.drift_prediction import (
    DriftPredictor,
)

from src.risk_engine import RiskEngine


def main():

    df = load_and_validate_data(
        "data/raw/component_data.csv"
    )

    df = preprocess_data(df)
    df = create_early_features(df)

    # Train models
    anomaly_model = DynamicAnomalyDetector(
        contamination=0.20
    )
    anomaly_model.fit(df)

    df["anomaly_flag"] = anomaly_model.predict(df)
    df["anomaly_score"] = anomaly_model.predict_score(df)

    predictor = DriftPredictor()
    predictor.fit(df)

    predictions = predictor.predict(df)

    engine = RiskEngine()

    df["safety_slope"] = (
        engine.calculate_lot_safety_slope(df)
    )

    # ---------------------------------------------------------
    # Test representative component types
    # ---------------------------------------------------------

    selected = []

    for defect_type in [
        "NORMAL",
        "LATENT_DEFECT",
        "GRADUAL_DRIFT",
        "SUDDEN_ANOMALY",
    ]:

        matching = df[
            df["defect_type"] == defect_type
        ]

        if not matching.empty:
            selected.append(matching.iloc[0])

    print("\nREPRESENTATIVE RISK TEST")
    print("=" * 70)

    for row in selected:

        index = row.name

        result = engine.evaluate_component(
            row=row,
            predicted_168h=float(
                predictions[index]
            ),
            safety_slope=float(
                df.loc[index, "safety_slope"]
            ),
        )

        print(
            f"\nComponent: {row['component_id']}"
        )

        print(
            f"Type: {row['defect_type']}"
        )

        print(
            f"0h: {row['value_0h']:.4f}"
        )

        print(
            f"24h: {row['value_24h']:.4f}"
        )

        print(
            f"Actual 168h: "
            f"{row['value_168h']:.4f}"
        )

        print(
            f"Predicted 168h: "
            f"{result.predicted_168h:.4f}"
        )

        print(
            f"Anomaly flag: "
            f"{result.anomaly_flag}"
        )

        print(
            f"Early slope: "
            f"{result.early_slope:.6f}"
        )

        print(
            f"Predicted slope: "
            f"{result.predicted_slope:.6f}"
        )

        print(
            f"Safety slope: "
            f"{result.safety_slope:.6f}"
        )

        print(
            f"Risk score: "
            f"{result.risk_score:.2f}"
        )

        print(
            f"Risk level: "
            f"{result.risk_level}"
        )

        print("Reasons:")

        for reason in result.reasons:
            print(f"  - {reason}")


if __name__ == "__main__":
    main()
