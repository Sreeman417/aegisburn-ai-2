from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.preprocessing import (
    load_and_validate_data,
    preprocess_data,
)
from src.features import create_early_features
from src.risk_engine import RiskEngine


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "component_data.csv"

ANOMALY_MODEL_PATH = (
    PROJECT_ROOT / "models" / "anomaly_model.joblib"
)

DRIFT_MODEL_PATH = (
    PROJECT_ROOT / "models" / "drift_model.joblib"
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AegisBurn AI",
    page_icon="🚀",
    layout="wide",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 1.05rem;
        color: #666;
        margin-bottom: 1.5rem;
    }

    .risk-high {
        padding: 15px;
        border-radius: 10px;
        background-color: #ffe5e5;
        border-left: 6px solid #d62728;
    }

    .risk-review {
        padding: 15px;
        border-radius: 10px;
        background-color: #fff4d6;
        border-left: 6px solid #f0ad00;
    }

    .risk-normal {
        padding: 15px;
        border-radius: 10px;
        background-color: #e7f7e7;
        border-left: 6px solid #2ca02c;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_models():

    if not ANOMALY_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Anomaly model not found: "
            f"{ANOMALY_MODEL_PATH}"
        )

    if not DRIFT_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Drift model not found: "
            f"{DRIFT_MODEL_PATH}"
        )

    anomaly_model = joblib.load(
        ANOMALY_MODEL_PATH
    )

    drift_artifact = joblib.load(
        DRIFT_MODEL_PATH
    )

    return anomaly_model, drift_artifact

def display_risk(
    risk_level: str,
    score: float,
) -> None:
    """Display the component's final risk classification."""

    if risk_level == "HIGH RISK":

        st.markdown(
            f"""
            <div style="
                background-color:#ffe5e5;
                border-left:8px solid #d62728;
                border-radius:12px;
                padding:22px;
                margin-bottom:15px;
            ">
                <div style="font-size:28px;font-weight:700;color:#b00020;">
                    🔴 HIGH RISK
                </div>
                <div style="font-size:18px;margin-top:8px;">
                    Risk score:
                    <strong>{score:.2f}/100</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    elif risk_level == "REVIEW":

        st.markdown(
            f"""
            <div style="
                background-color:#fff4d6;
                border-left:8px solid #f0ad00;
                border-radius:12px;
                padding:22px;
                margin-bottom:15px;
            ">
                <div style="font-size:28px;font-weight:700;color:#9a6700;">
                    🟡 REVIEW REQUIRED
                </div>
                <div style="font-size:18px;margin-top:8px;">
                    Risk score:
                    <strong>{score:.2f}/100</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            f"""
            <div style="
                background-color:#e7f7e7;
                border-left:8px solid #2ca02c;
                border-radius:12px;
                padding:22px;
                margin-bottom:15px;
            ">
                <div style="font-size:28px;font-weight:700;color:#176b17;">
                    🟢 NORMAL
                </div>
                <div style="font-size:18px;margin-top:8px;">
                    Risk score:
                    <strong>{score:.2f}/100</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
# ============================================================
# RUN ANALYSIS
# ============================================================

def analyze_data(df: pd.DataFrame) -> pd.DataFrame:

    df = preprocess_data(df)

    df = create_early_features(df)

    # --------------------------------------------------------
    # Module A
    # --------------------------------------------------------

    anomaly_artifact = st.session_state.anomaly_model

    anomaly_model = anomaly_artifact[
        "model"
    ]

    anomaly_scaler = anomaly_artifact[
        "scaler"
    ]

    anomaly_features = anomaly_artifact[
        "feature_names"
    ]

    X_anomaly = df[
        anomaly_features
    ]

    X_anomaly_scaled = anomaly_scaler.transform(
        X_anomaly
    )

    raw_predictions = anomaly_model.predict(
        X_anomaly_scaled
    )

    raw_scores = anomaly_model.decision_function(
        X_anomaly_scaled
    )

    df["anomaly_flag"] = (
        raw_predictions == -1
    ).astype(int)

    df["anomaly_score"] = -raw_scores

    # --------------------------------------------------------
    # Module B
    # --------------------------------------------------------

    drift_artifact = st.session_state.drift_model

    drift_model = drift_artifact["model"]

    drift_features = drift_artifact[
        "feature_names"
    ]

    X_prediction = df[
        drift_features
    ]

    df["predicted_168h"] = (
        drift_model.predict(
            X_prediction
        )
    )

    # --------------------------------------------------------
    # Risk engine
    # --------------------------------------------------------

    engine = RiskEngine()

    df["safety_slope"] = (
        engine.calculate_lot_safety_slope(
            df
        )
    )

    results = []

    for index, row in df.iterrows():

        result = engine.evaluate_component(
            row=row,
            predicted_168h=float(
                row["predicted_168h"]
            ),
            safety_slope=float(
                row["safety_slope"]
            ),
        )

        results.append(
            {
                "risk_score": result.risk_score,
                "risk_level": result.risk_level,
                "predicted_slope": result.predicted_slope,
                "reasons": result.reasons,
            }
        )

    results_df = pd.DataFrame(results)

    df = pd.concat(
        [
            df.reset_index(drop=True),
            results_df.reset_index(drop=True),
        ],
        axis=1,
    )

    return df


# ============================================================
# CHART
# ============================================================

def create_component_chart(
    row: pd.Series,
) -> go.Figure:

    times = [
        0,
        24,
        96,
        168,
    ]

    actual_values = [
        row["value_0h"],
        row["value_24h"],
        row["value_96h"],
        row["value_168h"],
    ]

    predicted_168h = float(
        row["predicted_168h"]
    )

    fig = go.Figure()

    # Actual measurements
    fig.add_trace(
        go.Scatter(
            x=times,
            y=actual_values,
            mode="lines+markers",
            name="Measured",
        )
    )

    # Prediction
    fig.add_trace(
        go.Scatter(
            x=[24, 168],
            y=[
                row["value_24h"],
                predicted_168h,
            ],
            mode="lines+markers",
            name="AI predicted",
            line=dict(
                dash="dash",
            ),
        )
    )

    # Engineering limit
    fig.add_hline(
        y=50,
        line_dash="dot",
        annotation_text="Engineering limit",
    )

    fig.update_layout(
        title=(
            f"Burn-In Trajectory — "
            f"{row['component_id']}"
        ),
        xaxis_title="Burn-in time (hours)",
        yaxis_title="Parameter value (µA)",
        height=480,
        hovermode="x unified",
        legend=dict(
            orientation="h",
        ),
    )

    return fig


# ============================================================
# RISK DISPLAY
# ============================================================

def calculate_risk_indices(row: pd.Series) -> dict[str, float]:
    """Calculate interpretable risk indices for dashboard display."""

    # ---------------------------------------------------------
    # Anomaly Index
    # ---------------------------------------------------------
    anomaly_score = float(row["anomaly_score"])

    anomaly_index = min(
        max(
            anomaly_score * 300 + (
                50 if row["anomaly_flag"] else 0
            ),
            0,
        ),
        100,
    )

    # ---------------------------------------------------------
    # Early Drift Index
    # Compare observed early slope against safety slope.
    # ---------------------------------------------------------
    early_slope = float(row["slope_early"])
    safety_slope = float(row["safety_slope"])

    if safety_slope > 0:
        early_ratio = early_slope / safety_slope
    else:
        early_ratio = 1.0 if early_slope > 0 else 0.0

    early_drift_index = min(
        max(early_ratio * 100, 0),
        100,
    )

    # ---------------------------------------------------------
    # Future Drift Index
    # Compare predicted slope against safety slope.
    # ---------------------------------------------------------
    predicted_slope = float(
        row["predicted_slope"]
    )

    if safety_slope > 0:
        predicted_ratio = (
            predicted_slope / safety_slope
        )
    else:
        predicted_ratio = (
            1.0 if predicted_slope > 0 else 0.0
        )

    future_drift_index = min(
        max(predicted_ratio * 100, 0),
        100,
    )

    # ---------------------------------------------------------
    # Limit Utilization
    # ---------------------------------------------------------
    engineering_limit = 50.0

    predicted_168h = float(
        row["predicted_168h"]
    )

    limit_utilization = min(
        max(
            predicted_168h
            / engineering_limit
            * 100,
            0,
        ),
        100,
    )

    return {
        "anomaly_index": anomaly_index,
        "early_drift_index": early_drift_index,
        "future_drift_index": future_drift_index,
        "limit_utilization": limit_utilization,
    }


def classify_behavior_pattern(
    row: pd.Series,
) -> str:
    """
    Classify the observed trajectory into an interpretable
    behavior pattern.

    This is derived from measurements and model outputs.
    It does NOT use defect_type.
    """

    early_slope = float(
        row["slope_early"]
    )

    predicted_slope = float(
        row["predicted_slope"]
    )

    safety_slope = float(
        row["safety_slope"]
    )

    v0 = float(row["value_0h"])
    v24 = float(row["value_24h"])

    early_ratio = (
        v24 / v0
        if v0 > 0
        else 1.0
    )

    # Strong early jump
    if early_slope > safety_slope * 2:
        return "SUDDEN-ANOMALY-LIKE"

    # Early measurement is modestly elevated and future
    # predicted drift is considerably stronger.
    if (
        early_ratio > 1.05
        and predicted_slope > early_slope
    ):
        return "LATENT-DRIFT-LIKE"

    # Future drift exceeds the normal slope substantially.
    if predicted_slope > safety_slope:
        return "ACCELERATING-DRIFT"

    return "NORMAL-BEHAVIOR"


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🚀 AegisBurn AI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "AI-Driven Component Burn-In & Screening"
    "</div>",
    unsafe_allow_html=True,
)


st.info(
    "Prototype system: analyzes early burn-in measurements, "
    "detects dynamic anomalies, predicts 168h behavior, "
    "and assigns an explainable risk level."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Data Source"
)

uploaded_file = st.sidebar.file_uploader(
    "Upload burn-in CSV",
    type=["csv"],
)

use_demo = st.sidebar.checkbox(
    "Use demonstration dataset",
    value=True,
)

st.sidebar.divider()

st.sidebar.caption(
    "AegisBurn AI is a prototype. "
    "Engineering limits and thresholds must be "
    "validated by qualified hardware/QA teams."
)


# ============================================================
# LOAD MODELS
# ============================================================

try:

    anomaly_model, drift_model = load_models()

    st.session_state.anomaly_model = (
        anomaly_model
    )

    st.session_state.drift_model = (
        drift_model
    )

except Exception as error:

    st.error(
        f"Unable to load trained models:\n\n{error}"
    )

    st.stop()


# ============================================================
# LOAD DATA
# ============================================================

if uploaded_file is not None:

    try:

        raw_df = pd.read_csv(
            uploaded_file
        )

    except Exception as error:

        st.error(
            f"Unable to read uploaded CSV: {error}"
        )

        st.stop()

elif use_demo:

    try:

        raw_df = load_and_validate_data(
            DATA_PATH
        )

    except Exception as error:

        st.error(
            f"Unable to load demo dataset: {error}"
        )

        st.stop()

else:

    st.warning(
        "Upload a CSV or enable the demo dataset."
    )

    st.stop()


# ============================================================
# RUN ANALYSIS
# ============================================================

try:

    with st.spinner(
        "Running AI screening..."
    ):

        df = analyze_data(
            raw_df.copy()
        )

except Exception as error:

    st.error(
        f"Analysis failed: {error}"
    )

    st.stop()


# ============================================================
# KPI SECTION
# ============================================================

total = len(df)

normal_count = (
    df["risk_level"]
    == "NORMAL"
).sum()

review_count = (
    df["risk_level"]
    == "REVIEW"
).sum()

high_risk_count = (
    df["risk_level"]
    == "HIGH RISK"
).sum()

anomaly_count = (
    df["anomaly_flag"]
    == 1
).sum()


st.subheader(
    "Screening Overview"
)

col1, col2, col3, col4, col5 = (
    st.columns(5)
)

col1.metric(
    "Components",
    f"{total:,}",
)

col2.metric(
    "Normal",
    f"{normal_count:,}",
)

col3.metric(
    "Review",
    f"{review_count:,}",
)

col4.metric(
    "High Risk",
    f"{high_risk_count:,}",
)

col5.metric(
    "Anomalies",
    f"{anomaly_count:,}",
)


# ============================================================
# RISK DISTRIBUTION
# ============================================================

st.subheader(
    "Risk Distribution"
)

risk_counts = (
    df["risk_level"]
    .value_counts()
    .reindex(
        [
            "NORMAL",
            "REVIEW",
            "HIGH RISK",
        ],
        fill_value=0,
    )
)

fig_risk = go.Figure()

fig_risk.add_trace(
    go.Bar(
        x=risk_counts.index,
        y=risk_counts.values,
        text=risk_counts.values,
        textposition="auto",
    )
)

fig_risk.update_layout(
    xaxis_title="Risk level",
    yaxis_title="Number of components",
    height=400,
)

st.plotly_chart(
    fig_risk,
    use_container_width=True,
)


# ============================================================
# COMPONENT INSPECTOR
# ============================================================

# ============================================================
# RISK OVERVIEW
# ============================================================

st.subheader("🎯 Screening Risk Overview")

risk_col1, risk_col2, risk_col3 = st.columns(3)

with risk_col1:
    st.markdown(
        f"""
        <div class="risk-normal"
             style="text-align:center; min-height:125px;">
            <h2>🟢 {normal_count:,}</h2>
            <h4>NORMAL</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )

with risk_col2:
    st.markdown(
        f"""
        <div class="risk-review"
             style="text-align:center; min-height:125px;">
            <h2>🟡 {review_count:,}</h2>
            <h4>REVIEW</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )

with risk_col3:
    st.markdown(
        f"""
        <div class="risk-high"
             style="text-align:center; min-height:125px;">
            <h2>🔴 {high_risk_count:,}</h2>
            <h4>HIGH RISK</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# COMPONENT INSPECTOR
# ============================================================

st.subheader("🔍 Component Inspector")

component_ids = df["component_id"].tolist()

selected_id = st.selectbox(
    "Select component",
    component_ids,
)

selected = df[
    df["component_id"] == selected_id
].iloc[0]


risk_indices = calculate_risk_indices(
    selected
)

behavior_pattern = classify_behavior_pattern(
    selected
)


left, right = st.columns(
    [2.2, 1]
)


# ------------------------------------------------------------
# LEFT: TREND GRAPH
# ------------------------------------------------------------

with left:

    fig = create_component_chart(
        selected
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# ------------------------------------------------------------
# RIGHT: DECISION
# ------------------------------------------------------------

with right:

    display_risk(
        selected["risk_level"],
        selected["risk_score"],
    )

    st.markdown(
        f"### Behavioral Pattern"
    )

    st.info(
        behavior_pattern
    )

    st.write(
        f"**Component:** "
        f"{selected['component_id']}"
    )

    st.write(
        f"**Lot:** "
        f"{selected['lot_id']}"
    )


# ============================================================
# RISK FACTOR BREAKDOWN
# ============================================================

st.subheader(
    "📊 Why did the AI assign this risk?"
)

factor_col1, factor_col2 = st.columns(2)


with factor_col1:

    st.metric(
        "Anomaly Index",
        f"{risk_indices['anomaly_index']:.1f}/100",
    )

    st.metric(
        "Early Drift Index",
        f"{risk_indices['early_drift_index']:.1f}/100",
    )

with factor_col2:

    st.metric(
        "Future Drift Index",
        f"{risk_indices['future_drift_index']:.1f}/100",
    )

    st.metric(
        "Limit Utilization",
        f"{risk_indices['limit_utilization']:.1f}%",
    )


# ============================================================
# COMPONENT VALUES
# ============================================================

st.subheader(
    "🔬 Component Measurements"
)

m1, m2, m3, m4, m5 = st.columns(5)

m1.metric(
    "0h",
    f"{selected['value_0h']:.3f} µA",
)

m2.metric(
    "24h",
    f"{selected['value_24h']:.3f} µA",
)

m3.metric(
    "Predicted 168h",
    f"{selected['predicted_168h']:.3f} µA",
)

m4.metric(
    "Predicted slope",
    f"{selected['predicted_slope']:.4f}",
)

m5.metric(
    "Safety slope",
    f"{selected['safety_slope']:.4f}",
)


# ============================================================
# EXPLAINABILITY
# ============================================================

st.subheader(
    "💡 Why was this component flagged?"
)

reasons = selected[
    "reasons"
]

for reason in reasons:

    st.write(
        f"• {reason}"
    )


# ============================================================
# LOT ANALYSIS
# ============================================================

st.subheader(
    "🏭 Lot Analysis"
)

selected_lot = st.selectbox(
    "Select manufacturing lot",
    sorted(
        df["lot_id"].unique()
    ),
)

lot_df = df[
    df["lot_id"]
    == selected_lot
]

lot_col1, lot_col2, lot_col3 = (
    st.columns(3)
)

lot_col1.metric(
    "Lot components",
    len(lot_df),
)

lot_col2.metric(
    "Lot mean 0h",
    f"{lot_df['value_0h'].mean():.3f} µA",
)

lot_col3.metric(
    "Lot mean 24h",
    f"{lot_df['value_24h'].mean():.3f} µA",
)


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander(
    "🤖 Model Information"
):

    st.write(
        "**Module A — Dynamic Anomaly Detection**"
    )

    st.write(
        "Isolation Forest using early-stage "
        "measurements available at 24h."
    )

    st.write(
        "**Module B — 168h Prediction**"
    )

    st.write(
        "Regression model uses Value_0h and "
        "Value_24h to predict Value_168h."
    )

    st.write(
        f"Selected prediction model: "
        f"{st.session_state.drift_model['model_name']}"
    )


# ============================================================
# DOWNLOAD RESULTS
# ============================================================

st.subheader(
    "📥 Download Screening Results"
)

download_df = df.copy()

download_df[
    "reasons"
] = download_df[
    "reasons"
].apply(
    lambda x: " | ".join(x)
)

csv_data = download_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download analyzed CSV",
    data=csv_data,
    file_name="aegisburn_screening_results.csv",
    mime="text/csv",
)
