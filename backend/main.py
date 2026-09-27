from __future__ import annotations

import io
import sys
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = BASE_DIR / "backend"
STATIC_DIR = BACKEND_DIR / "static"

DATA_DIR = BASE_DIR / "src" / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

MODELS_DIR = BASE_DIR / "models"
ANOMALY_MODEL_DIR = MODELS_DIR / "anomaly"
PREDICTION_MODEL_DIR = MODELS_DIR / "prediction"

DEFAULT_DATASET = RAW_DATA_DIR / "component_data.csv"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


# =============================================================================
# PROJECT IMPORTS
# =============================================================================

try:
    from src.anomaly_detection import ParameterModelRegistry
except Exception as error:
    print(f"WARNING: Could not import anomaly detector: {error}")
    ParameterModelRegistry = None


try:
    from src.drift_prediction import PredictionModelRegistry
except Exception as error:
    print(f"WARNING: Could not import prediction models: {error}")
    PredictionModelRegistry = None


try:
    from src.risk_engine import RiskEngine
except Exception as error:
    print(f"WARNING: Could not import risk engine: {error}")
    RiskEngine = None


# =============================================================================
# GLOBAL STATE
# =============================================================================

COMPONENT_DATA: pd.DataFrame | None = None
FEATURE_DATA: pd.DataFrame | None = None

ANOMALY_MODELS: Any = None
PREDICTION_MODELS: Any = None
RISK_ENGINE: Any = None

ACTIVE_DATASET_NAME = DEFAULT_DATASET.name
ACTIVE_DATASET_PATH = DEFAULT_DATASET

ANALYSIS_CACHE: dict[str, dict[str, Any]] = {}


# =============================================================================
# REQUEST MODELS
# =============================================================================

class AnalyzeRequest(BaseModel):
    component_id: str


class UploadCSVRequest(BaseModel):
    filename: str
    csv_text: str


# =============================================================================
# ENGINEERING LIMITS
# =============================================================================
#
# Prototype engineering limits.
# These are NOT official ISRO specifications.
# They are used only by the prototype risk engine.
# =============================================================================

ENGINEERING_LIMITS = {
    "iddq": 50.0,
    "standby current": 80.0,
    "leakage current": 25.0,
    "propagation delay": 40.0,
}


def get_engineering_limit(parameter_name: Any) -> float | None:
    if parameter_name is None:
        return None

    name = str(parameter_name).strip().lower()

    if name in ENGINEERING_LIMITS:
        return ENGINEERING_LIMITS[name]

    for parameter, limit in ENGINEERING_LIMITS.items():
        if parameter in name:
            return float(limit)

    return None


# =============================================================================
# TEXT CLEANING
# =============================================================================

def clean_text(value: Any) -> str:
    """
    Fix common UTF-8/Windows-1252 mojibake seen in uploaded CSV/API data.
    """

    if value is None:
        return ""

    text = str(value)

    replacements = {
        "Âµ": "µ",
        "Î¼": "µ",
        "â€“": "–",
        "â€”": "—",
        "â€˜": "‘",
        "â€™": "’",
        "â€œ": "“",
        "â€": "”",
        "â€¦": "…",
        "â€¢": "•",
        "â†’": "→",
        "â‰¥": "≥",
        "â‰¤": "≤",
    }

    for bad, good in replacements.items():
        text = text.replace(bad, good)

    return text


# =============================================================================
# JSON HELPERS
# =============================================================================

def json_safe(value: Any) -> Any:

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, float):
        if not np.isfinite(value):
            return None

    if isinstance(value, str):
        return clean_text(value)

    return value


def clean_record(record: dict[str, Any]) -> dict[str, Any]:

    cleaned: dict[str, Any] = {}

    for key, value in record.items():

        if isinstance(value, dict):
            cleaned[key] = {
                str(k): json_safe(v)
                for k, v in value.items()
            }

        elif isinstance(value, (list, tuple)):
            cleaned[key] = [
                json_safe(item)
                for item in value
            ]

        else:
            cleaned[key] = json_safe(value)

    return cleaned


# =============================================================================
# COLUMN NORMALIZATION
# =============================================================================

def normalize_column_name(column: str) -> str:

    return (
        str(column)
        .strip()
        .lower()
        .replace("\ufeff", "")
        .replace(" ", "_")
        .replace("-", "_")
        .replace(".", "_")
    )


def normalize_dataset_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    df.columns = [
        normalize_column_name(column)
        for column in df.columns
    ]

    aliases = {
        "componentid": "component_id",
        "component": "component_id",
        "id": "component_id",

        "componenttype": "component_type",
        "type": "component_type",

        "parameter": "parameter_name",
        "parametername": "parameter_name",

        "lot": "lot_id",
        "lotid": "lot_id",

        "temperature": "temperature_c",
        "temperaturec": "temperature_c",

        "value0h": "value_0h",
        "value0": "value_0h",
        "value_0": "value_0h",
        "0h": "value_0h",

        "value24h": "value_24h",
        "value24": "value_24h",
        "value_24": "value_24h",
        "24h": "value_24h",

        "value96h": "value_96h",
        "value96": "value_96h",
        "value_96": "value_96h",
        "96h": "value_96h",

        "value168h": "value_168h",
        "value168": "value_168h",
        "value_168": "value_168h",
        "168h": "value_168h",

        "defect": "defect_type",
        "defecttype": "defect_type",

        "defective": "is_defective",
        "isdefective": "is_defective",
    }

    rename_map: dict[str, str] = {}

    for column in df.columns:

        compact = column.replace("_", "")

        if column in aliases:
            rename_map[column] = aliases[column]

        elif compact in aliases:
            rename_map[column] = aliases[compact]

    df = df.rename(columns=rename_map)

    return df


# =============================================================================
# DATASET VALIDATION
# =============================================================================

def validate_dataset(
    df: pd.DataFrame,
) -> tuple[bool, list[str]]:

    required = [
        "component_id",
        "component_type",
        "parameter_name",
        "value_0h",
        "value_24h",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    return len(missing) == 0, missing


# =============================================================================
# DATASET PREPARATION
# =============================================================================

def prepare_dataset(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = normalize_dataset_columns(df)

    valid, missing = validate_dataset(df)

    if not valid:
        raise ValueError(
            f"Dataset is missing required columns: {missing}"
        )

    defaults = {
        "lot_id": "UNKNOWN",
        "temperature_c": 25.0,
        "unit": "",
        "value_96h": np.nan,
        "value_168h": np.nan,
    }

    for column, default in defaults.items():

        if column not in df.columns:
            df[column] = default

    string_columns = [
        "component_id",
        "component_type",
        "parameter_name",
        "lot_id",
        "unit",
    ]

    for column in string_columns:

        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
            .map(clean_text)
        )

    numeric_columns = [
        "temperature_c",
        "value_0h",
        "value_24h",
        "value_96h",
        "value_168h",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "component_id",
            "component_type",
            "parameter_name",
            "value_0h",
            "value_24h",
        ]
    )

    df = df.reset_index(drop=True)

    return df


def is_valid_dataset(path: Path) -> bool:

    try:

        sample = pd.read_csv(
            path,
            nrows=5,
        )

        sample = normalize_dataset_columns(sample)

        valid, _ = validate_dataset(sample)

        return valid

    except Exception:
        return False


def find_dataset() -> Path:

    if (
        DEFAULT_DATASET.exists()
        and is_valid_dataset(DEFAULT_DATASET)
    ):
        return DEFAULT_DATASET

    valid_files: list[Path] = []

    for path in BASE_DIR.rglob("*.csv"):

        parts_lower = {
            part.lower()
            for part in path.parts
        }

        if "models" in parts_lower:
            continue

        if "registry" in path.name.lower():
            continue

        if is_valid_dataset(path):
            valid_files.append(path)

    if not valid_files:

        raise FileNotFoundError(
            "No valid AegisBurn dataset was found."
        )

    valid_files.sort(
        key=lambda p: p.stat().st_size,
        reverse=True,
    )

    return valid_files[0]


def load_dataset_from_path(
    path: Path,
) -> pd.DataFrame:

    df = pd.read_csv(path)

    return prepare_dataset(df)


def load_dataset() -> pd.DataFrame:

    global ACTIVE_DATASET_NAME
    global ACTIVE_DATASET_PATH

    path = find_dataset()

    print("=" * 70)
    print("AEGISBURN AI DATASET")
    print("=" * 70)
    print(f"Dataset: {path}")

    df = load_dataset_from_path(path)

    ACTIVE_DATASET_NAME = path.name
    ACTIVE_DATASET_PATH = path

    print(f"Rows loaded: {len(df):,}")
    print("=" * 70)

    return df


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def create_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    features = df.copy()

    # -------------------------------------------------------------------------
    # Basic burn-in drift
    # -------------------------------------------------------------------------

    features["drift_0_24"] = (
        features["value_24h"]
        - features["value_0h"]
    )

    features["drift_24_96"] = (
        features["value_96h"]
        - features["value_24h"]
    )

    features["drift_96_168"] = (
        features["value_168h"]
        - features["value_96h"]
    )

    features["drift_0_168"] = (
        features["value_168h"]
        - features["value_0h"]
    )

    # -------------------------------------------------------------------------
    # Slopes
    # -------------------------------------------------------------------------

    features["slope_early"] = (
        features["drift_0_24"] / 24.0
    )

    features["slope_mid"] = (
        features["drift_24_96"] / 72.0
    )

    features["slope_late"] = (
        features["drift_96_168"] / 72.0
    )

    # -------------------------------------------------------------------------
    # Ratios
    # -------------------------------------------------------------------------

    value_0 = (
        features["value_0h"]
        .replace(0, np.nan)
    )

    value_24 = (
        features["value_24h"]
        .replace(0, np.nan)
    )

    value_96 = (
        features["value_96h"]
        .replace(0, np.nan)
    )

    features["ratio_24_0"] = (
        features["value_24h"]
        / value_0
    )

    features["ratio_96_24"] = (
        features["value_96h"]
        / value_24
    )

    features["ratio_168_96"] = (
        features["value_168h"]
        / value_96
    )

    features["ratio_168_0"] = (
        features["value_168h"]
        / value_0
    )

    # -------------------------------------------------------------------------
    # Acceleration
    # -------------------------------------------------------------------------

    features["acceleration"] = (
        features["slope_late"]
        - features["slope_early"]
    )

    early_abs = (
        features["drift_0_24"]
        .abs()
        .replace(0, np.nan)
    )

    features["late_growth_ratio"] = (
        features["drift_96_168"]
        .abs()
        / early_abs
    )

    # -------------------------------------------------------------------------
    # Group z-scores
    # -------------------------------------------------------------------------

    grouped = features.groupby(
        [
            "component_type",
            "parameter_name",
        ]
    )

    for column, output_column in [
        (
            "value_0h",
            "z_score_0h",
        ),
        (
            "value_24h",
            "z_score_24h",
        ),
        (
            "slope_early",
            "z_score_slope_early",
        ),
    ]:

        group_mean = grouped[column].transform(
            "mean"
        )

        group_std = grouped[column].transform(
            "std"
        )

        group_std = group_std.replace(
            0,
            np.nan,
        )

        z_score = (
            features[column]
            - group_mean
        ) / group_std

        features[output_column] = (
            z_score
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .fillna(0.0)
        )

    return features


# =============================================================================
# MODEL HELPERS
# =============================================================================

def count_models(
    registry: Any,
) -> int:

    if registry is None:
        return 0

    models = getattr(
        registry,
        "models",
        None,
    )

    if isinstance(models, dict):
        return len(models)

    return 0


def try_registry_load(
    registry_class: Any,
    model_dir: Path,
) -> Any:

    if registry_class is None:
        return None

    if not model_dir.exists():
        return None

    load_method = getattr(
        registry_class,
        "load",
        None,
    )

    if not callable(load_method):
        return None

    for argument in [
        str(model_dir),
        model_dir,
    ]:

        try:
            return load_method(argument)

        except Exception:
            continue

    return None


def find_model_directory(
    preferred: Path,
    alternatives: list[Path],
) -> Path | None:

    candidates = [
        preferred,
        *alternatives,
    ]

    for directory in candidates:

        if directory.exists():
            return directory

    return None


def load_anomaly_models() -> Any:
    """
    Load the current AegisBurn anomaly model registry.

    Preferred model:
        models/anomaly/anomaly_models.pkl

    The legacy directory-based loader is kept as a fallback so older
    model files do not break the application.
    """

    # ---------------------------------------------------------
    # 1. Prefer the new consolidated anomaly registry
    # ---------------------------------------------------------
    registry_path = ANOMALY_MODEL_DIR / "anomaly_models.pkl"

    if registry_path.exists():
        try:
            models = ParameterModelRegistry.load(str(registry_path))

            if models is not None:
                print(
                    "Anomaly models loaded from:",
                    registry_path,
                )
                print(
                    "Anomaly model groups:",
                    count_models(models),
                )
                return models

        except Exception as exc:
            print(
                "Failed to load consolidated anomaly model registry:",
                exc,
            )

    # ---------------------------------------------------------
    # 2. Fallback to legacy model-directory loading
    # ---------------------------------------------------------
    directory = find_model_directory(
        ANOMALY_MODEL_DIR,
        [
            MODELS_DIR / "anomaly_models",
            MODELS_DIR / "anomaly_detection",
        ],
    )

    if directory is None:
        print("Anomaly model directory not found.")
        return None

    models = try_registry_load(
        ParameterModelRegistry,
        directory,
    )

    if models is None:
        print("Anomaly models could not be loaded.")

    else:
        print(
            "Legacy anomaly models loaded from:",
            directory,
        )
        print(
            "Anomaly model groups:",
            count_models(models),
        )

    return models

def load_prediction_models() -> Any:

    directory = find_model_directory(
        PREDICTION_MODEL_DIR,
        [
            MODELS_DIR / "prediction_models",
            MODELS_DIR / "drift_prediction",
            MODELS_DIR / "drift",
        ],
    )

    if directory is None:
        print("Prediction model directory not found.")
        return None

    models = try_registry_load(
        PredictionModelRegistry,
        directory,
    )

    if models is None:
        print("Prediction models could not be loaded.")

    else:
        print(
            "Prediction models loaded:",
            count_models(models),
        )

    return models


# =============================================================================
# RISK ENGINE
# =============================================================================

def initialize_risk_engine() -> Any:

    if RiskEngine is None:
        return None

    try:
        return RiskEngine()

    except Exception as error:

        print(
            "WARNING: RiskEngine initialization failed:",
            error,
        )

        return None


# =============================================================================
# FALLBACK ANOMALY DETECTOR
# =============================================================================

def heuristic_anomaly(
    row: pd.Series,
) -> dict[str, Any]:
    """
    Measurement-only fallback.

    IMPORTANT:
    defect_type and is_defective are never used.
    """

    value_0 = float(
        row.get(
            "value_0h",
            0.0,
        )
    )

    value_24 = float(
        row.get(
            "value_24h",
            0.0,
        )
    )

    drift = value_24 - value_0

    relative_drift = 0.0

    if abs(value_0) > 1e-9:

        relative_drift = abs(
            drift / value_0
        )

    z_slope = abs(
        float(
            row.get(
                "z_score_slope_early",
                0.0,
            )
        )
    )

    score = (
        min(
            relative_drift * 100.0,
            100.0,
        ) * 0.55
        +
        min(
            z_slope * 15.0,
            100.0,
        ) * 0.45
    )

    score = float(
        np.clip(
            score,
            0.0,
            100.0,
        )
    )

    flag = score >= 45.0

    return {
        "anomaly_flag": int(flag),
        "anomaly_label": (
            "ANOMALY"
            if flag
            else "NORMAL"
        ),
        "anomaly_score": score,
        "anomaly_index": score,
        "anomaly_raw_score": score,
    }


# =============================================================================
# ANOMALY CALCULATION
# =============================================================================

def calculate_anomaly(
    row: pd.Series,
) -> dict[str, Any]:

    if ANOMALY_MODELS is None:
        return heuristic_anomaly(row)

    try:

        # NOTE: ParameterModelRegistry does not have a
        # predict_component method. analyze_component is the real
        # API — it runs the trained Isolation Forest model (with
        # lot-relative features) and returns a plain dict.
        result = ANOMALY_MODELS.analyze_component(
            row.to_dict()
            if hasattr(row, "to_dict")
            else row
        )

        is_anomaly = result.get(
            "anomaly_flag",
            0,
        )

        anomaly_score = result.get(
            "anomaly_score",
            0.0,
        )

        anomaly_index = result.get(
            "anomaly_index",
            anomaly_score,
        )

        raw_score = result.get(
            "anomaly_raw_score",
            anomaly_score,
        )

        flag = int(bool(is_anomaly))

        # Find which early measurement deviates most from this
        # component's own lot baseline — used to build a QA-facing
        # explanation (e.g. "24h reading is 3.8sigma above this
        # lot's average").
        lot_hours = [
            "0h",
            "24h",
            "96h",
        ]

        worst_hour = None
        worst_zscore = 0.0

        for hour in lot_hours:

            zscore = float(
                result.get(
                    f"lot_zscore_{hour}",
                    0.0,
                )
                or 0.0
            )

            if abs(zscore) > abs(
                worst_zscore
            ):
                worst_zscore = zscore
                worst_hour = hour

        lot_context: dict[str, Any] = {
            "lot_id": result.get(
                "lot_id",
                row.get(
                    "lot_id",
                    None,
                ),
            ),
            "lot_worst_hour": worst_hour,
            "lot_worst_zscore": (
                round(worst_zscore, 3)
                if worst_hour
                else None
            ),
        }

        if worst_hour is not None:

            lot_context[
                "lot_worst_value"
            ] = float(
                result.get(
                    f"value_{worst_hour}",
                    0.0,
                )
                or 0.0
            )

            lot_context[
                "lot_worst_mean"
            ] = float(
                result.get(
                    f"lot_mean_{worst_hour}",
                    0.0,
                )
                or 0.0
            )

        return {
            "anomaly_flag": flag,
            "anomaly_label": (
                "ANOMALY"
                if flag
                else "NORMAL"
            ),
            "anomaly_score": float(
                anomaly_score
            ),
            "anomaly_index": float(
                anomaly_index
            ),
            "anomaly_raw_score": float(
                raw_score
            ),
            "input_stage": result.get(
                "input_stage",
                "0h_24h_96h",
            ),
            **lot_context,
        }

    except Exception as error:

        print(
            "Anomaly model failed for",
            row.get("component_id"),
            ":",
            error,
        )

        return heuristic_anomaly(row)


# =============================================================================
# PREDICTION FALLBACK
# =============================================================================

def linear_prediction_fallback(
    row: pd.Series,
) -> float:
    """
    Uses only 0h and 24h measurements.

    This is intentionally an early prediction.
    """

    value_0 = float(
        row.get(
            "value_0h",
            0.0,
        )
    )

    value_24 = float(
        row.get(
            "value_24h",
            value_0,
        )
    )

    slope = (
        value_24 - value_0
    ) / 24.0

    prediction = (
        value_24
        + slope * 144.0
    )

    return float(prediction)


def extract_prediction_value(
    result: Any,
) -> float | None:

    if result is None:
        return None

    if isinstance(result, dict):

        keys = [
            "predicted_168h",
            "prediction",
            "predicted_value",
            "value_168h",
            "predicted_value_168h",
            "future_value",
        ]

        for key in keys:

            if key in result:

                try:

                    value = float(
                        result[key]
                    )

                    if np.isfinite(value):
                        return value

                except Exception:
                    pass

    for attribute in [
        "predicted_168h",
        "prediction",
        "predicted_value",
        "value_168h",
        "predicted_value_168h",
        "future_value",
    ]:

        if hasattr(result, attribute):

            try:

                value = float(
                    getattr(
                        result,
                        attribute,
                    )
                )

                if np.isfinite(value):
                    return value

            except Exception:
                pass

    try:

        value = float(result)

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return None


# =============================================================================
# PREDICTION
# =============================================================================

def calculate_prediction(
    row: pd.Series,
) -> float:

    fallback = linear_prediction_fallback(
        row
    )

    if PREDICTION_MODELS is None:
        return fallback

    try:

        predict_component = getattr(
            PREDICTION_MODELS,
            "predict_component",
            None,
        )

        if callable(predict_component):

            result = predict_component(
                row
            )

            value = extract_prediction_value(
                result
            )

            if value is not None:
                return value

        models = getattr(
            PREDICTION_MODELS,
            "models",
            {},
        )

        component_type = str(
            row.get(
                "component_type",
                "",
            )
        ).strip()

        parameter_name = str(
            row.get(
                "parameter_name",
                "",
            )
        ).strip()

        key = (
            f"{component_type}__"
            f"{parameter_name}"
        )

        model = None

        if isinstance(models, dict):
            model = models.get(key)

        if model is not None:

            feature_values = pd.DataFrame(
                [
                    {
                        "value_0h": float(
                            row.get(
                                "value_0h",
                                0.0,
                            )
                        ),
                        "value_24h": float(
                            row.get(
                                "value_24h",
                                0.0,
                            )
                        ),
                    }
                ]
            )

            if hasattr(model, "predict"):

                result = model.predict(
                    feature_values
                )

                if isinstance(
                    result,
                    (
                        list,
                        tuple,
                        np.ndarray,
                    ),
                ):

                    result = result[0]

                value = extract_prediction_value(
                    result
                )

                if value is not None:
                    return value

        return fallback

    except Exception as error:

        print(
            "Prediction model failed for",
            row.get("component_id"),
            ":",
            error,
        )

        return fallback


# =============================================================================
# FALLBACK RISK ENGINE
# =============================================================================

def fallback_risk(
    row: pd.Series,
    anomaly_result: dict[str, Any],
    predicted_168h: float,
) -> dict[str, Any]:

    parameter_name = str(
        row.get(
            "parameter_name",
            "",
        )
    )

    value_0 = float(
        row.get(
            "value_0h",
            0.0,
        )
    )

    value_24 = float(
        row.get(
            "value_24h",
            0.0,
        )
    )

    engineering_limit = get_engineering_limit(
        parameter_name
    )

    # -------------------------------------------------------------------------
    # Limit utilization
    # -------------------------------------------------------------------------

    if (
        engineering_limit is not None
        and engineering_limit > 0
    ):

        limit_utilization = (
            abs(predicted_168h)
            / engineering_limit
            * 100.0
        )

        current_limit_utilization = (
            abs(value_24)
            / engineering_limit
            * 100.0
        )

    else:

        limit_utilization = 0.0
        current_limit_utilization = 0.0

    # -------------------------------------------------------------------------
    # Early drift
    # -------------------------------------------------------------------------

    early_drift = (
        value_24 - value_0
    )

    early_drift_percent = 0.0

    if abs(value_0) > 1e-9:

        early_drift_percent = (
            abs(early_drift)
            / abs(value_0)
            * 100.0
        )

    # -------------------------------------------------------------------------
    # Future drift
    # -------------------------------------------------------------------------

    future_drift = (
        predicted_168h
        - value_24
    )

    future_drift_percent = 0.0

    if abs(value_24) > 1e-9:

        future_drift_percent = (
            abs(future_drift)
            / abs(value_24)
            * 100.0
        )

    # -------------------------------------------------------------------------
    # Anomaly
    # -------------------------------------------------------------------------

    anomaly_component = float(
        anomaly_result.get(
            "anomaly_index",
            0.0,
        )
    )

    # -------------------------------------------------------------------------
    # Risk components
    # -------------------------------------------------------------------------

    drift_component = min(
        max(
            early_drift_percent * 5.0,
            0.0,
        ),
        100.0,
    )

    future_component = min(
        max(
            future_drift_percent * 2.0,
            0.0,
        ),
        100.0,
    )

    limit_component = min(
        max(
            current_limit_utilization,
            0.0,
        ),
        100.0,
    )

    predicted_limit_component = min(
        max(
            limit_utilization,
            0.0,
        ),
        100.0,
    )

    # -------------------------------------------------------------------------
    # Weighted risk
    # -------------------------------------------------------------------------

    risk_score = (
        anomaly_component * 0.40
        +
        future_component * 0.25
        +
        drift_component * 0.15
        +
        predicted_limit_component * 0.15
        +
        limit_component * 0.05
    )

    risk_score = float(
        np.clip(
            risk_score,
            0.0,
            100.0,
        )
    )

    # -------------------------------------------------------------------------
    # Risk level
    # -------------------------------------------------------------------------

    if risk_score >= 80.0:

        risk_level = "CRITICAL"

    elif risk_score >= 60.0:

        risk_level = "HIGH"

    elif risk_score >= 30.0:

        risk_level = "MEDIUM"

    else:

        risk_level = "LOW"

    # -------------------------------------------------------------------------
    # Explainability
    # -------------------------------------------------------------------------

    reasons: list[str] = []

    if anomaly_component >= 60:

        reasons.append(
            "AI anomaly detector identified abnormal burn-in behavior."
        )

    elif anomaly_component >= 40:

        reasons.append(
            "Burn-in trajectory shows elevated anomaly characteristics."
        )

    if early_drift_percent >= 5:

        reasons.append(
            "Moderate early parameter drift detected."
        )

    if early_drift_percent >= 15:

        reasons.append(
            "Severe early parameter drift detected."
        )

    if future_drift_percent >= 20:

        reasons.append(
            "Model predicts significant future parameter degradation."
        )

    if future_drift_percent >= 50:

        reasons.append(
            "Predicted future degradation is severe."
        )

    if engineering_limit is not None:

        if limit_utilization >= 100:

            reasons.append(
                "Predicted 168h value exceeds the engineering limit."
            )

        elif limit_utilization >= 80:

            reasons.append(
                "Predicted 168h value approaches the engineering limit."
            )

        if current_limit_utilization >= 80:

            reasons.append(
                "Current burn-in value is close to the engineering limit."
            )

    slope_early = float(
        row.get(
            "slope_early",
            0.0,
        )
    )

    if slope_early > 0:

        reasons.append(
            "Burn-in measurements show a continuing degradation trend."
        )

    if not reasons:

        reasons.append(
            "No major early-risk indicator was detected."
        )

    # -------------------------------------------------------------------------
    # Recommendation
    # -------------------------------------------------------------------------

    if risk_level == "CRITICAL":

        recommendation = (
            "FLAG COMPONENT — severe degradation or anomaly "
            "indicators detected. Immediate inspection recommended."
        )

    elif risk_level == "HIGH":

        recommendation = (
            "REVIEW COMPONENT — elevated reliability risk "
            "detected during burn-in screening."
        )

    elif risk_level == "MEDIUM":

        recommendation = (
            "MONITOR COMPONENT — moderate degradation "
            "indicators detected."
        )

    else:

        recommendation = (
            "PASS — no major early-risk indicator detected."
        )

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "risk_decision": risk_level,

        "early_drift_index": float(
            np.clip(
                early_drift_percent,
                0.0,
                100.0,
            )
        ),

        "future_drift_index": float(
            np.clip(
                future_drift_percent,
                0.0,
                100.0,
            )
        ),

        "limit_utilization": float(
            np.clip(
                limit_utilization,
                0.0,
                150.0,
            )
        ),

        "engineering_limit": (
            float(engineering_limit)
            if engineering_limit is not None
            else None
        ),

        "early_drift": float(
            early_drift
        ),

        "future_drift": float(
            future_drift
        ),

        "safety_slope": float(
            slope_early
        ),

        "risk_reasons": reasons,
        "reasons": reasons,
        "recommendation": recommendation,
    }


# =============================================================================
# RISK ENGINE ADAPTER
# =============================================================================

def calculate_risk(
    row: pd.Series,
    anomaly_result: dict[str, Any],
    predicted_168h: float,
) -> dict[str, Any]:

    risk_row = row.copy()

    # -------------------------------------------------------------------------
    # Inject AI anomaly information
    # -------------------------------------------------------------------------

    risk_row["anomaly_flag"] = int(
        anomaly_result.get(
            "anomaly_flag",
            0,
        )
    )

    risk_row["anomaly_score"] = float(
        anomaly_result.get(
            "anomaly_score",
            0.0,
        )
    )

    risk_row["anomaly_index"] = float(
        anomaly_result.get(
            "anomaly_index",
            0.0,
        )
    )

    risk_row["anomaly_raw_score"] = float(
        anomaly_result.get(
            "anomaly_raw_score",
            0.0,
        )
    )

    # -------------------------------------------------------------------------
    # Lot-relative context (for explainability — lets the risk
    # engine cite "this reading is Nsigma off its own lot's
    # baseline" instead of only the global population).
    # -------------------------------------------------------------------------

    risk_row["lot_worst_hour"] = (
        anomaly_result.get(
            "lot_worst_hour"
        )
    )

    risk_row["lot_worst_zscore"] = (
        anomaly_result.get(
            "lot_worst_zscore"
        )
    )

    risk_row["lot_worst_value"] = (
        anomaly_result.get(
            "lot_worst_value"
        )
    )

    risk_row["lot_worst_mean"] = (
        anomaly_result.get(
            "lot_worst_mean"
        )
    )

    # -------------------------------------------------------------------------
    # IMPORTANT FIX:
    # Explicitly inject engineering limit BEFORE calling RiskEngine.
    #
    # This prevents engineering_limit from becoming 0.0.
    # -------------------------------------------------------------------------

    engineering_limit = get_engineering_limit(
        row.get("parameter_name")
    )

    if engineering_limit is not None:

        risk_row["engineering_limit"] = float(
            engineering_limit
        )

    # -------------------------------------------------------------------------
    # Risk engine
    # -------------------------------------------------------------------------

    if RISK_ENGINE is not None:

        try:

            evaluate = getattr(
                RISK_ENGINE,
                "evaluate_component",
                None,
            )

            if callable(evaluate):

                slope_early = float(
                    row.get(
                        "slope_early",
                        0.0,
                    )
                )

                result = evaluate(
                    risk_row,
                    float(predicted_168h),
                    slope_early,
                )

                if isinstance(result, dict):

                    result_engineering_limit = result.get(
                        "engineering_limit",
                        None,
                    )

                    if (
                        result_engineering_limit is None
                        or float(
                            result_engineering_limit
                        ) == 0.0
                    ):
                        result_engineering_limit = engineering_limit

                    normalized = {
                        **result,

                        "risk_score": float(
                            result.get(
                                "risk_score",
                                0.0,
                            )
                        ),

                        "risk_level": str(
                            result.get(
                                "risk_level",
                                "LOW",
                            )
                        ),

                        "engineering_limit": (
                            float(
                                result_engineering_limit
                            )
                            if result_engineering_limit is not None
                            else None
                        ),

                        "limit_utilization": float(
                            result.get(
                                "limit_utilization",
                                0.0,
                            )
                        ),

                        "early_drift_index": float(
                            result.get(
                                "early_drift_index",
                                0.0,
                            )
                        ),

                        "future_drift_index": float(
                            result.get(
                                "future_drift_index",
                                0.0,
                            )
                        ),
                    }

                    reasons = normalized.get(
                        "risk_reasons",
                        normalized.get(
                            "reasons",
                            [],
                        ),
                    )

                    if not isinstance(
                        reasons,
                        list,
                    ):
                        reasons = [
                            str(reasons)
                        ]

                    normalized["risk_reasons"] = reasons
                    normalized["reasons"] = reasons

                    normalized["risk_decision"] = normalized[
                        "risk_level"
                    ]

                    if "recommendation" not in normalized:

                        level = normalized[
                            "risk_level"
                        ].upper()

                        if level == "CRITICAL":

                            normalized["recommendation"] = (
                                "FLAG COMPONENT — severe degradation "
                                "or anomaly indicators detected. "
                                "Immediate inspection recommended."
                            )

                        elif level == "HIGH":

                            normalized["recommendation"] = (
                                "REVIEW COMPONENT — elevated "
                                "reliability risk detected."
                            )

                        elif level == "MEDIUM":

                            normalized["recommendation"] = (
                                "MONITOR COMPONENT — moderate "
                                "degradation indicators detected."
                            )

                        else:

                            normalized["recommendation"] = (
                                "PASS — no major early-risk "
                                "indicator detected."
                            )

                    return normalized

        except Exception as error:

            print(
                "RiskEngine failed:",
                error,
            )

            traceback.print_exc()

    # -------------------------------------------------------------------------
    # Fallback
    # -------------------------------------------------------------------------

    return fallback_risk(
        row,
        anomaly_result,
        predicted_168h,
    )


# =============================================================================
# COMPONENT ANALYSIS
# =============================================================================

def analyze_component(
    component_id: str,
) -> dict[str, Any]:

    component_id = str(
        component_id
    ).strip()

    if component_id in ANALYSIS_CACHE:
        return ANALYSIS_CACHE[
            component_id
        ]

    if (
        FEATURE_DATA is None
        or FEATURE_DATA.empty
    ):

        raise RuntimeError(
            "Feature dataset is not loaded."
        )

    matches = FEATURE_DATA[
        FEATURE_DATA[
            "component_id"
        ]
        .astype(str)
        .str.strip()
        == component_id
    ]

    if matches.empty:

        raise KeyError(
            f"Component not found: {component_id}"
        )

    row = matches.iloc[0].copy()

    # -------------------------------------------------------------------------
    # 1. ANOMALY
    # -------------------------------------------------------------------------

    anomaly_result = calculate_anomaly(
        row
    )

    row["anomaly_flag"] = int(
        anomaly_result[
            "anomaly_flag"
        ]
    )

    row["anomaly_score"] = float(
        anomaly_result[
            "anomaly_score"
        ]
    )

    row["anomaly_index"] = float(
        anomaly_result[
            "anomaly_index"
        ]
    )

    row["anomaly_raw_score"] = float(
        anomaly_result[
            "anomaly_raw_score"
        ]
    )

    # -------------------------------------------------------------------------
    # 2. FUTURE PREDICTION
    # -------------------------------------------------------------------------

    predicted_168h = calculate_prediction(
        row
    )

    prediction_input_stage = (
        "0h_24h_96h"
        if pd.notna(
            row.get("value_96h")
        )
        else "0h_24h"
    )

    value_24h = float(
        row.get(
            "value_24h",
            0.0,
        )
    )

    predicted_slope = (
        predicted_168h
        - value_24h
    ) / 144.0

    # -------------------------------------------------------------------------
    # 3. RISK
    # -------------------------------------------------------------------------

    risk_result = calculate_risk(
        row,
        anomaly_result,
        predicted_168h,
    )

    # -------------------------------------------------------------------------
    # 4. ENGINEERING LIMIT
    #
    # Always use the known parameter mapping if the risk engine
    # returned 0 or None.
    # -------------------------------------------------------------------------

    engineering_limit = risk_result.get(
        "engineering_limit"
    )

    if (
        engineering_limit is None
        or float(engineering_limit) == 0.0
    ):

        engineering_limit = get_engineering_limit(
            row.get(
                "parameter_name"
            )
        )

    if engineering_limit is not None:

        engineering_limit = float(
            engineering_limit
        )

    # -------------------------------------------------------------------------
    # 5. LIMIT UTILIZATION
    #
    # Recalculate if the risk engine did not provide a useful value.
    # -------------------------------------------------------------------------

    limit_utilization = risk_result.get(
        "limit_utilization",
        0.0,
    )

    try:
        limit_utilization = float(
            limit_utilization
        )
    except Exception:
        limit_utilization = 0.0

    if (
        engineering_limit is not None
        and engineering_limit > 0
    ):

        recalculated_utilization = (
            abs(predicted_168h)
            / engineering_limit
            * 100.0
        )

        if (
            not np.isfinite(
                limit_utilization
            )
            or limit_utilization <= 0.0
        ):

            limit_utilization = (
                recalculated_utilization
            )

    # -------------------------------------------------------------------------
    # 6. REASONS
    # -------------------------------------------------------------------------

    reasons = risk_result.get(
        "risk_reasons",
        risk_result.get(
            "reasons",
            [],
        ),
    )

    if not isinstance(
        reasons,
        list,
    ):

        reasons = [
            str(reasons)
        ]

    # -------------------------------------------------------------------------
    # 7. FINAL RESULT
    # -------------------------------------------------------------------------

    # These are derived directly from the actual value_168h (only
    # present when a dataset happens to include real 168h data,
    # e.g. an uploaded CSV). They are never used in any anomaly or
    # risk decision, but leaving them in the API response would
    # still expose future-derived information in what is meant to
    # be a pre-168h predictive screening payload. Strip them here.
    leaked_future_fields = (
        "drift_96_168",
        "drift_0_168",
        "ratio_168_96",
        "ratio_168_0",
        "slope_late",
        "late_growth_ratio",
    )

    row_dict = row.to_dict()

    for field in leaked_future_fields:
        row_dict.pop(field, None)

    result = {
        **row_dict,

        **anomaly_result,

        # Prediction
        "predicted_168h": float(
            predicted_168h
        ),

        "predicted_slope": float(
            predicted_slope
        ),

        "prediction_input_stage":
            prediction_input_stage,

        # Risk
        "risk_score": float(
            risk_result.get(
                "risk_score",
                0.0,
            )
        ),

        "risk_level": str(
            risk_result.get(
                "risk_level",
                "LOW",
            )
        ),

        "risk_decision": str(
            risk_result.get(
                "risk_decision",
                risk_result.get(
                    "risk_level",
                    "LOW",
                ),
            )
        ),

        # Behavior pattern classification
        "behavior_pattern": str(
            risk_result.get(
                "behavior_pattern",
                "NORMAL",
            )
        ),

        "behavior_pattern_label": str(
            risk_result.get(
                "behavior_pattern_label",
                "Normal",
            )
        ),

        "behavior_pattern_explanation": str(
            risk_result.get(
                "behavior_pattern_explanation",
                "",
            )
        ),

        # Drift
        "early_drift_index": float(
            risk_result.get(
                "early_drift_index",
                0.0,
            )
        ),

        "future_drift_index": float(
            risk_result.get(
                "future_drift_index",
                0.0,
            )
        ),

        "early_drift": float(
            row.get(
                "drift_0_24",
                0.0,
            )
        ),

        "future_drift": float(
            predicted_168h
            - value_24h
        ),

        "safety_slope": float(
            row.get(
                "slope_early",
                0.0,
            )
        ),

        # Engineering limit
        "engineering_limit": (
            engineering_limit
        ),

        "limit_utilization": float(
            limit_utilization
        ),

        # Explainability
        "risk_reasons": reasons,
        "reasons": reasons,

        "recommendation": str(
            risk_result.get(
                "recommendation",
                "PASS — no major early-risk indicator detected.",
            )
        ),

        # Units
        "prediction_unit": clean_text(
            row.get(
                "unit",
                "",
            )
        ),

        "parameter_unit": clean_text(
            row.get(
                "unit",
                "",
            )
        ),
    }

    result = clean_record(
        result
    )

    ANALYSIS_CACHE[
        component_id
    ] = result

    return result


# =============================================================================
# DATASET REPLACEMENT
# =============================================================================

def replace_dataset(
    df: pd.DataFrame,
    filename: str,
) -> None:

    global COMPONENT_DATA
    global FEATURE_DATA
    global ACTIVE_DATASET_NAME
    global ACTIVE_DATASET_PATH
    global ANALYSIS_CACHE

    prepared = prepare_dataset(
        df
    )

    if prepared.empty:

        raise ValueError(
            "Uploaded CSV contains no valid component rows."
        )

    COMPONENT_DATA = prepared

    FEATURE_DATA = create_features(
        COMPONENT_DATA
    )

    ACTIVE_DATASET_NAME = filename

    ACTIVE_DATASET_PATH = Path(
        filename
    )

    ANALYSIS_CACHE = {}

    print("=" * 70)
    print("ACTIVE DATASET UPDATED")
    print("=" * 70)
    print(f"Name: {filename}")
    print(f"Rows: {len(prepared):,}")
    print("=" * 70)


# =============================================================================
# APPLICATION LIFESPAN
# =============================================================================

@asynccontextmanager
async def lifespan(
    app: FastAPI,
):

    global COMPONENT_DATA
    global FEATURE_DATA
    global ANOMALY_MODELS
    global PREDICTION_MODELS
    global RISK_ENGINE
    global ANALYSIS_CACHE

    print("=" * 70)
    print("STARTING AEGISBURN AI")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------------------

    try:

        COMPONENT_DATA = load_dataset()

        FEATURE_DATA = create_features(
            COMPONENT_DATA
        )

        print(
            f"Feature rows: {len(FEATURE_DATA):,}"
        )

    except Exception as error:

        print(
            f"ERROR loading dataset: {error}"
        )

        traceback.print_exc()

        COMPONENT_DATA = pd.DataFrame()
        FEATURE_DATA = pd.DataFrame()

    # -------------------------------------------------------------------------
    # Models
    # -------------------------------------------------------------------------

    ANOMALY_MODELS = (
        load_anomaly_models()
    )

    PREDICTION_MODELS = (
        load_prediction_models()
    )

    RISK_ENGINE = (
        initialize_risk_engine()
    )

    ANALYSIS_CACHE = {}

    print("=" * 70)
    print("AEGISBURN AI — BACKEND READY")
    print("=" * 70)

    yield

    print(
        "AEGISBURN AI shutting down."
    )


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="AegisBurn AI",
    version="1.0.0",
    description=(
        "AI-driven burn-in anomaly detection, "
        "drift prediction and predictive component screening."
    ),
    lifespan=lifespan,
)


# =============================================================================
# CORS
# =============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# STATIC FILES
# =============================================================================

if STATIC_DIR.exists():

    app.mount(
        "/static",
        StaticFiles(
            directory=str(
                STATIC_DIR
            )
        ),
        name="static",
    )


# =============================================================================
# DASHBOARD
# =============================================================================

@app.get(
    "/",
    include_in_schema=False,
)
async def dashboard():

    index_file = (
        STATIC_DIR
        / "index.html"
    )

    if not index_file.exists():

        return JSONResponse(
            status_code=404,
            content={
                "detail": (
                    "Dashboard not found. "
                    "Expected backend/static/index.html"
                )
            },
        )

    return FileResponse(
        str(index_file)
    )


# =============================================================================
# HEALTH
# =============================================================================

@app.get("/health")
async def health():

    return {
        "status": "healthy",
        "system": "AegisBurn AI",

        "anomaly_models": count_models(
            ANOMALY_MODELS
        ),

        "prediction_models": count_models(
            PREDICTION_MODELS
        ),

        "risk_engine": (
            RISK_ENGINE is not None
        ),

        "dataset_loaded": (
            COMPONENT_DATA is not None
            and not COMPONENT_DATA.empty
        ),
    }


# =============================================================================
# METADATA
# =============================================================================

@app.get("/metadata")
async def metadata():

    total = 0

    component_types: list[str] = []
    parameters: list[str] = []

    if (
        COMPONENT_DATA is not None
        and not COMPONENT_DATA.empty
    ):

        total = len(
            COMPONENT_DATA
        )

        component_types = sorted(
            COMPONENT_DATA[
                "component_type"
            ]
            .dropna()
            .astype(str)
            .map(clean_text)
            .unique()
            .tolist()
        )

        parameters = sorted(
            COMPONENT_DATA[
                "parameter_name"
            ]
            .dropna()
            .astype(str)
            .map(clean_text)
            .unique()
            .tolist()
        )

    return {
        "system": "AegisBurn AI",
        "version": "1.0.0",
        "status": "running",

        "total": total,
        "rows": total,

        "dataset_name": ACTIVE_DATASET_NAME,

        "dataset_path": str(
            ACTIVE_DATASET_PATH
        ),

        "component_types": component_types,

        "parameters": parameters,

        "anomaly_models": count_models(
            ANOMALY_MODELS
        ),

        "prediction_models": count_models(
            PREDICTION_MODELS
        ),

        "risk_engine": (
            RISK_ENGINE is not None
        ),

        "message": (
            "AI-driven anomaly detection "
            "and burn-in drift prediction."
        ),
    }


# =============================================================================
# COMPONENT LIST
# =============================================================================

@app.get("/components")
async def get_components():

    if (
        COMPONENT_DATA is None
        or COMPONENT_DATA.empty
    ):

        raise HTTPException(
            status_code=503,
            detail="Dataset is not loaded.",
        )

    columns = [
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
    ]

    available_columns = [
        column
        for column in columns
        if column in COMPONENT_DATA.columns
    ]

    records = COMPONENT_DATA[
        available_columns
    ].to_dict(
        orient="records"
    )

    return [
        clean_record(record)
        for record in records
    ]


# =============================================================================
# SINGLE COMPONENT
# =============================================================================

@app.get(
    "/components/{component_id}"
)
async def get_component(
    component_id: str,
):

    if COMPONENT_DATA is None:

        raise HTTPException(
            status_code=503,
            detail="Dataset is not loaded.",
        )

    matches = COMPONENT_DATA[
        COMPONENT_DATA[
            "component_id"
        ]
        .astype(str)
        .str.strip()
        == str(component_id).strip()
    ]

    if matches.empty:

        raise HTTPException(
            status_code=404,
            detail=(
                f"Component not found: "
                f"{component_id}"
            ),
        )

    return clean_record(
        matches.iloc[0].to_dict()
    )


# =============================================================================
# ANALYZE — GET
# =============================================================================

@app.get(
    "/analyze/{component_id}"
)
async def analyze_component_get(
    component_id: str,
):

    try:

        return analyze_component(
            component_id
        )

    except KeyError as error:

        raise HTTPException(
            status_code=404,
            detail=str(error),
        )

    except Exception as error:

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


# =============================================================================
# ANALYZE — POST
# =============================================================================

@app.post("/analyze")
async def analyze_component_post(
    request: AnalyzeRequest,
):

    try:

        return analyze_component(
            request.component_id
        )

    except KeyError as error:

        raise HTTPException(
            status_code=404,
            detail=str(error),
        )

    except Exception as error:

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


# =============================================================================
# CSV UPLOAD
# =============================================================================

@app.post("/upload-csv")
async def upload_csv(
    file: UploadFile = File(...),
):
    global COMPONENT_DATA
    global FEATURE_DATA
    global ANALYSIS_CACHE
    global ACTIVE_DATASET_NAME
    global ACTIVE_DATASET_PATH

    # -------------------------------------------------------------
    # Validate filename
    # -------------------------------------------------------------

    filename = (
        file.filename or "uploaded_dataset.csv"
    ).strip()

    if not filename:
        filename = "uploaded_dataset.csv"

    if not filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV file.",
        )

    # -------------------------------------------------------------
    # Read uploaded file
    # -------------------------------------------------------------

    try:
        contents = await file.read()

    except Exception as error:
        traceback.print_exc()

        raise HTTPException(
            status_code=400,
            detail=f"Could not read uploaded CSV: {error}",
        )

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Uploaded CSV is empty.",
        )

    # -------------------------------------------------------------
    # Decode CSV
    #
    # utf-8-sig handles both normal UTF-8 and UTF-8 files with BOM.
    # latin-1 is a fallback for older CSV files.
    # -------------------------------------------------------------

    try:
        try:
            csv_text = contents.decode(
                "utf-8-sig"
            )
        except UnicodeDecodeError:
            csv_text = contents.decode(
                "latin-1"
            )

    except Exception as error:
        traceback.print_exc()

        raise HTTPException(
            status_code=400,
            detail=f"Could not decode CSV: {error}",
        )

    # -------------------------------------------------------------
    # Parse CSV
    # -------------------------------------------------------------

    try:
        df = pd.read_csv(
            io.StringIO(csv_text)
        )

    except Exception as error:
        traceback.print_exc()

        raise HTTPException(
            status_code=400,
            detail=f"Could not parse CSV: {error}",
        )

    # -------------------------------------------------------------
    # Validate and prepare dataset
    # -------------------------------------------------------------

    try:
        prepared = prepare_dataset(
            df
        )

    except Exception as error:
        traceback.print_exc()

        raise HTTPException(
            status_code=400,
            detail=(
                f"Could not process CSV: "
                f"{error}"
            ),
        )

    if prepared.empty:
        raise HTTPException(
            status_code=400,
            detail="No valid component rows were found in the CSV.",
        )

    # -------------------------------------------------------------
    # Make sure component IDs are unique
    # -------------------------------------------------------------

    if prepared["component_id"].duplicated().any():
        duplicate_count = int(
            prepared["component_id"]
            .duplicated()
            .sum()
        )

        raise HTTPException(
            status_code=400,
            detail=(
                f"CSV contains {duplicate_count} "
                "duplicate component IDs."
            ),
        )

    # -------------------------------------------------------------
    # Replace active dataset
    # -------------------------------------------------------------

    try:
        COMPONENT_DATA = prepared.copy()

        FEATURE_DATA = create_features(
            COMPONENT_DATA
        )

        ACTIVE_DATASET_NAME = filename

        ACTIVE_DATASET_PATH = (
            DEFAULT_DATASET
        )

        ANALYSIS_CACHE = {}

    except Exception as error:
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=(
                f"Could not activate uploaded dataset: "
                f"{error}"
            ),
        )

    # -------------------------------------------------------------
    # Return upload result
    # -------------------------------------------------------------

    return {
        "status": "success",
        "message": "CSV uploaded successfully.",
        "filename": filename,
        "dataset_name": filename,
        "rows": len(COMPONENT_DATA),
        "components": len(COMPONENT_DATA),
        "columns": list(
            COMPONENT_DATA.columns
        ),
    }

# =============================================================================
# REFRESH
# =============================================================================

@app.post("/refresh")
async def refresh():

    global ANALYSIS_CACHE

    ANALYSIS_CACHE = {}

    return {
        "status": "success",
        "message": (
            "Analysis cache cleared."
        ),
    }


# =============================================================================
# RUN DIRECTLY
# =============================================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=8001,
        reload=True,
    )