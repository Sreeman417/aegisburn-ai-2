"""
AEGISBURN AI — Parameter-Aware 168h Drift Prediction

Module B:
Predict Value_168h from early burn-in measurements.

One model is trained for each:
    component_type + parameter_name

Input features:
    value_0h
    value_24h

Target:
    value_168h

The registry uses a canonical model key so filenames and
runtime lookups remain consistent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# FEATURES
# ============================================================

PREDICTION_FEATURES: List[str] = [
    "value_0h",
    "value_24h",
    "value_96h",
    "drift_0_24",
    "drift_24_96",
    "drift_0_96",
    "relative_drift_0_24",
    "relative_drift_24_96",
    "relative_drift_0_96",
    "slope_early",
    "slope_mid",
    "ratio_24_0",
    "ratio_96_24",
    "ratio_96_0",
    "acceleration",
]

# The three raw measurements a caller must supply. Everything else
# in PREDICTION_FEATURES is derived from these via
# build_prediction_features().
PREDICTION_INPUT_COLUMNS: List[str] = [
    "value_0h",
    "value_24h",
    "value_96h",
]

TARGET_COLUMN = "value_168h"

# ---------------------------------------------------------------
# EARLY-ONLY (0h/24h) prediction path
#
# The literal problem statement specifies a model that "takes
# Value_0h and Value_24h as inputs" -- for components still mid
# burn-in where value_96h genuinely isn't measured yet. This is
# used automatically whenever value_96h is missing for a given
# component; PREDICTION_FEATURES (0h/24h/96h) is used whenever
# 96h is available, since it is meaningfully more accurate.
# ---------------------------------------------------------------

EARLY_INPUT_COLUMNS: List[str] = [
    "value_0h",
    "value_24h",
]

EARLY_FEATURES: List[str] = [
    "value_0h",
    "value_24h",
    "drift_0_24",
    "relative_drift_0_24",
    "slope_early",
    "ratio_24_0",
]


def build_early_prediction_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute the 0h/24h-only feature set, used when value_96h is
    not yet available for a component.
    """

    missing = [
        column
        for column in EARLY_INPUT_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing early prediction input columns: {missing}"
        )

    value_0h = pd.to_numeric(
        df["value_0h"],
        errors="coerce",
    )

    value_24h = pd.to_numeric(
        df["value_24h"],
        errors="coerce",
    )

    features = pd.DataFrame(
        index=df.index
    )

    features["value_0h"] = value_0h
    features["value_24h"] = value_24h

    features["drift_0_24"] = (
        value_24h - value_0h
    )

    safe_0h = value_0h.replace(0, np.nan)

    features["relative_drift_0_24"] = (
        features["drift_0_24"]
        / safe_0h.abs()
    )

    features["slope_early"] = (
        features["drift_0_24"] / 24.0
    )

    features["ratio_24_0"] = (
        value_24h / safe_0h
    )

    features = features.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return features


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_prediction_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute the full prediction feature set from the three early
    burn-in measurements (value_0h, value_24h, value_96h).

    Captures acceleration between the 24h and 96h readings, which
    is essential for detecting components that look fine early on
    but are already ramping toward failure by 96h (e.g. a jump
    from ~10 at 24h to ~20+ at 96h).
    """

    missing = [
        column
        for column in PREDICTION_INPUT_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing prediction input columns: {missing}"
        )

    value_0h = pd.to_numeric(
        df["value_0h"],
        errors="coerce",
    )

    value_24h = pd.to_numeric(
        df["value_24h"],
        errors="coerce",
    )

    value_96h = pd.to_numeric(
        df["value_96h"],
        errors="coerce",
    )

    features = pd.DataFrame(
        index=df.index
    )

    features["value_0h"] = value_0h
    features["value_24h"] = value_24h
    features["value_96h"] = value_96h

    features["drift_0_24"] = (
        value_24h - value_0h
    )

    features["drift_24_96"] = (
        value_96h - value_24h
    )

    features["drift_0_96"] = (
        value_96h - value_0h
    )

    safe_0h = value_0h.replace(0, np.nan)
    safe_24h = value_24h.replace(0, np.nan)

    features["relative_drift_0_24"] = (
        features["drift_0_24"]
        / safe_0h.abs()
    )

    features["relative_drift_24_96"] = (
        features["drift_24_96"]
        / safe_24h.abs()
    )

    features["relative_drift_0_96"] = (
        features["drift_0_96"]
        / safe_0h.abs()
    )

    # Average rate of change per hour, each interval.
    features["slope_early"] = (
        features["drift_0_24"] / 24.0
    )

    features["slope_mid"] = (
        features["drift_24_96"] / 72.0
    )

    features["ratio_24_0"] = (
        value_24h / safe_0h
    )

    features["ratio_96_24"] = (
        value_96h / safe_24h
    )

    features["ratio_96_0"] = (
        value_96h / safe_0h
    )

    # Acceleration: is the rate of change speeding up between the
    # early (0-24h) and mid (24-96h) intervals? This is the key
    # signal for components that look normal at 24h but are
    # already ramping by 96h.
    features["acceleration"] = (
        features["slope_mid"]
        - features["slope_early"]
    )

    features = features.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return features


# ============================================================
# RESULT
# ============================================================

@dataclass
class PredictionResult:
    predicted_168h: float
    component_type: str
    parameter_name: str
    model_name: str
    input_stage: str = "0h_24h_96h"


# ============================================================
# CANDIDATE MODEL SELECTION (shared by both the full 0h/24h/96h
# model and the early-only 0h/24h model)
# ============================================================

def _select_best_model(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int,
):
    """
    Fit LinearRegression / RandomForest / GradientBoosting on an
    80/20 split, pick the lowest-MAE model, then refit that model
    on all available data. Returns (model, name, mae, rmse, r2)
    where the metrics are from the held-out 20% split.
    """

    if len(X) < 20:
        raise ValueError(
            "Not enough valid rows for prediction model. "
            f"Found {len(X)}, need at least 20."
        )

    models = {
        "LinearRegression": Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "model",
                    LinearRegression(),
                ),
            ]
        ),

        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
            min_samples_leaf=2,
        ),

        "GradientBoostingRegressor": GradientBoostingRegressor(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=3,
            random_state=random_state,
        ),
    }

    best_model = None
    best_model_name = None
    best_mae = float("inf")
    best_rmse = None
    best_r2 = None

    split_index = int(len(X) * 0.80)

    if split_index < 10:
        split_index = len(X) - 10

    X_train = X.iloc[:split_index]
    y_train = y.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_test = y.iloc[split_index:]

    for name, model in models.items():

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        mae = mean_absolute_error(
            y_test,
            predictions,
        )

        rmse = float(
            np.sqrt(
                mean_squared_error(
                    y_test,
                    predictions,
                )
            )
        )

        r2 = r2_score(
            y_test,
            predictions,
        )

        if mae < best_mae:

            best_mae = mae
            best_rmse = rmse
            best_r2 = r2
            best_model = model
            best_model_name = name

    if best_model is None:
        raise RuntimeError(
            "Unable to select a prediction model."
        )

    # Refit the selected model on all available data.
    best_model.fit(X, y)

    return (
        best_model,
        best_model_name,
        float(best_mae),
        float(best_rmse),
        float(best_r2),
    )


# ============================================================
# SINGLE PREDICTOR
# ============================================================

class ParameterDriftPredictor:

    def __init__(
        self,
        random_state: int = 42,
    ) -> None:

        self.random_state = random_state

        self.model: Optional[Pipeline] = None

        self.model_name: Optional[str] = None

        self.component_type: Optional[str] = None

        self.parameter_name: Optional[str] = None

        self.feature_names = (
            PREDICTION_FEATURES.copy()
        )

        self.training_count: int = 0

        self.mae: Optional[float] = None

        self.rmse: Optional[float] = None

        self.r2: Optional[float] = None

        # -------------------------------------------------------
        # Early-only (0h/24h) model -- used automatically when
        # value_96h is not available for a component, matching
        # the literal problem statement's two-input spec.
        # -------------------------------------------------------

        self.early_model: Optional[Pipeline] = None

        self.early_model_name: Optional[str] = None

        self.early_feature_names = (
            EARLY_FEATURES.copy()
        )

        self.early_training_count: int = 0

        self.early_mae: Optional[float] = None

        self.early_rmse: Optional[float] = None

        self.early_r2: Optional[float] = None

    # ========================================================
    # FIT
    # ========================================================

    def fit(
        self,
        df: pd.DataFrame,
        component_type: Optional[str] = None,
        parameter_name: Optional[str] = None,
    ) -> "ParameterDriftPredictor":

        required = [
            *PREDICTION_INPUT_COLUMNS,
            TARGET_COLUMN,
        ]

        missing = [
            column
            for column in required
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing prediction columns: {missing}"
            )

        if component_type is not None:
            self.component_type = (
                str(component_type).strip()
            )

        if parameter_name is not None:
            self.parameter_name = (
                str(parameter_name).strip()
            )

        train_df = df.copy()

        # -----------------------------------------------------
        # Full model: 0h/24h/96h -> 168h
        # -----------------------------------------------------

        X_full = build_prediction_features(
            train_df
        )

        y_full = pd.to_numeric(
            train_df[TARGET_COLUMN],
            errors="coerce",
        )

        X_full = X_full.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        valid_full = (
            X_full.notna().all(axis=1)
            & y_full.notna()
        )

        X_full = X_full.loc[valid_full]
        y_full = y_full.loc[valid_full]

        (
            self.model,
            self.model_name,
            self.mae,
            self.rmse,
            self.r2,
        ) = _select_best_model(
            X_full,
            y_full,
            self.random_state,
        )

        self.training_count = len(X_full)

        # -----------------------------------------------------
        # Early-only model: 0h/24h -> 168h
        #
        # Used automatically whenever value_96h is not yet
        # available for a component, matching the literal
        # problem-statement spec of a two-input model.
        # -----------------------------------------------------

        X_early = build_early_prediction_features(
            train_df
        )

        y_early = pd.to_numeric(
            train_df[TARGET_COLUMN],
            errors="coerce",
        )

        X_early = X_early.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        valid_early = (
            X_early.notna().all(axis=1)
            & y_early.notna()
        )

        X_early = X_early.loc[valid_early]
        y_early = y_early.loc[valid_early]

        (
            self.early_model,
            self.early_model_name,
            self.early_mae,
            self.early_rmse,
            self.early_r2,
        ) = _select_best_model(
            X_early,
            y_early,
            self.random_state,
        )

        self.early_training_count = len(
            X_early
        )

        return self

    # ========================================================
    # VALIDATION
    # ========================================================

    def _check_fitted(self) -> None:

        if self.model is None:
            raise RuntimeError(
                "Prediction model is not fitted."
            )

        if self.early_model is None:
            raise RuntimeError(
                "Early-stage (0h/24h) prediction model is not "
                "fitted."
            )

    def _has_96h(
        self,
        df: pd.DataFrame,
    ) -> pd.Series:
        """
        Per-row boolean: is value_96h actually available for this
        component? False for components still mid burn-in that
        only have 0h/24h so far.
        """

        if "value_96h" not in df.columns:

            return pd.Series(
                False,
                index=df.index,
            )

        return pd.to_numeric(
            df["value_96h"],
            errors="coerce",
        ).notna()

    def _prepare_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        missing = [
            column
            for column in PREDICTION_INPUT_COLUMNS
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing prediction features: {missing}"
            )

        X = build_prediction_features(
            df
        )

        X = X[
            self.feature_names
        ]

        X = X.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        for column in self.feature_names:

            if X[column].isna().any():

                median_value = X[
                    column
                ].median()

                if pd.isna(
                    median_value
                ):

                    median_value = 0.0

                X[column] = X[
                    column
                ].fillna(
                    median_value
                )

        return X

    def _prepare_early_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        missing = [
            column
            for column in EARLY_INPUT_COLUMNS
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing early prediction features: {missing}"
            )

        X = build_early_prediction_features(
            df
        )

        X = X[
            self.early_feature_names
        ]

        X = X.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        for column in self.early_feature_names:

            if X[column].isna().any():

                median_value = X[
                    column
                ].median()

                if pd.isna(
                    median_value
                ):

                    median_value = 0.0

                X[column] = X[
                    column
                ].fillna(
                    median_value
                )

        return X

    # ========================================================
    # PREDICT
    # ========================================================

    def predict(
        self,
        df: pd.DataFrame,
    ) -> np.ndarray:

        self._check_fitted()

        has_96h = self._has_96h(df)

        predictions = np.full(
            len(df),
            np.nan,
            dtype=float,
        )

        # Rows with a real value_96h use the more accurate full
        # (0h/24h/96h) model.
        if has_96h.any():

            full_rows = df.loc[has_96h]

            X_full = self._prepare_features(
                full_rows
            )

            predictions[
                has_96h.to_numpy()
            ] = self.model.predict(
                X_full
            )

        # Rows without value_96h (still mid burn-in) use the
        # early-only (0h/24h) model -- the literal problem
        # statement's two-input spec.
        missing_96h = ~has_96h

        if missing_96h.any():

            early_rows = df.loc[
                missing_96h
            ]

            X_early = self._prepare_early_features(
                early_rows
            )

            predictions[
                missing_96h.to_numpy()
            ] = self.early_model.predict(
                X_early
            )

        return predictions

    # ========================================================
    # ONE COMPONENT
    # ========================================================

    def predict_component(
        self,
        row: pd.Series | Dict,
    ) -> PredictionResult:

        if isinstance(
            row,
            pd.Series,
        ):

            record = row.to_dict()

        else:

            record = dict(row)

        component_type = str(
            record.get(
                "component_type",
                "",
            )
        ).strip()

        parameter_name = str(
            record.get(
                "parameter_name",
                "",
            )
        ).strip()

        if not component_type:

            raise ValueError(
                "component_type is required."
            )

        if not parameter_name:

            raise ValueError(
                "parameter_name is required."
            )

        record_df = pd.DataFrame(
            [record]
        )

        has_96h = self._has_96h(
            record_df
        ).iloc[0]

        prediction = self.predict(
            record_df
        )[0]

        if has_96h:

            input_stage = "0h_24h_96h"
            model_name = (
                self.model_name
                or "unknown"
            )

        else:

            input_stage = "0h_24h"
            model_name = (
                self.early_model_name
                or "unknown"
            )

        return PredictionResult(
            predicted_168h=float(
                prediction
            ),
            component_type=component_type,
            parameter_name=parameter_name,
            model_name=model_name,
            input_stage=input_stage,
        )

    # ========================================================
    # SAVE
    # ========================================================

    def save(
        self,
        path: str | Path,
    ) -> None:

        self._check_fitted()

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "model": self.model,

            "model_name": self.model_name,

            "component_type":
                self.component_type,

            "parameter_name":
                self.parameter_name,

            "feature_names":
                self.feature_names,

            "training_count":
                self.training_count,

            "mae":
                self.mae,

            "rmse":
                self.rmse,

            "r2":
                self.r2,

            "random_state":
                self.random_state,

            "early_model":
                self.early_model,

            "early_model_name":
                self.early_model_name,

            "early_feature_names":
                self.early_feature_names,

            "early_training_count":
                self.early_training_count,

            "early_mae":
                self.early_mae,

            "early_rmse":
                self.early_rmse,

            "early_r2":
                self.early_r2,
        }

        joblib.dump(
            payload,
            path,
        )

    # ========================================================
    # LOAD
    # ========================================================

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> "ParameterDriftPredictor":

        path = Path(path)

        if not path.exists():

            raise FileNotFoundError(
                f"Prediction model not found: {path}"
            )

        payload = joblib.load(
            path
        )

        predictor = cls(
            random_state=payload.get(
                "random_state",
                42,
            )
        )

        predictor.model = payload[
            "model"
        ]

        predictor.model_name = payload.get(
            "model_name"
        )

        predictor.component_type = payload.get(
            "component_type"
        )

        predictor.parameter_name = payload.get(
            "parameter_name"
        )

        predictor.feature_names = payload.get(
            "feature_names",
            PREDICTION_FEATURES.copy(),
        )

        predictor.training_count = payload.get(
            "training_count",
            0,
        )

        predictor.mae = payload.get(
            "mae"
        )

        predictor.rmse = payload.get(
            "rmse"
        )

        predictor.r2 = payload.get(
            "r2"
        )

        predictor.early_model = payload.get(
            "early_model"
        )

        predictor.early_model_name = payload.get(
            "early_model_name"
        )

        predictor.early_feature_names = payload.get(
            "early_feature_names",
            EARLY_FEATURES.copy(),
        )

        predictor.early_training_count = payload.get(
            "early_training_count",
            0,
        )

        predictor.early_mae = payload.get(
            "early_mae"
        )

        predictor.early_rmse = payload.get(
            "early_rmse"
        )

        predictor.early_r2 = payload.get(
            "early_r2"
        )

        return predictor


# ============================================================
# MODEL REGISTRY
# ============================================================

class PredictionModelRegistry:

    def __init__(self) -> None:

        self.models: Dict[
            str,
            ParameterDriftPredictor
        ] = {}

    # ========================================================
    # CANONICAL KEY
    # ========================================================

    @staticmethod
    def make_key(
        component_type: str,
        parameter_name: str,
    ) -> str:

        component_type = str(
            component_type
        ).strip()

        parameter_name = str(
            parameter_name
        ).strip()

        return (
            f"{component_type}"
            f"__"
            f"{parameter_name}"
        )

    # ========================================================
    # FILE KEY
    # ========================================================

    @staticmethod
    def make_filename_key(
        component_type: str,
        parameter_name: str,
    ) -> str:

        component_type = str(
            component_type
        ).strip()

        parameter_name = str(
            parameter_name
        ).strip()

        # Filenames use underscores instead of spaces.
        component_type = (
            component_type
            .replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
        )

        parameter_name = (
            parameter_name
            .replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
        )

        return (
            f"{component_type}"
            f"__"
            f"{parameter_name}"
        )

    # ========================================================
    # ADD
    # ========================================================

    def add(
        self,
        component_type: str,
        parameter_name: str,
        predictor: ParameterDriftPredictor,
    ) -> None:

        key = self.make_key(
            component_type,
            parameter_name,
        )

        self.models[key] = predictor

    # ========================================================
    # GET
    # ========================================================

    def get(
        self,
        component_type: str,
        parameter_name: str,
    ) -> ParameterDriftPredictor:

        canonical_key = self.make_key(
            component_type,
            parameter_name,
        )

        # Exact canonical lookup.
        if canonical_key in self.models:

            return self.models[
                canonical_key
            ]

        # Some previously saved models used underscores
        # in the registry key. Support that format too.
        filename_key = (
            self.make_filename_key(
                component_type,
                parameter_name,
            )
        )

        if filename_key in self.models:

            return self.models[
                filename_key
            ]

        # Compare normalized forms defensively.
        normalized_requested = (
            canonical_key
            .lower()
            .replace(" ", "_")
            .strip()
        )

        for key, model in self.models.items():

            normalized_key = (
                key
                .lower()
                .replace(" ", "_")
                .strip()
            )

            if normalized_key == normalized_requested:

                return model

        available = ", ".join(
            sorted(
                self.models.keys()
            )
        )

        raise KeyError(
            f"No prediction model available for "
            f"{canonical_key}. "
            f"Available models: {available}"
        )

    # ========================================================
    # FIT ALL
    # ========================================================

    def fit_all(
        self,
        df: pd.DataFrame,
        random_state: int = 42,
    ) -> "PredictionModelRegistry":

        required = [
            "component_type",
            "parameter_name",
            *PREDICTION_INPUT_COLUMNS,
            TARGET_COLUMN,
        ]

        missing = [
            column
            for column in required
            if column not in df.columns
        ]

        if missing:

            raise ValueError(
                f"Missing prediction columns: {missing}"
            )

        self.models = {}

        grouped = df.groupby(
            [
                "component_type",
                "parameter_name",
            ],
            dropna=False,
        )

        for (
            component_type,
            parameter_name,
        ), group in grouped:

            component_type = str(
                component_type
            ).strip()

            parameter_name = str(
                parameter_name
            ).strip()

            if not component_type:
                continue

            if not parameter_name:
                continue

            predictor = ParameterDriftPredictor(
                random_state=random_state
            )

            predictor.fit(
                group,
                component_type=component_type,
                parameter_name=parameter_name,
            )

            self.add(
                component_type,
                parameter_name,
                predictor,
            )

        if not self.models:

            raise ValueError(
                "No prediction models were trained."
            )

        return self

    # ========================================================
    # PREDICT ONE COMPONENT
    # ========================================================

    def predict_component(
        self,
        row: pd.Series | Dict,
    ) -> PredictionResult:

        if isinstance(
            row,
            pd.Series,
        ):

            record = row.to_dict()

        else:

            record = dict(row)

        component_type = str(
            record.get(
                "component_type",
                "",
            )
        ).strip()

        parameter_name = str(
            record.get(
                "parameter_name",
                "",
            )
        ).strip()

        predictor = self.get(
            component_type,
            parameter_name,
        )

        return predictor.predict_component(
            record
        )

    # ========================================================
    # SAVE
    # ========================================================

    def save(
        self,
        directory: str | Path,
    ) -> None:

        directory = Path(
            directory
        )

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        metadata_rows = []

        for key, predictor in self.models.items():

            component_type = (
                predictor.component_type
                or ""
            )

            parameter_name = (
                predictor.parameter_name
                or ""
            )

            filename_key = (
                self.make_filename_key(
                    component_type,
                    parameter_name,
                )
            )

            model_path = (
                directory
                / f"{filename_key}.joblib"
            )

            predictor.save(
                model_path
            )

            metadata_rows.append(
                {
                    "model_key": key,

                    "filename":
                        model_path.name,

                    "component_type":
                        component_type,

                    "parameter_name":
                        parameter_name,

                    "model":
                        predictor.model_name,

                    "mae":
                        predictor.mae,

                    "rmse":
                        predictor.rmse,

                    "r2":
                        predictor.r2,
                }
            )

        metadata = pd.DataFrame(
            metadata_rows
        )

        metadata.to_csv(
            directory / "registry.csv",
            index=False,
        )

    # ========================================================
    # LOAD
    # ========================================================

    @classmethod
    def load(
        cls,
        directory: str | Path,
    ) -> "PredictionModelRegistry":

        directory = Path(
            directory
        )

        if not directory.exists():

            raise FileNotFoundError(
                f"Prediction model directory "
                f"not found: {directory}"
            )

        registry = cls()

        model_files = sorted(
            directory.glob(
                "*.joblib"
            )
        )

        if not model_files:

            raise FileNotFoundError(
                f"No prediction .joblib models "
                f"found in {directory}"
            )

        for model_path in model_files:

            predictor = (
                ParameterDriftPredictor.load(
                    model_path
                )
            )

            component_type = (
                predictor.component_type
            )

            parameter_name = (
                predictor.parameter_name
            )

            if not component_type:
                raise ValueError(
                    f"Missing component_type "
                    f"in {model_path}"
                )

            if not parameter_name:
                raise ValueError(
                    f"Missing parameter_name "
                    f"in {model_path}"
                )

            registry.add(
                component_type,
                parameter_name,
                predictor,
            )

        return registry


# ============================================================
# TRAINING CONVENIENCE FUNCTION
# ============================================================

def train_prediction_models(
    df: pd.DataFrame,
    output_dir: str | Path = "models/prediction",
    random_state: int = 42,
) -> PredictionModelRegistry:

    registry = (
        PredictionModelRegistry()
    )

    registry.fit_all(
        df,
        random_state=random_state,
    )

    registry.save(
        output_dir
    )

    return registry


# ============================================================
# LOADING CONVENIENCE FUNCTION
# ============================================================

def load_prediction_models(
    model_dir: str | Path = "models/prediction",
) -> PredictionModelRegistry:

    return (
        PredictionModelRegistry.load(
            model_dir
        )
    )


# ============================================================
# STANDALONE
# ============================================================

if __name__ == "__main__":

    print(
        "AegisBurn AI prediction module loaded."
    )

    print(
        "Prediction features:",
        PREDICTION_FEATURES,
    )