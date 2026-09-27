"""Project-wide paths, constants, and configuration."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Burn-in test parameters (168-hour screening window)
BURN_IN_HOURS = 168
EARLY_MEASUREMENT_HOURS = (24, 48, 72)

# Anomaly / risk thresholds (tuned during training)
ANOMALY_SCORE_THRESHOLD = 0.75
DRIFT_RISK_THRESHOLD = 0.60

# Random seed for reproducibility
RANDOM_SEED = 42
