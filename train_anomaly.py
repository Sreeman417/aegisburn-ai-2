"""
AegisBurn AI — Anomaly Model Training Entrypoint
==================================================

ALWAYS retrain the anomaly detector by running THIS script:

    python train_anomaly.py

Do NOT run `python src/anomaly_detection.py` directly.

WHY THIS MATTERS:

When a .py file is executed directly (`python src/anomaly_detection.py`),
Python sets that module's __name__ to "__main__". Any class defined in
that file -- including ParameterAnomalyDetector -- then gets pickled
with its module recorded as "__main__" instead of "src.anomaly_detection".

That pickle can then ONLY be unpickled by a process whose own __main__
namespace happens to also define ParameterAnomalyDetector. Loading it
from anywhere else (the FastAPI backend, evaluate_anomaly.py, a fresh
shell, etc.) fails with:

    AttributeError: Can't get attribute 'ParameterAnomalyDetector'
    on <module '__main__' from '...'>

This script avoids the problem entirely: it IMPORTS anomaly_detection
as a normal module (never executes it directly), so its classes keep
their correct, portable module path (src.anomaly_detection) in the
saved pickle, loadable from any script.
"""

from src.anomaly_detection import main


if __name__ == "__main__":
    main()