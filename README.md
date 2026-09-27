# AegisBurn AI

**AI-Driven Anomaly Detection in Component Burn-In & Screening**  
ISRO Problem Statement 26170 — Hackathon Prototype

## Overview

AegisBurn AI analyzes electronic component burn-in test data to:

- Detect components whose behavior is anomalous relative to their lot
- Predict 168-hour parameter values from early measurements (24–72 h)
- Calculate drift risk and surface explainable PASS / REVIEW / REJECT decisions

## Project Structure

```
aegisburn-ai/
├── data/
│   ├── raw/          # Raw burn-in CSV/JSON uploads
│   └── processed/    # Cleaned, feature-ready datasets
├── models/           # Trained model artifacts (.joblib)
├── src/              # Core Python modules
├── tests/            # Unit & integration tests
├── app.py            # GUI / dashboard entry point
└── train.py          # Model training pipeline
```

## Requirements

- Python 3.11+

## Setup

```bash
cd aegisburn-ai
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Usage (scaffold only)

```bash
# Verify project loads
python train.py
python app.py
```

## Status

This repository contains **initial project scaffolding only**. ML models, data generation, and the dashboard GUI are not yet implemented.

## License

Hackathon prototype — internal use.
