#!/usr/bin/env bash
set -euo pipefail

echo "Running quality checks"
black src tests scripts locustfile.py
flake8 src tests scripts locustfile.py
PYTHONPATH=src pytest

echo "Running ML pipeline"
PYTHONPATH=src python src/mlops_lr/pipeline.py

echo "Running hyperparameter tuning pipeline"
PYTHONPATH=src python src/mlops_lr/tuning_pipeline.py

echo "Running drift monitoring pipeline"
PYTHONPATH=src python src/mlops_lr/drift_pipeline.py

echo "Evaluating retraining trigger"
PYTHONPATH=src python src/mlops_lr/retraining_pipeline.py

echo "Generating release manifest"
PYTHONPATH=src python src/mlops_lr/release_manifest.py

echo "Generating release notes"
python scripts/generate_release_notes.py

echo "Building Docker image"
docker build -t mlops-logistic-regression-api:local .

echo "Production flow completed"