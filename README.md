# MLOps Logistic Regression

A step-by-step local production-grade MLOps platform built around a logistic regression model.

## Phase 1: Software Engineering Foundation

### Step 1: Project Structure

Created the base repository structure:

```text
mlops-logistic-regression/
├── src/
├── tests/
├── notebooks/
├── data/
├── configs/
├── requirements.txt
├── README.md
└── .gitignore
```

### Step 2: Virtual Environment and Dependencies

Created a Python virtual environment and installed core development dependencies:

- pandas
- numpy
- scikit-learn
- pytest
- pydantic
- pyyaml
- black
- flake8

### Step 3: Python Package Structure

Created the initial Python package:

```text
src/
└── mlops_lr/
    ├── __init__.py
    ├── config.py
    ├── data.py
    ├── features.py
    ├── train.py
    └── evaluate.py
```

### Step 4: Configuration Management

Added YAML-based configuration management using:

- `configs/config.yaml`
- Pydantic models
- `load_config()` helper function

This gives the project a single place to manage paths, model settings, and reproducibility parameters.

### Step 5: Raw Data Generation

Generated a synthetic loan approval dataset.

This step created:

- a reproducible dataset
- numeric applicant features
- a binary target column named `loan_approved`
- a CSV file at `data/raw.csv`

The generated features are:

- `age`
- `income`
- `loan_amount`
- `credit_score`
- `employment_years`
- `debt_to_income`
- `loan_approved`

The raw dataset will be used by later steps for validation, feature engineering, training, and evaluation.

Implementation in `src/mlops_lr/data.py`:

```python
from pathlib import Path

import numpy as np
import pandas as pd

from mlops_lr.config import load_config


def generate_raw_data(n_samples: int = 1000) -> pd.DataFrame:
    config = load_config()
    rng = np.random.default_rng(config.data.random_state)

    age = rng.integers(21, 65, size=n_samples)
    income = rng.normal(75000, 25000, size=n_samples).clip(20000, 200000)
    loan_amount = rng.normal(25000, 10000, size=n_samples).clip(1000, 100000)
    credit_score = rng.normal(680, 80, size=n_samples).clip(300, 850)
    employment_years = rng.integers(0, 40, size=n_samples)
    debt_to_income = rng.uniform(0.05, 0.65, size=n_samples)

    approval_score = (
        0.004 * (credit_score - 650)
        + 0.000015 * (income - loan_amount)
        + 0.03 * employment_years
        - 3.0 * debt_to_income
        + rng.normal(0, 0.5, size=n_samples)
    )

    approval_probability = 1 / (1 + np.exp(-approval_score))
    loan_approved = (approval_probability >= 0.5).astype(int)

    data = pd.DataFrame(
        {
            "age": age,
            "income": income.round(2),
            "loan_amount": loan_amount.round(2),
            "credit_score": credit_score.round(0).astype(int),
            "employment_years": employment_years,
            "debt_to_income": debt_to_income.round(3),
            "loan_approved": loan_approved,
        }
    )

    output_path = Path(config.data.raw_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)

    return data
```

Run:

```bash
PYTHONPATH=src python -c "from mlops_lr.data import generate_raw_data; generate_raw_data()"
```

Check output:

```bash
ls data
head data/raw.csv
```

### Step 6: Data Validation

Add validation for the raw dataset so bad data fails early before feature engineering or model training.

Update `src/mlops_lr/data.py` by adding this function below `generate_raw_data`:

```python
def validate_raw_data(data: pd.DataFrame) -> bool:
    config = load_config()
    target_column = config.data.target_column

    required_columns = {
        "age",
        "income",
        "loan_amount",
        "credit_score",
        "employment_years",
        "debt_to_income",
        target_column,
    }

    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if data.empty:
        raise ValueError("Dataset is empty")

    if data.isnull().sum().sum() > 0:
        raise ValueError("Dataset contains missing values")

    if not data[target_column].isin([0, 1]).all():
        raise ValueError(f"Target column must contain only 0 and 1: {target_column}")

    if not data["age"].between(21, 65).all():
        raise ValueError("Age values are outside expected range")

    if not data["credit_score"].between(300, 850).all():
        raise ValueError("Credit score values are outside expected range")

    if not data["debt_to_income"].between(0, 1).all():
        raise ValueError("Debt-to-income values are outside expected range")

    return True
```

Create `tests/test_data.py`:

```python
import pandas as pd
import pytest

from mlops_lr.data import generate_raw_data, validate_raw_data


def test_generate_raw_data():
    data = generate_raw_data(n_samples=100)

    assert isinstance(data, pd.DataFrame)
    assert len(data) == 100
    assert "loan_approved" in data.columns


def test_validate_raw_data():
    data = generate_raw_data(n_samples=100)

    assert validate_raw_data(data) is True


def test_validate_raw_data_missing_column():
    data = generate_raw_data(n_samples=100)
    data = data.drop(columns=["age"])

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_raw_data(data)
```

Run:

```bash
PYTHONPATH=src pytest
```

### Step 7: Feature Engineering

Created a feature engineering layer in `src/mlops_lr/features.py`.

This step:

- loads raw loan approval data
- validates the raw data before processing
- separates feature columns from the target column
- scales numeric feature columns with `StandardScaler`
- writes processed data to `data/processed.csv`

Implementation:

```python
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

from mlops_lr.config import load_config
from mlops_lr.data import validate_raw_data


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    config = load_config()
    target_column = config.data.target_column

    validate_raw_data(data)

    feature_columns = [
        "age",
        "income",
        "loan_amount",
        "credit_score",
        "employment_years",
        "debt_to_income",
    ]

    features = data[feature_columns].copy()
    target = data[target_column].copy()

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    processed_data = pd.DataFrame(scaled_features, columns=feature_columns)
    processed_data[target_column] = target.values

    return processed_data


def create_processed_data() -> pd.DataFrame:
    config = load_config()

    raw_data = pd.read_csv(config.data.raw_path)
    processed_data = build_features(raw_data)

    output_path = Path(config.data.processed_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_data.to_csv(output_path, index=False)

    return processed_data

Tests:

```python
import pandas as pd

from mlops_lr.data import generate_raw_data
from mlops_lr.features import build_features


def test_build_features():
    raw_data = generate_raw_data(n_samples=100)
    processed_data = build_features(raw_data)

    assert isinstance(processed_data, pd.DataFrame)
    assert len(processed_data) == 100
    assert "loan_approved" in processed_data.columns


def test_build_features_scales_numeric_columns():
    raw_data = generate_raw_data(n_samples=100)
    processed_data = build_features(raw_data)

    feature_columns = [
        "age",
        "income",
        "loan_amount",
        "credit_score",
        "employment_years",
        "debt_to_income",
    ]

    means = processed_data[feature_columns].mean().round(1)

    assert all(means == 0)
```

Run:

```bash
PYTHONPATH=src pytest
PYTHONPATH=src python -c "from mlops_lr.features import create_processed_data; create_processed_data()"
```

### Step 8: Model Training

Created the model training layer in `src/mlops_lr/train.py`.

This step:

- loads processed data from `data/processed.csv`
- splits data into train and test sets
- trains a scikit-learn `LogisticRegression` model
- saves the trained model to `models/logistic_regression.pkl`

Configuration update:

```yaml
model:
  name: LogisticRegression
  max_iter: 1000
  output_path: models/logistic_regression.pkl
```

Implementation:

```python
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from mlops_lr.config import load_config


def train_model() -> LogisticRegression:
    config = load_config()
    target_column = config.data.target_column

    data = pd.read_csv(config.data.processed_path)

    X = data.drop(columns=[target_column])
    y = data[target_column]

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=config.data.test_size,
        random_state=config.data.random_state,
        stratify=y,
    )

    model = LogisticRegression(max_iter=config.model.max_iter)
    model.fit(X_train, y_train)

    output_path = Path(config.model.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)

    return model
```

Tests:

```python
from pathlib import Path

from sklearn.linear_model import LogisticRegression

from mlops_lr.config import load_config
from mlops_lr.data import generate_raw_data
from mlops_lr.features import build_features
from mlops_lr.train import train_model


def test_train_model():
    config = load_config()

    raw_data = generate_raw_data(n_samples=200)
    processed_data = build_features(raw_data)
    processed_data.to_csv(config.data.processed_path, index=False)

    model = train_model()

    assert isinstance(model, LogisticRegression)
    assert Path(config.model.output_path).exists()
```

Run:

```bash
PYTHONPATH=src pytest
PYTHONPATH=src python -c "from mlops_lr.train import train_model; train_model()"
```

### Step 9: Model Evaluation

Created the model evaluation layer in `src/mlops_lr/evaluate.py`.

This step:

- loads processed data from `data/processed.csv`
- recreates the same test split used during training
- loads the trained model from `models/logistic_regression.pkl`
- calculates classification metrics
- writes metrics to `reports/metrics.json`

Metrics:

- accuracy
- precision
- recall
- f1
- roc_auc

Configuration update:

```yaml
model:
  name: LogisticRegression
  max_iter: 1000
  output_path: models/logistic_regression.pkl
  metrics_path: reports/metrics.json
```

Implementation:

```python
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from mlops_lr.config import load_config


def evaluate_model() -> dict[str, float]:
    config = load_config()
    target_column = config.data.target_column

    data = pd.read_csv(config.data.processed_path)

    X = data.drop(columns=[target_column])
    y = data[target_column]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=config.data.test_size,
        random_state=config.data.random_state,
        stratify=y,
    )

    model = joblib.load(config.model.output_path)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

    output_path = Path(config.model.metrics_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as file:
        json.dump(metrics, file, indent=2)

    return metrics
```

Tests:

```python
from pathlib import Path

from mlops_lr.config import load_config
from mlops_lr.data import generate_raw_data
from mlops_lr.evaluate import evaluate_model
from mlops_lr.features import build_features
from mlops_lr.train import train_model


def test_evaluate_model():
    config = load_config()

    raw_data = generate_raw_data(n_samples=300)
    processed_data = build_features(raw_data)
    processed_data.to_csv(config.data.processed_path, index=False)

    train_model()
    metrics = evaluate_model()

    assert Path(config.model.metrics_path).exists()

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics

    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert 0 <= metrics["f1"] <= 1
    assert 0 <= metrics["roc_auc"] <= 1
```

Run:

```bash
PYTHONPATH=src pytest
PYTHONPATH=src python -c "from mlops_lr.evaluate import evaluate_model; print(evaluate_model())"
```

### Step 10: Code Quality with Black and Flake8

Added code formatting and linting.

This step introduced:

- `black` for automatic Python formatting
- `flake8` for lint checks
- `pyproject.toml` for Black configuration
- `.flake8` for Flake8 configuration

Black configuration:

```toml
[tool.black]
line-length = 88
target-version = ["py311"]
include = '\.pyi?$'
```

Flake8 configuration:

```ini
[flake8]
max-line-length = 88
extend-ignore = E203,W503
exclude =
    .git,
    __pycache__,
    .venv,
    venv,
    mlruns,
    data,
    models,
    reports
```

Run:

```bash
black src tests
flake8 src tests
PYTHONPATH=src pytest
```

### Step 11: Logging

Added basic production-style logging.

This step introduced:

- `src/mlops_lr/logger.py`
- reusable `get_logger()` helper
- timestamped logs
- log levels
- module-specific loggers

Logger implementation:

```python
import logging
import sys


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.propagate = False

    return logger
```

Example usage:

```python
from mlops_lr.logger import get_logger


logger = get_logger(__name__)
logger.info("Training model")
```

Run:

```bash
black src tests
flake8 src tests
PYTHONPATH=src pytest
```

### Step 12: Pipeline Entry Point

Created a single pipeline entry point in `src/mlops_lr/pipeline.py`.

The pipeline runs:

```text
generate raw data
→ create processed data
→ train model
→ evaluate model
```

Implementation:

```python
from mlops_lr.data import generate_raw_data
from mlops_lr.evaluate import evaluate_model
from mlops_lr.features import create_processed_data
from mlops_lr.logger import get_logger
from mlops_lr.train import train_model


logger = get_logger(__name__)


def run_pipeline() -> dict[str, float]:
    logger.info("Starting ML pipeline")

    generate_raw_data()
    create_processed_data()
    train_model()
    metrics = evaluate_model()

    logger.info("ML pipeline completed")
    return metrics


if __name__ == "__main__":
    run_pipeline()
```

Tests:

```python
from pathlib import Path

from mlops_lr.config import load_config
from mlops_lr.pipeline import run_pipeline


def test_run_pipeline():
    config = load_config()

    metrics = run_pipeline()

    assert Path(config.data.raw_path).exists()
    assert Path(config.data.processed_path).exists()
    assert Path(config.model.output_path).exists()
    assert Path(config.model.metrics_path).exists()

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics
```

Run:

```bash
black src tests
flake8 src tests
PYTHONPATH=src pytest
PYTHONPATH=src python src/mlops_lr/pipeline.py
```

## Phase 2: Git Workflow

### Step 13: Initialize Git and Create Foundation Commit

Initialized Git for the project and committed the completed Phase 1 foundation.

Commands:

```bash
git init
git status
git add .
git commit -m "feat: complete phase 1 software engineering foundation"
git branch -M main
git tag v0.1-foundation
```

Verification:

```bash
git log --oneline --decorate
git tag
```

Created tag:

```text
v0.1-foundation
```

At this point, the project has a stable foundation checkpoint.

### Step 14: Create Develop Branch

Created the long-lived `develop` branch from `main`.

Command:

```bash
git checkout -b develop
```

Verification:

```bash
git branch
```

Expected branch state:

```text
* develop
  main
```

From now on:

```text
feature branches
↓
develop
↓
main
```

### Step 15: Create Feature Branches

Created feature branches for future isolated work.

Commands:

```bash
git checkout develop
git branch feature/data-pipeline
git branch feature/model-training
git branch feature/api-serving
```

Verification:

```bash
git branch
```

Expected branches:

```text
* develop
  feature/api-serving
  feature/data-pipeline
  feature/model-training
  main
```

Workflow:

```text
feature branch
↓
pull request
↓
develop
↓
pull request
↓
main
```

### Step 16: Simulate Feature Branch PR into Develop

Practiced a local feature branch workflow using `feature/data-pipeline`.

Flow:

```text
feature/data-pipeline
↓
develop
```

This simulates how a pull request would move feature work into the integration branch.

### Step 17: Merge Develop into Main

Practiced the release flow from `develop` into `main`.

Flow:

```text
develop
↓
main
```

Commands:

```bash
git checkout main
git merge --no-ff develop -m "merge: develop into main"
git tag v0.2-git-workflow
git checkout develop
```

Verification:

```bash
git log --oneline --graph --decorate --all --max-count=15
git tag
```

Created tag:

```text
v0.2-git-workflow
```

## Phase 3: MLflow Tracking

### Step 18: Add MLflow Dependency and Config

Added MLflow as the experiment tracking tool.

This step introduced:

- `mlflow` dependency
- MLflow tracking URI
- MLflow experiment name
- Pydantic config support for MLflow settings

Configuration:

```yaml
mlflow:
  tracking_uri: file:./mlruns
  experiment_name: loan-approval-logistic-regression
```

Config model:

```python
class MLflowConfig(BaseModel):
    tracking_uri: str
    experiment_name: str
```

Updated app config:

```python
class AppConfig(BaseModel):
    project: ProjectConfig
    data: DataConfig
    model: ModelConfig
    mlflow: MLflowConfig
```

Run:

```bash
black src tests
flake8 src tests
PYTHONPATH=src pytest
```

### Step 19: Track Training Runs with MLflow

Updated training so each model training run is tracked in MLflow.

This step logs:

- model name
- max iterations
- test size
- random state
- trained scikit-learn model
- serialized local model artifact

Implementation:

```python
import mlflow
import mlflow.sklearn


def train_model() -> LogisticRegression:
    config = load_config()
    target_column = config.data.target_column

    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)

    data = pd.read_csv(config.data.processed_path)

    X = data.drop(columns=[target_column])
    y = data[target_column]

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=config.data.test_size,
        random_state=config.data.random_state,
        stratify=y,
    )

    logger.info("Training model")

    with mlflow.start_run(run_name="logistic-regression-training"):
        mlflow.log_param("model_name", config.model.name)
        mlflow.log_param("max_iter", config.model.max_iter)
        mlflow.log_param("test_size", config.data.test_size)
        mlflow.log_param("random_state", config.data.random_state)

        model = LogisticRegression(max_iter=config.model.max_iter)
        model.fit(X_train, y_train)

        output_path = Path(config.model.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, output_path)

        mlflow.sklearn.log_model(model, name="model")
        mlflow.log_artifact(str(output_path))

    logger.info("Model saved to %s", output_path)

    return model
```

Model signature update:

```python
from mlflow.models.signature import infer_signature

input_example = X_train.head(5)
model_signature = infer_signature(X_train, model.predict(X_train))

mlflow.sklearn.log_model(
    model,
    name="model",
    signature=model_signature,
    input_example=input_example,
)
```

This removes the MLflow warning about missing model signature and gives the model an explicit input/output schema.

Run:

```bash
black src tests
flake8 src tests
PYTHONPATH=src pytest
PYTHONPATH=src python src/mlops_lr/pipeline.py
mlflow ui --backend-store-uri ./mlruns
```

Open:

```text
http://127.0.0.1:5000
```

### Step 20: Log Evaluation Metrics to MLflow

Updated the pipeline so evaluation metrics are logged to the same MLflow run created during training.

This step logs:

- accuracy
- precision
- recall
- f1
- roc_auc
- metrics JSON artifact

Training now returns the active MLflow run ID:

```python
def train_model() -> tuple[LogisticRegression, str]:
    ...
    with mlflow.start_run(run_name="logistic-regression-training"):
        ...
        run_id = mlflow.active_run().info.run_id

    return model, run_id
```

Evaluation accepts an optional MLflow run ID:

```python
from typing import Optional


def evaluate_model(run_id: Optional[str] = None) -> dict[str, float]:
    ...
    if run_id:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(output_path))

    return metrics
```

Pipeline connects both steps:

```python
_, run_id = train_model()
metrics = evaluate_model(run_id=run_id)
```

Run:

```bash
black src tests
flake8 src tests
PYTHONPATH=src pytest
PYTHONPATH=src python src/mlops_lr/pipeline.py
mlflow ui --backend-store-uri ./mlruns
```

### Step 21: Log Confusion Matrix Artifact to MLflow

Added a confusion matrix artifact during model evaluation.

This step introduced:

- `matplotlib`
- confusion matrix generation
- `reports/confusion_matrix.png`
- MLflow artifact logging for the confusion matrix

Implementation:

```python
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

with output_path.open("w") as file:
    json.dump(metrics, file, indent=2)

cm = confusion_matrix(y_test, y_pred)
display = ConfusionMatrixDisplay(confusion_matrix=cm)
display.plot()
plt.title("Confusion Matrix")
plt.savefig(confusion_matrix_path, bbox_inches="tight")
plt.close()

if run_id:
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(output_path))
        mlflow.log_artifact(str(confusion_matrix_path))
```

Run:

```bash
black src tests
flake8 src tests
PYTHONPATH=src pytest
PYTHONPATH=src python src/mlops_lr/pipeline.py
```

### Step 22: Standalone MLflow Evaluation Logging

Updated evaluation so it can log to MLflow with or without a training run ID.

Behavior:

- with `run_id`: logs metrics and artifacts to the existing training run
- without `run_id`: creates a new evaluation run

Implementation:

```python
mlflow.set_tracking_uri(config.mlflow.tracking_uri)
mlflow.set_experiment(config.mlflow.experiment_name)

if run_id:
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(output_path))
        mlflow.log_artifact(str(confusion_matrix_path))
else:
    with mlflow.start_run(run_name="logistic-regression-evaluation"):
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(output_path))
        mlflow.log_artifact(str(confusion_matrix_path))
```

Run:

```bash
black src tests
flake8 src tests
PYTHONPATH=src pytest
PYTHONPATH=src python -c "from mlops_lr.evaluate import evaluate_model; print(evaluate_model())"
```

Troubleshooting:

When running the full pipeline, pass the training run ID into evaluation so evaluation artifacts land on the same training run:

```python
_, run_id = train_model()
metrics = evaluate_model(run_id=run_id)
```

If `evaluate_model()` is called without a `run_id`, MLflow creates a separate evaluation run. In that case, artifacts like `confusion_matrix.png` appear on the evaluation run, not the training run.

### Step 23: MLflow Helper Module

Created a reusable MLflow helper in `src/mlops_lr/mlflow_utils.py`.

This keeps MLflow setup consistent across training, evaluation, and future tracking features.

Implementation:

```python
import mlflow

from mlops_lr.config import load_config


def configure_mlflow() -> None:
    config = load_config()

    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)
```

Usage:

```python
from mlops_lr.mlflow_utils import configure_mlflow

configure_mlflow()
```

Run:

```bash
black src tests
flake8 src tests
PYTHONPATH=src pytest
```

### Step 24: MLflow Tracking Verification Test

Added a focused MLflow tracking test to verify that a training/evaluation run contains:

- logged parameters
- logged metrics
- logged artifacts

Artifacts verified:

- `metrics.json`
- `confusion_matrix.png`
- `logistic_regression.pkl`

Test:

```python
from pathlib import Path

import mlflow

from mlops_lr.config import load_config
from mlops_lr.data import generate_raw_data
from mlops_lr.evaluate import evaluate_model
from mlops_lr.features import build_features
from mlops_lr.mlflow_utils import configure_mlflow
from mlops_lr.train import train_model


def test_mlflow_tracking_run_contains_params_metrics_and_artifacts():
    config = load_config()
    configure_mlflow()

    raw_data = generate_raw_data(n_samples=300)
    processed_data = build_features(raw_data)
    processed_data.to_csv(config.data.processed_path, index=False)

    _, run_id = train_model()
    metrics = evaluate_model(run_id=run_id)

    run = mlflow.get_run(run_id)

    assert run.data.params["model_name"] == config.model.name
    assert run.data.params["max_iter"] == str(config.model.max_iter)

    assert run.data.metrics["accuracy"] == metrics["accuracy"]
    assert run.data.metrics["precision"] == metrics["precision"]
    assert run.data.metrics["recall"] == metrics["recall"]
    assert run.data.metrics["f1"] == metrics["f1"]
    assert run.data.metrics["roc_auc"] == metrics["roc_auc"]

    artifact_uri = run.info.artifact_uri.replace("file://", "")
    artifact_path = Path(artifact_uri)

    assert (artifact_path / "metrics.json").exists()
    assert (artifact_path / "confusion_matrix.png").exists()
    assert (artifact_path / "logistic_regression.pkl").exists()
```

Run:

```bash
black src tests
flake8 src tests
PYTHONPATH=src pytest
```

### Step 25: Tag MLflow Tracking Milestone

Merged the MLflow tracking work into `main` and tagged the Phase 3 milestone.

Commands:

```bash
git checkout main
git merge --no-ff develop -m "merge: mlflow tracking into main"
git tag v0.3-mlflow-tracking
git checkout develop
```

Verification:

```bash
git log --oneline --graph --decorate --all --max-count=20
git tag
```

Created tag:

```text
v0.3-mlflow-tracking
```

## Phase 4: Hyperparameter Tuning

### Step 26: Add Hyperopt Dependency and Tuning Config

Added Hyperopt for hyperparameter tuning.

This step introduced:

- `hyperopt` dependency
- tuning configuration
- Pydantic config support for tuning settings

Configuration:

```yaml
tuning:
  max_evals: 10
```

Config model:

```python
class TuningConfig(BaseModel):
    max_evals: int
```

Updated app config:

```python
class AppConfig(BaseModel):
    project: ProjectConfig
    data: DataConfig
    model: ModelConfig
    mlflow: MLflowConfig
    tuning: TuningConfig
```

Run:

```bash
black src tests
flake8 src tests
PYTHONPATH=src pytest
```

### Step 27: Create Hyperparameter Tuning Module

Created a Hyperopt tuning module in `src/mlops_lr/tune.py`.

This step:

- defines a search space for Logistic Regression
- runs multiple trials with Hyperopt
- logs every trial to MLflow
- selects the best model by F1 score
- logs the best model with signature and input example

Search space:

```python
search_space = {
    "C": hp.loguniform("C", -4, 2),
    "solver": hp.choice("solver", ["liblinear", "lbfgs"]),
    "max_iter": hp.choice("max_iter", [100, 500, 1000]),
}
```

Run:

```bash
black src tests
flake8 src tests
PYTHONPATH=src pytest
PYTHONPATH=src python -c "from mlops_lr.tune import tune_model; print(tune_model())"
```

### Step 28: Make Hyperopt Reproducible

Updated hyperparameter tuning to use the project random seed.

This step makes tuning more reproducible by setting:

- Hyperopt random state
- Logistic Regression random state

Implementation:

```python
import numpy as np

model = LogisticRegression(
    C=params["C"],
    solver=params["solver"],
    max_iter=params["max_iter"],
    random_state=config.data.random_state,
)

best_params = fmin(
    fn=objective,
    space=search_space,
    algo=tpe.suggest,
    max_evals=config.tuning.max_evals,
    trials=trials,
    rstate=np.random.default_rng(config.data.random_state),
)
```

Run:

```bash
black src tests
flake8 src tests
PYTHONPATH=src pytest
```

### Step 29: Decode and Log Best Hyperparameters

Updated Hyperopt tuning so the best parameters are logged as real values instead of choice indexes.

Before:

```text
best_solver_index
best_max_iter_index
```

After:

```text
best_solver
best_max_iter
```

Implementation:

```python
solver_options = ["liblinear", "lbfgs"]
max_iter_options = [100, 500, 1000]

search_space = {
    "C": hp.loguniform("C", -4, 2),
    "solver": hp.choice("solver", solver_options),
    "max_iter": hp.choice("max_iter", max_iter_options),
}

best_decoded_params = {
    "best_C": best_params["C"],
    "best_solver": solver_options[best_params["solver"]],
    "best_max_iter": max_iter_options[best_params["max_iter"]],
}

mlflow.log_params(best_decoded_params)
```

Run:

```bash
black src tests
flake8 src tests
PYTHONPATH=src pytest
PYTHONPATH=src python -c "from mlops_lr.tune import tune_model; print(tune_model())"
```

### Step 30: Add Tuning Pipeline Entry Point

Created a tuning pipeline entry point in `src/mlops_lr/tuning_pipeline.py`.

The tuning pipeline runs:

```text
generate raw data
→ create processed data
→ run Hyperopt tuning
→ log trials and best model to MLflow
```

Implementation:

```python
from mlops_lr.data import generate_raw_data
from mlops_lr.features import create_processed_data
from mlops_lr.logger import get_logger
from mlops_lr.tune import tune_model


logger = get_logger(__name__)


def run_tuning_pipeline() -> tuple:
    logger.info("Starting tuning pipeline")

    generate_raw_data()
    create_processed_data()
    model, metrics, best_params = tune_model()

    logger.info("Tuning pipeline completed")
    return model, metrics, best_params


if __name__ == "__main__":
    run_tuning_pipeline()
```

Run:

```bash
black src tests
flake8 src tests
PYTHONPATH=src pytest
PYTHONPATH=src python src/mlops_lr/tuning_pipeline.py
```

### Step 31: Tag Hyperparameter Tuning Milestone

Merged the Hyperopt tuning work into `main` and tagged the Phase 4 milestone.

Commands:

```bash
git checkout main
git merge --no-ff develop -m "merge: hyperparameter tuning into main"
git tag v0.4-hyperparameter-tuning
git checkout develop
```

Verification:

```bash
git log --oneline --graph --decorate --all --max-count=25
git tag
```

Created tag:

```text
v0.4-hyperparameter-tuning
```

## Phase 5: MLflow Registry

### Step 32: Add Registry Configuration

Added configuration for the MLflow Model Registry.

This step introduced the registered model name:

```text
LoanApprovalModel
```

Configuration:

```yaml
mlflow:
  tracking_uri: file:./mlruns
  experiment_name: loan-approval-logistic-regression
  registered_model_name: LoanApprovalModel
```

Config model:

```python
class MLflowConfig(BaseModel):
    tracking_uri: str
    experiment_name: str
    registered_model_name: str
```

Run:

```bash
black src tests
flake8 src tests
PYTHONPATH=src pytest
```

### Step 33: Register Best Tuned Model

Updated tuning so the best Hyperopt model is registered in the MLflow Model Registry.

Registered model:

```text
LoanApprovalModel
```

MLflow setup now includes the registry URI:

```python
def configure_mlflow() -> None:
    config = load_config()

    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_registry_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)
```

Best model registration:

```python
mlflow.sklearn.log_model(
    best_model,
    name="best_model",
    signature=model_signature,
    input_example=input_example,
    registered_model_name=config.mlflow.registered_model_name,
)
```

Run:

```bash
black src tests
flake8 src tests
PYTHONPATH=src pytest
PYTHONPATH=src python src/mlops_lr/tuning_pipeline.py
```

Then check MLflow UI:

```text
Models → LoanApprovalModel
```