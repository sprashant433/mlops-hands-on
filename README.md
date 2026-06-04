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
prediction_example = pd.DataFrame(
    {
        config.data.target_column: model.predict(X_train),
    }
)
model_signature = infer_signature(X_train, prediction_example)

mlflow.sklearn.log_model(
    model,
    name="model",
    signature=model_signature,
    input_example=input_example,
)
```

This removes the MLflow warning about missing model signature and gives the model an explicit input/output schema.

Schema note:

- input schema contains only feature columns
- output schema contains the predicted target column, `loan_approved`
- old MLflow runs do not change after this update; create a new run to see the named output schema

The target column should not appear in the input schema because inference requests do not provide the label.

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
- logs a named output schema for `loan_approved`

Search space:

```python
search_space = {
    "C": hp.loguniform("C", -4, 2),
    "solver": hp.choice("solver", ["liblinear", "lbfgs"]),
    "max_iter": hp.choice("max_iter", [100, 500, 1000]),
}
```

Named output schema:

```python
prediction_example = pd.DataFrame(
    {
        config.data.target_column: best_model.predict(X_train),
    }
)
model_signature = infer_signature(X_train, prediction_example)
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

### Step 34: Add Registry Utility Functions

Added utility functions for working with the MLflow Model Registry.

Utilities:

- `get_mlflow_client()`
- `get_latest_model_version(model_name)`

Implementation:

```python
from mlflow.tracking import MlflowClient


def get_mlflow_client() -> MlflowClient:
    configure_mlflow()
    return MlflowClient()


def get_latest_model_version(model_name: str):
    client = get_mlflow_client()
    versions = client.search_model_versions(f"name='{model_name}'")

    if not versions:
        raise ValueError(f"No model versions found for: {model_name}")

    return max(versions, key=lambda version: int(version.version))
```

Run:

```bash
black src tests
flake8 src tests
PYTHONPATH=src pytest
```

### Step 35: Promote Latest Model to Staging

Added model promotion support for the MLflow Model Registry.

Promotion function:

```python
def promote_latest_model_to_stage(model_name: str, stage: str):
    client = get_mlflow_client()
    latest_version = get_latest_model_version(model_name)

    client.transition_model_version_stage(
        name=model_name,
        version=latest_version.version,
        stage=stage,
        archive_existing_versions=False,
    )

    return client.get_model_version(
        name=model_name,
        version=latest_version.version,
    )
```

Manual promotion:

```bash
PYTHONPATH=src python -c "from mlops_lr.config import load_config; from mlops_lr.mlflow_utils import promote_latest_model_to_stage; config = load_config(); print(promote_latest_model_to_stage(config.mlflow.registered_model_name, 'Staging'))"
```

Check:

```text
Models → LoanApprovalModel → latest version → Staging
```

MLflow 3 UI note:

MLflow registry stages are deprecated and may not be shown clearly in the UI. To make promotion visible, the promotion helper also sets a model alias matching the stage name.

For staging promotion:

```text
stage: Staging
alias: staging
```

The UI should show the alias on the registered model version:

```text
Models → LoanApprovalModel → version → aliases
```

Promotion now verifies both:

```python
assert promoted_version.current_stage == "Staging"

client = get_mlflow_client()
aliased_version = client.get_model_version_by_alias(
    config.mlflow.registered_model_name,
    "staging",
)

assert aliased_version.version == promoted_version.version
```

### Step 36: Promote Latest Model to Production

Added support for promoting the latest registered model version to `Production`.

Because MLflow registry stages are deprecated and may not be shown clearly in MLflow 3 UI, promotion also sets a model alias.

For production promotion:

```text
stage: Production
alias: production
```

Test:

```python
from mlops_lr.config import load_config
from mlops_lr.data import generate_raw_data
from mlops_lr.features import build_features
from mlops_lr.mlflow_utils import get_mlflow_client, promote_latest_model_to_stage
from mlops_lr.tune import tune_model


def test_promote_latest_model_to_production():
    config = load_config()

    raw_data = generate_raw_data(n_samples=300)
    processed_data = build_features(raw_data)
    processed_data.to_csv(config.data.processed_path, index=False)

    tune_model()

    promoted_version = promote_latest_model_to_stage(
        config.mlflow.registered_model_name,
        "Production",
    )

    assert promoted_version.current_stage == "Production"

    client = get_mlflow_client()
    aliased_version = client.get_model_version_by_alias(
        config.mlflow.registered_model_name,
        "production",
    )

    assert aliased_version.version == promoted_version.version
```

Manual promotion:

```bash
PYTHONPATH=src python -c "from mlops_lr.config import load_config; from mlops_lr.mlflow_utils import promote_latest_model_to_stage; config = load_config(); print(promote_latest_model_to_stage(config.mlflow.registered_model_name, 'Production'))"
```

Check:

```text
Models → LoanApprovalModel → latest version → alias: production
```

### Step 37: Add Rollback Support

Added support for promoting a specific model version to a target stage.

This enables rollback by moving an older known-good version back to `Production`.

Implementation:

```python
def promote_model_version_to_stage(model_name: str, version: str, stage: str):
    client = get_mlflow_client()

    try:
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=False,
        )
    except RepresenterError:
        _transition_file_store_model_version_stage(
            model_name=model_name,
            version=version,
            stage=stage,
        )

    alias = stage.lower()
    client.set_registered_model_alias(
        name=model_name,
        alias=alias,
        version=version,
    )

    return client.get_model_version(
        name=model_name,
        version=version,
    )
```

Manual rollback example:

```bash
PYTHONPATH=src python -c "from mlops_lr.config import load_config; from mlops_lr.mlflow_utils import promote_model_version_to_stage; config = load_config(); print(promote_model_version_to_stage(config.mlflow.registered_model_name, '1', 'Production'))"
```

Check:

```text
Models → LoanApprovalModel → alias: production
```

### Step 38: Tag Model Registry Milestone

Merged the MLflow Model Registry work into `main` and tagged the Phase 5 milestone.

Commands:

```bash
git checkout main
git merge --no-ff develop -m "merge: mlflow model registry into main"
git tag v0.5-model-registry
git checkout develop
```

Verification:

```bash
git log --oneline --graph --decorate --all --max-count=30
git tag
```

Created tag:

```text
v0.5-model-registry
```

## Phase 6: MLflow Projects

### Step 39: Add MLflow Project Files

Added MLflow Project support for reproducible execution.

Files added:

- `MLproject`
- `conda.yaml`
- `src/mlops_lr/project_entry.py`

The project supports two modes:

```text
pipeline
tuning
```

`MLproject`:

```yaml
name: mlops-logistic-regression

conda_env: conda.yaml

entry_points:
  main:
    parameters:
      mode: {type: str, default: pipeline}
    command: "PYTHONPATH=src python src/mlops_lr/project_entry.py --mode {mode}"
```

Project entry point:

```python
import argparse

from mlops_lr.pipeline import run_pipeline
from mlops_lr.tuning_pipeline import run_tuning_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["pipeline", "tuning"],
        default="pipeline",
    )
    args = parser.parse_args()

    if args.mode == "pipeline":
        run_pipeline()
    elif args.mode == "tuning":
        run_tuning_pipeline()


if __name__ == "__main__":
    main()
```

Run:

```bash
mlflow run . -P mode=pipeline --experiment-name loan-approval-logistic-regression
mlflow run . -P mode=tuning --experiment-name loan-approval-logistic-regression
```

### Step 40: Run MLflow Project with Local Environment

Added local execution commands for MLflow Projects using the existing virtual environment.

This avoids creating a separate Conda environment during local development.

Run pipeline mode:

```bash
.venv/bin/mlflow run . -P mode=pipeline --experiment-name loan-approval-logistic-regression --env-manager=local
```

Run tuning mode:

```bash
.venv/bin/mlflow run . -P mode=tuning --experiment-name loan-approval-logistic-regression --env-manager=local
```

Use this for faster local iteration. Keep `conda.yaml` for reproducible environment definitions.

### Step 41: Tag MLflow Projects Milestone

Merged the MLflow Projects work into `main` and tagged the Phase 6 milestone.

Commands:

```bash
git checkout main
git merge --no-ff develop -m "merge: mlflow projects into main"
git tag v0.6-mlflow-projects
git checkout develop
```

Verification:

```bash
git log --oneline --graph --decorate --all --max-count=35
git tag
```

Created tag:

```text
v0.6-mlflow-projects
```

## Phase 7: Build Inference Layer

### Step 42: Add FastAPI Dependencies and Serving Config

Added FastAPI serving dependencies and serving configuration.

Dependencies:

- `fastapi`
- `uvicorn`
- `httpx`

Configuration:

```yaml
serving:
  host: 0.0.0.0
  port: 8000
  model_stage: Production
```

Config model:

```python
class ServingConfig(BaseModel):
    host: str
    port: int
    model_stage: str
```

Run:

```bash
black src tests
flake8 src tests
PYTHONPATH=src pytest
```

### Step 43: Create Prediction Schemas

Created Pydantic schemas for API request and response validation.

Request schema:

```python
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    age: float = Field(..., ge=21, le=65)
    income: float = Field(..., ge=0)
    loan_amount: float = Field(..., ge=0)
    credit_score: float = Field(..., ge=300, le=850)
    employment_years: float = Field(..., ge=0)
    debt_to_income: float = Field(..., ge=0, le=1)
```

Response schema:

```python
class PredictionResponse(BaseModel):
    loan_approved: int
    probability: float
```

Run:

```bash
black src tests
flake8 src tests
PYTHONPATH=src pytest
```

### Step 44: Create Model Service

Created a model service layer in `src/mlops_lr/model_service.py`.

This service:

- configures MLflow
- loads the configured model from MLflow Registry
- falls back to local `mlruns` artifacts when Docker sees host-specific MLflow paths
- checks readiness
- performs prediction
- returns prediction and probability

Model URI pattern:

```text
models:/LoanApprovalModel/Production
```

Implementation:

```python
from pathlib import Path

import mlflow.pyfunc
import pandas as pd

from mlops_lr.config import load_config
from mlops_lr.mlflow_utils import configure_mlflow, get_mlflow_client
from mlops_lr.schemas import PredictionRequest


class ModelService:
    def __init__(self) -> None:
        self.config = load_config()
        configure_mlflow()
        self.model = None

    def load_model(self) -> None:
        model_uri = (
            f"models:/{self.config.mlflow.registered_model_name}"
            f"/{self.config.serving.model_stage}"
        )

        try:
            self.model = mlflow.pyfunc.load_model(model_uri)
        except OSError:
            self.model = mlflow.pyfunc.load_model(self._resolve_local_model_path())

    def _resolve_local_model_path(self) -> str:
        client = get_mlflow_client()
        model_versions = client.search_model_versions(
            f"name='{self.config.mlflow.registered_model_name}'"
        )

        matching_versions = [
            version
            for version in model_versions
            if version.current_stage == self.config.serving.model_stage
        ]

        if not matching_versions:
            raise ValueError(
                "No model version found for "
                f"{self.config.mlflow.registered_model_name} "
                f"at stage {self.config.serving.model_stage}"
            )

        latest_version = max(matching_versions, key=lambda version: int(version.version))
        model_id = latest_version.source.replace("models:/", "")
        tracking_path = Path(
            self.config.mlflow.tracking_uri.replace("file:", "", 1)
        ).resolve()

        matches = list(tracking_path.glob(f"*/models/{model_id}/artifacts"))

        if not matches:
            raise FileNotFoundError(f"Local model artifacts not found for: {model_id}")

        return str(matches[0])

    def is_ready(self) -> bool:
        return self.model is not None

    def predict(self, request: PredictionRequest) -> tuple[int, float]:
        if self.model is None:
            self.load_model()

        input_data = pd.DataFrame([request.model_dump()])
        predictions = self.model.predict(input_data)

        prediction = int(predictions[0])
        probability = float(prediction)

        if hasattr(self.model, "_model_impl"):
            sklearn_model = getattr(self.model._model_impl, "sklearn_model", None)
            if sklearn_model is not None and hasattr(sklearn_model, "predict_proba"):
                probability = float(sklearn_model.predict_proba(input_data)[0][1])

        return prediction, probability
```

Run:

```bash
black src tests
flake8 src tests
PYTHONPATH=src pytest
```

Docker note:

MLflow model versions can store absolute artifact paths from the machine where the
model was registered. In Docker, the registry metadata may still point to the Mac
path, while the mounted artifacts live under `/app/mlruns`. The local fallback
uses the model ID from the registry version and resolves the matching artifact
folder inside the container's configured tracking directory.

### Step 45: Create FastAPI App

Created the FastAPI inference app.

Endpoints:

```text
/health
/ready
/model-info
/predict
```

Run API:

```bash
PYTHONPATH=src uvicorn mlops_lr.api:app --host 0.0.0.0 --port 8000
```

Open Swagger docs:

```text
http://127.0.0.1:8000/docs
```

Example prediction request:

```json
{
  "age": 35,
  "income": 75000,
  "loan_amount": 25000,
  "credit_score": 700,
  "employment_years": 5,
  "debt_to_income": 0.3
}
```

Example response:

```json
{
  "loan_approved": 1,
  "probability": 0.82
}
```

Run:

```bash
black src tests
flake8 src tests
PYTHONPATH=src pytest
```

### Step 46: Add API Server Entry Point

Created a server entry point in `src/mlops_lr/serve.py`.

This uses serving configuration from `configs/config.yaml`.

Implementation:

```python
import uvicorn

from mlops_lr.config import load_config


def serve() -> None:
    config = load_config()

    uvicorn.run(
        "mlops_lr.api:app",
        host=config.serving.host,
        port=config.serving.port,
        reload=False,
    )


if __name__ == "__main__":
    serve()
```

Run:

```bash
black src tests
flake8 src tests
PYTHONPATH=src pytest
PYTHONPATH=src python src/mlops_lr/serve.py
```

Open:

```text
http://127.0.0.1:8000/docs
```

### Step 47: Test Real API Prediction

Tested the FastAPI app with a real MLflow Registry `Production` model.

Promote latest model to production:

```bash
PYTHONPATH=src python -c "from mlops_lr.config import load_config; from mlops_lr.mlflow_utils import promote_latest_model_to_stage; config = load_config(); print(promote_latest_model_to_stage(config.mlflow.registered_model_name, 'Production'))"
```

Start API:

```bash
PYTHONPATH=src python src/mlops_lr/serve.py
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Prediction:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "income": 75000,
    "loan_amount": 25000,
    "credit_score": 700,
    "employment_years": 5,
    "debt_to_income": 0.3
  }'
```

Readiness:

```bash
curl http://127.0.0.1:8000/ready
```

### Step 48: Tag FastAPI Serving Milestone

Merged the FastAPI serving work into `main` and tagged the Phase 7 milestone.

Commands:

```bash
git checkout main
git merge --no-ff develop -m "merge: fastapi serving into main"
git tag v0.6-fastapi-serving
git checkout develop
```

Verification:

```bash
git log --oneline --graph --decorate --all --max-count=40
git tag
```

Created tag:

```text
v0.6-fastapi-serving
```

## Phase 8: Dockerization

### Step 49: Add Dockerfile

Added a Dockerfile for packaging the FastAPI inference service.

Build:

```bash
docker build -t mlops-logistic-regression-api .
```

Run:

```bash
docker run --rm -p 8000:8000 mlops-logistic-regression-api
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Dockerfile:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY configs ./configs
COPY src ./src
RUN mkdir -p ./mlruns

EXPOSE 8000

CMD ["python", "src/mlops_lr/serve.py"]
```

Note:

The Docker image creates an empty `/app/mlruns` directory instead of copying local MLflow artifacts. Local Compose mounts `./mlruns:/app/mlruns`; production deployments should use a mounted registry volume or external MLflow tracking server.

### Step 50: Add Docker Ignore File

Added `.dockerignore` to keep Docker build context smaller.

Important:

`mlruns` is ignored by `.dockerignore` because the Dockerfile no longer copies local MLflow artifacts into the image. Local Compose can still mount `./mlruns:/app/mlruns` at runtime because `.dockerignore` only affects build context, not runtime volumes.

`.dockerignore`:

```dockerignore
.git
.github
.venv
venv
__pycache__
*.pyc
.pytest_cache
.mypy_cache
.ipynb_checkpoints
notebooks
data
reports
models
mlruns
.DS_Store
README.md
```

Build:

```bash
docker build -t mlops-logistic-regression-api .
```

Run:

```bash
docker run --rm -p 8000:8000 mlops-logistic-regression-api
```

### Step 51: Add Docker Compose

Added Docker Compose for running the API service.

`docker-compose.yml`:

```yaml
services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
    image: mlops-logistic-regression-api
    container_name: mlops-logistic-regression-api
    ports:
      - "8000:8000"
    environment:
      - PYTHONPATH=/app/src
    volumes:
      - ./mlruns:/app/mlruns
      - ./configs:/app/configs
```

Run:

```bash
docker compose up --build
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Stop:

```bash
docker compose down
```

### Step 52: Tag Docker Milestone

Merged the Dockerization work into `main` and tagged the Phase 8 milestone.

Commands:

```bash
git checkout main
git merge --no-ff develop -m "merge: dockerization into main"
git tag v0.7-docker
git checkout develop
```

Verification:

```bash
git log --oneline --graph --decorate --all --max-count=45
git tag
```

Created tag:

```text
v0.7-docker
```

## Phase 9: CI Pipeline

### Step 53: Add GitHub Actions CI Workflow

Added a GitHub Actions CI workflow.

Triggers:

- pull request
- manual workflow dispatch

Checks:

- install dependencies
- Black formatting check
- Flake8 linting
- Pytest test suite

Workflow:

```yaml
name: CI

on:
  pull_request:
  workflow_dispatch:

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.9"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Run Black check
        run: black --check src tests

      - name: Run Flake8
        run: flake8 src tests

      - name: Run Pytest
        run: PYTHONPATH=src pytest
```

### Step 54: Add CI Model Pipeline Check

Extended CI to run the ML pipeline and verify generated outputs.

Additional CI steps:

```yaml
      - name: Run ML pipeline
        run: PYTHONPATH=src python src/mlops_lr/pipeline.py

      - name: Verify pipeline outputs
        run: |
          test -f data/raw.csv
          test -f data/processed.csv
          test -f models/logistic_regression.pkl
          test -f reports/metrics.json
          test -f reports/confusion_matrix.png
```

This ensures CI verifies both code quality and end-to-end ML pipeline execution.

### Step 55: Upload CI Artifacts

Updated CI to upload ML pipeline artifacts.

Uploaded artifacts:

- `reports/metrics.json`
- `reports/confusion_matrix.png`
- `models/logistic_regression.pkl`

Workflow step:

```yaml
      - name: Upload pipeline artifacts
        uses: actions/upload-artifact@v4
        with:
          name: pipeline-artifacts
          path: |
            reports/metrics.json
            reports/confusion_matrix.png
            models/logistic_regression.pkl
```

This makes model evaluation outputs available from CI runs.

### Step 56: Add Docker Build Check to CI

Updated CI to verify the API Docker image builds successfully.

Workflow step:

```yaml
      - name: Build Docker image
        run: docker build -t mlops-logistic-regression-api .
```

This catches Dockerfile and dependency issues before deployment.

### Step 57: Tag CI Milestone

Merged the CI pipeline work into `main` and tagged the Phase 9 milestone.

Commands:

```bash
git checkout main
git merge --no-ff develop -m "merge: ci pipeline into main"
git tag v0.8-ci
git checkout develop
```

Verification:

```bash
git log --oneline --graph --decorate --all --max-count=50
git tag
```

Created tag:

```text
v0.8-ci
```

## Phase 10: CD Pipeline

### Step 58: Add GitHub Actions CD Workflow

Added a GitHub Actions CD workflow.

Triggers:

- push to `main`
- manual workflow dispatch

CD steps:

- checkout repository
- set up Docker Buildx
- build Docker image tagged with commit SHA
- run container smoke test
- verify `/health`

Workflow:

```yaml
name: CD

on:
  push:
    branches:
      - main
  workflow_dispatch:

jobs:
  docker-build:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Build Docker image
        run: docker build -t mlops-logistic-regression-api:${{ github.sha }} .

      - name: Smoke test Docker image
        run: |
          docker run -d --name api-test -p 8000:8000 mlops-logistic-regression-api:${{ github.sha }}
          sleep 10
          curl --fail http://127.0.0.1:8000/health
          docker stop api-test
          docker rm api-test
```

### Step 59: Add Docker Image Tagging Strategy

Updated CD to build Docker images with two tags:

- commit SHA tag for immutable release tracking
- `latest` tag for convenient local/runtime usage

Workflow step:

```yaml
      - name: Build Docker image
        run: |
          docker build \
            -t mlops-logistic-regression-api:${{ github.sha }} \
            -t mlops-logistic-regression-api:latest \
            .
```

Smoke tests continue to use the immutable commit SHA tag.

### Step 60: Add CD Artifact Export

Updated CD to export the built Docker image as an artifact.

This simulates producing a deployable release artifact before pushing to a real container registry.

Workflow steps:

```yaml
      - name: Save Docker image artifact
        run: docker save mlops-logistic-regression-api:${{ github.sha }} -o mlops-logistic-regression-api.tar

      - name: Upload Docker image artifact
        uses: actions/upload-artifact@v4
        with:
          name: docker-image
          path: mlops-logistic-regression-api.tar
```

The artifact can later be downloaded and loaded with:

```bash
docker load -i mlops-logistic-regression-api.tar
```

### Step 61: Add Deployment Smoke Test Script

Added a reusable Docker smoke test script.

Script:

```bash
#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${1:-mlops-logistic-regression-api:latest}"
CONTAINER_NAME="mlops-api-smoke-test"

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

docker run -d \
  --name "${CONTAINER_NAME}" \
  -p 8000:8000 \
  "${IMAGE_NAME}"

cleanup() {
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
}

trap cleanup EXIT

sleep 10

curl --fail http://127.0.0.1:8000/health
```

Run locally:

```bash
docker build -t mlops-logistic-regression-api:latest .
./scripts/smoke_test_api.sh mlops-logistic-regression-api:latest
```

CD now reuses the same script:

```yaml
      - name: Smoke test Docker image
        run: ./scripts/smoke_test_api.sh mlops-logistic-regression-api:${{ github.sha }}
```

### Step 62: Tag CD Milestone

Merged the CD pipeline work into `main` and tagged the Phase 10 milestone.

Commands:

```bash
git checkout main
git merge --no-ff develop -m "merge: cd pipeline into main"
git tag v0.9-cd
git checkout develop
```

Verification:

```bash
git log --oneline --graph --decorate --all --max-count=55
git tag
```

Created tag:

```text
v0.9-cd
```

## Phase 11: Monitoring

### Step 63: Add Prometheus Metrics Dependency

Added Prometheus metrics support for FastAPI.

Dependency:

```text
prometheus-fastapi-instrumentator
```

Install:

```bash
pip install -r requirements.txt
```

This dependency will expose application metrics through a `/metrics` endpoint.

### Step 64: Expose `/metrics` Endpoint

Instrumented the FastAPI app with Prometheus metrics.

This adds:

```text
/metrics
```

Implementation:

```python
from prometheus_fastapi_instrumentator import Instrumentator

model_service = ModelService()

Instrumentator().instrument(app).expose(app)
```

Test:

```python
def test_metrics():
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "http_requests_total" in response.text or "http_request" in response.text
```

Run:

```bash
black src tests
flake8 src tests
PYTHONPATH=src pytest
PYTHONPATH=src python src/mlops_lr/serve.py
curl http://127.0.0.1:8000/metrics
```

### Step 65: Add Prometheus Config

Added Prometheus configuration for scraping FastAPI metrics.

Prometheus config:

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: "mlops-api"
    metrics_path: /metrics
    static_configs:
      - targets: ["api:8000"]
```

Docker Compose Prometheus service:

```yaml
  prometheus:
    image: prom/prometheus:latest
    container_name: mlops-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    depends_on:
      - api
```

Run:

```bash
docker compose up --build
```

Open:

```text
http://127.0.0.1:9090
```

Check:

```text
Status → Targets → mlops-api
```

### Step 66: Add Grafana to Docker Compose

Added Grafana for metrics visualization.

Docker Compose Grafana service:

```yaml
  grafana:
    image: grafana/grafana:latest
    container_name: mlops-grafana
    ports:
      - "3000:3000"
    depends_on:
      - prometheus
```

Run:

```bash
docker compose up --build
```

Open Grafana:

```text
http://127.0.0.1:3000
```

Default login:

```text
username: admin
password: admin
```

Add Prometheus data source:

```text
http://prometheus:9090
```

### Step 67: Provision Grafana Data Source

Added Grafana provisioning so Prometheus is automatically configured as a data source.

Provisioning file:

```yaml
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
```

Docker Compose volume:

```yaml
    volumes:
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning
```

Run:

```bash
docker compose up --build
```

Open:

```text
http://127.0.0.1:3000
```

### Step 68: Add Custom Prediction Metrics

Added ML-specific Prometheus metrics.

Metrics:

```text
prediction_requests_total
prediction_errors_total
prediction_probability
```

Implementation:

```python
from prometheus_client import Counter, Histogram

PREDICTION_COUNT = Counter(
    "prediction_requests_total",
    "Total number of prediction requests",
)

PREDICTION_ERRORS = Counter(
    "prediction_errors_total",
    "Total number of prediction errors",
)

PREDICTION_PROBABILITY = Histogram(
    "prediction_probability",
    "Predicted probability distribution",
)
```

Prediction endpoint instrumentation:

```python
PREDICTION_COUNT.inc()

try:
    prediction, probability = model_service.predict(request)
    PREDICTION_PROBABILITY.observe(probability)
except Exception as error:
    PREDICTION_ERRORS.inc()
    raise HTTPException(status_code=500, detail=str(error)) from error
```

Check:

```bash
curl http://127.0.0.1:8000/metrics | grep prediction
```

### Step 69: Add Grafana Dashboard Provisioning

Added a provisioned Grafana dashboard for API monitoring.

Dashboard panels:

- Prediction Requests
- Prediction Errors
- Request Latency
- Prediction Probability

Provisioning file:

```yaml
apiVersion: 1

providers:
  - name: MLOps Dashboards
    orgId: 1
    folder: MLOps
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    options:
      path: /var/lib/grafana/dashboards
```

Grafana dashboard volume:

```yaml
    volumes:
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning
      - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards
```

Open:

```text
http://127.0.0.1:3000
```

Check:

```text
Dashboards → MLOps → MLOps API Monitoring
```

### Step 70: Tag Monitoring Milestone

Merged the monitoring work into `main` and tagged the Phase 11 milestone.

Commands:

```bash
git checkout main
git merge --no-ff develop -m "merge: monitoring into main"
git tag v1.0-monitoring
git checkout develop
```

Verification:

```bash
git log --oneline --graph --decorate --all --max-count=60
git tag
```

Created tag:

```text
v1.0-monitoring
```

## Phase 12: OpenTelemetry

### Step 71: Add OpenTelemetry Dependencies

Added OpenTelemetry dependencies for distributed tracing.

Dependencies:

```text
opentelemetry-api
opentelemetry-sdk
opentelemetry-instrumentation-fastapi
opentelemetry-exporter-otlp
```

Install:

```bash
pip install -r requirements.txt
```

### Step 72: Add OpenTelemetry Tracing Module

Added OpenTelemetry tracing support for FastAPI.

Tracing service name:

```text
mlops-logistic-regression-api
```

Tracing module:

```python
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


def configure_tracing(app) -> None:
    resource = Resource.create(
        {
            "service.name": "mlops-logistic-regression-api",
        }
    )

    provider = TracerProvider(resource=resource)
    processor = BatchSpanProcessor(
        OTLPSpanExporter(
            endpoint="http://otel-collector:4317",
            insecure=True,
        )
    )

    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

    FastAPIInstrumentor.instrument_app(app)
```

FastAPI integration:

```python
configure_tracing(app)
```

### Step 73: Add OpenTelemetry Collector Config

Added OpenTelemetry Collector and Jaeger for distributed tracing.

Collector config:

```yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

processors:
  batch:

exporters:
  otlp/jaeger:
    endpoint: jaeger:4317
    tls:
      insecure: true

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch]
      exporters: [otlp/jaeger]
```

Docker Compose services:

```yaml
  otel-collector:
    image: otel/opentelemetry-collector:latest
    container_name: mlops-otel-collector
    command: ["--config=/etc/otel-collector-config.yml"]
    volumes:
      - ./monitoring/otel-collector-config.yml:/etc/otel-collector-config.yml
    ports:
      - "4317:4317"
      - "4318:4318"
    depends_on:
      - jaeger

  jaeger:
    image: jaegertracing/all-in-one:latest
    container_name: mlops-jaeger
    environment:
      - COLLECTOR_OTLP_ENABLED=true
    ports:
      - "16686:16686"
      - "14250:14250"
```

Open Jaeger:

```text
http://127.0.0.1:16686
```

Search service:

```text
mlops-logistic-regression-api
```

### Step 74: Add Trace IDs to API Responses

Added trace context to API responses for easier correlation with Jaeger.

Helper:

```python
from opentelemetry import trace


def get_current_trace_context() -> dict[str, str]:
    span = trace.get_current_span()
    span_context = span.get_span_context()

    return {
        "trace_id": format(span_context.trace_id, "032x"),
        "span_id": format(span_context.span_id, "016x"),
    }
```

Example response:

```json
{
  "status": "ok",
  "trace_id": "...",
  "span_id": "..."
}
```

Use the `trace_id` to search traces in Jaeger.

### Step 75: Add Manual Spans Around Prediction

Added a custom OpenTelemetry span around model prediction.

Span name:

```text
model_prediction
```

Span attributes:

```text
model.name
model.stage
request.credit_score
request.debt_to_income
prediction.loan_approved
prediction.probability
```

Implementation:

```python
tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("model_prediction") as span:
    config = load_config()

    span.set_attribute("model.name", config.mlflow.registered_model_name)
    span.set_attribute("model.stage", config.serving.model_stage)
    span.set_attribute("request.credit_score", request.credit_score)
    span.set_attribute("request.debt_to_income", request.debt_to_income)

    prediction, probability = model_service.predict(request)

    span.set_attribute("prediction.loan_approved", prediction)
    span.set_attribute("prediction.probability", probability)
```

Check in Jaeger:

```text
Service → mlops-logistic-regression-api → operation → model_prediction
```

### Step 76: Add Structured JSON Logging

Phase 13 starts with application logs.

Goal:

```text
Human-readable message + machine-readable JSON fields
```

Create:

```text
src/mlops_lr/json_logging.py
```

Implementation:

```python
import json
import logging
from datetime import datetime, timezone
from typing import Any


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_record: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created,
                tz=timezone.utc,
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        extra_fields = {
            key: value
            for key, value in record.__dict__.items()
            if key
            not in {
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "taskName",
            }
        }

        log_record.update(extra_fields)

        return json.dumps(log_record, default=str)


def configure_json_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)
```

Update `src/mlops_lr/api.py`:

```python
import logging

from mlops_lr.json_logging import configure_json_logging

configure_json_logging()
logger = logging.getLogger(__name__)
```

Inside `/predict`, log success:

```python
logger.info(
    "prediction_completed",
    extra={
        "loan_approved": prediction,
        "probability": probability,
        "credit_score": request.credit_score,
        "debt_to_income": request.debt_to_income,
        **get_current_trace_context(),
    },
)
```

Inside the exception block, log failure:

```python
logger.exception(
    "prediction_failed",
    extra={
        "credit_score": request.credit_score,
        "debt_to_income": request.debt_to_income,
        **get_current_trace_context(),
    },
)
```

Add test:

```text
tests/test_json_logging.py
```

Implementation:

```python
import json
import logging

from mlops_lr.json_logging import JsonFormatter


def test_json_formatter_outputs_valid_json():
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="mlops_lr.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="prediction_completed",
        args=(),
        exc_info=None,
    )
    record.trace_id = "abc123"
    record.loan_approved = 1

    output = formatter.format(record)
    payload = json.loads(output)

    assert payload["level"] == "INFO"
    assert payload["logger"] == "mlops_lr.test"
    assert payload["message"] == "prediction_completed"
    assert payload["trace_id"] == "abc123"
    assert payload["loan_approved"] == 1
```

Run:

```bash
black src tests
flake8 src tests
PYTHONPATH=src pytest
```

Test JSON logs from Docker:

```bash
docker compose up -d --build api
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "income": 75000,
    "loan_amount": 25000,
    "credit_score": 700,
    "employment_years": 5,
    "debt_to_income": 0.3
  }'
docker logs mlops-logistic-regression-api --tail 20
```

Expected log shape:

```json
{
  "timestamp": "...",
  "level": "INFO",
  "logger": "mlops_lr.api",
  "message": "prediction_completed",
  "loan_approved": 1,
  "probability": 0.91,
  "credit_score": 700,
  "debt_to_income": 0.3,
  "trace_id": "...",
  "span_id": "..."
}
```

### Step 77: Add Loki and Promtail

Added centralized log collection using Loki and Promtail.

Log flow:

```text
FastAPI JSON logs
    ↓
Docker logs
    ↓
Promtail
    ↓
Loki
    ↓
Grafana
```

Promtail config:

```yaml
server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /tmp/positions.yml

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
  - job_name: docker-containers
    docker_sd_configs:
      - host: unix:///var/run/docker.sock
        refresh_interval: 5s

    relabel_configs:
      - source_labels: ["__meta_docker_container_name"]
        target_label: "container"
      - source_labels: ["__meta_docker_container_log_stream"]
        target_label: "stream"
      - source_labels: ["__meta_docker_container_label_com_docker_compose_service"]
        target_label: "compose_service"
```

Docker Compose services:

```yaml
  loki:
    image: grafana/loki:latest
    container_name: mlops-loki
    ports:
      - "3100:3100"
    command: -config.file=/etc/loki/local-config.yaml

  promtail:
    image: grafana/promtail:latest
    container_name: mlops-promtail
    volumes:
      - ./monitoring/promtail-config.yml:/etc/promtail/config.yml
      - /var/run/docker.sock:/var/run/docker.sock
    command: -config.file=/etc/promtail/config.yml
    depends_on:
      - loki
```

Grafana Loki datasource:

```yaml
  - name: Loki
    type: loki
    access: proxy
    url: http://loki:3100
    isDefault: false
```

Run:

```bash
docker compose up -d
docker compose ps
```

Generate logs:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "income": 75000,
    "loan_amount": 25000,
    "credit_score": 700,
    "employment_years": 5,
    "debt_to_income": 0.3
  }'
```

Open Grafana:

```text
http://127.0.0.1:3000
```

Explore logs:

```text
Explore → Loki
```

Query:

```logql
{job="docker"}|= "prediction_completed"
```

### Step 78: Add Log Correlation IDs

Added request correlation IDs to API responses and structured logs.

A request ID helps connect one request across API response headers, JSON logs, Loki, and Jaeger.

Request ID middleware:

```python
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("x-request-id", str(uuid4()))
    request.state.request_id = request_id

    response = await call_next(request)
    response.headers["x-request-id"] = request_id

    return response
```

Prediction endpoint now uses separate names for the request body and HTTP request:

```python
@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest, http_request: Request) -> PredictionResponse:
    PREDICTION_COUNT.inc()
    request_id = http_request.state.request_id
```

Prediction success log:

```python
logger.info(
    "prediction_completed",
    extra={
        "request_id": request_id,
        "loan_approved": prediction,
        "probability": probability,
        "credit_score": payload.credit_score,
        "debt_to_income": payload.debt_to_income,
        **get_current_trace_context(),
    },
)
```

Prediction failure log:

```python
logger.exception(
    "prediction_failed",
    extra={
        "request_id": request_id,
        "credit_score": payload.credit_score,
        "debt_to_income": payload.debt_to_income,
        **get_current_trace_context(),
    },
)
```

Run:

```bash
black src tests
flake8 src tests
PYTHONPATH=src pytest
```

Rebuild API:

```bash
docker compose up -d --build api
```

Test with custom request ID:

```bash
curl -i -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -H "x-request-id: test-request-123" \
  -d '{
    "age": 35,
    "income": 75000,
    "loan_amount": 25000,
    "credit_score": 700,
    "employment_years": 5,
    "debt_to_income": 0.3
  }'
```

Expected response header:

```text
x-request-id: test-request-123
```

Check Docker logs:

```bash
docker logs mlops-logistic-regression-api --tail 20
```

Expected log field:

```json
"request_id": "test-request-123"
```

Search in Grafana Loki:

```logql
{compose_service="api"} |= "test-request-123"
```

### Step 79: Add Loki Log Dashboard Panel

Added a Loki logs panel to the existing Grafana API dashboard.

This panel shows prediction logs generated by the FastAPI service and collected through Loki.

Log flow:

```text
FastAPI JSON logs
    ↓
Docker logs
    ↓
Promtail
    ↓
Loki
    ↓
Grafana dashboard
```

Files updated:

```text
monitoring/grafana/dashboards/mlops-api-dashboard.json
README.md
```

Dashboard file:

```text
monitoring/grafana/dashboards/mlops-api-dashboard.json
```

Existing last panel before this step:

```json
{
  "type": "timeseries",
  "title": "Prediction Probability",
  "gridPos": { "x": 12, "y": 8, "w": 12, "h": 8 },
  "targets": [
    {
      "expr": "histogram_quantile(0.95, rate(prediction_probability_bucket[5m]))",
      "legendFormat": "p95 probability",
      "refId": "A"
    }
  ]
}
```

Added a comma after the last panel and appended this logs panel inside the top-level `panels` array:

```json
{
  "type": "logs",
  "title": "API Prediction Logs",
  "gridPos": { "x": 0, "y": 16, "w": 24, "h": 8 },
  "datasource": {
    "type": "loki",
    "uid": "Loki"
  },
  "targets": [
    {
      "datasource": {
        "type": "loki",
        "uid": "Loki"
      },
      "expr": "{job=\"docker\"} |= \"prediction_completed\"",
      "queryType": "range",
      "refId": "A"
    }
  ],
  "options": {
    "showTime": true,
    "showLabels": true,
    "showCommonLabels": false,
    "wrapLogMessage": true,
    "prettifyLogMessage": true,
    "enableLogDetails": true,
    "sortOrder": "Descending",
    "dedupStrategy": "none"
  }
}
```

The end of the dashboard JSON should look like this:

```json
    {
      "type": "timeseries",
      "title": "Prediction Probability",
      "gridPos": { "x": 12, "y": 8, "w": 12, "h": 8 },
      "targets": [
        {
          "expr": "histogram_quantile(0.95, rate(prediction_probability_bucket[5m]))",
          "legendFormat": "p95 probability",
          "refId": "A"
        }
      ]
    },
    {
      "type": "logs",
      "title": "API Prediction Logs",
      "gridPos": { "x": 0, "y": 16, "w": 24, "h": 8 },
      "datasource": {
        "type": "loki",
        "uid": "Loki"
      },
      "targets": [
        {
          "datasource": {
            "type": "loki",
            "uid": "Loki"
          },
          "expr": "{job=\"docker\"} |= \"prediction_completed\"",
          "queryType": "range",
          "refId": "A"
        }
      ],
      "options": {
        "showTime": true,
        "showLabels": true,
        "showCommonLabels": false,
        "wrapLogMessage": true,
        "prettifyLogMessage": true,
        "enableLogDetails": true,
        "sortOrder": "Descending",
        "dedupStrategy": "none"
      }
    }
  ]
}
```

Panel query:

```logql
{job="docker"} |= "prediction_completed"
```

Why this query is used:

```text
Promtail currently labels Docker logs with job="docker".
```

The active Promtail config uses:

```yaml
labels:
  job: docker
  __path__: /var/lib/docker/containers/*/*-json.log
```

So do not use this query unless Promtail is changed to create `compose_service` labels:

```logql
{compose_service="api"} |= "prediction_completed"
```

Validate dashboard JSON:

```bash
python -m json.tool monitoring/grafana/dashboards/mlops-api-dashboard.json > /tmp/mlops-dashboard.json
```

Restart Grafana so it reloads the dashboard:

```bash
docker compose restart grafana
```

Check Grafana logs:

```bash
docker logs mlops-grafana --tail 50
```

Generate a fresh prediction log:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -H "x-request-id: dashboard-test-123" \
  -d '{
    "age": 35,
    "income": 75000,
    "loan_amount": 25000,
    "credit_score": 700,
    "employment_years": 5,
    "debt_to_income": 0.3
  }'
```

Check Loki directly in Grafana Explore:

```text
Grafana → Explore → Loki
```

Use this query:

```logql
{job="docker"} |= "dashboard-test-123"
```

Open the dashboard:

```text
Grafana → Dashboards → MLOps API Monitoring
```

Set time range:

```text
Last 15 minutes
```

Expected dashboard panel:

```text
API Prediction Logs
```

Expected result:

```text
The API Prediction Logs panel shows prediction_completed log lines.
```

If the panel is empty:

```text
1. Confirm the API generated a log.
2. Confirm Loki has the log in Explore.
3. Confirm the dashboard query uses {job="docker"}.
4. Confirm the Grafana time range includes the log time.
5. Restart Grafana after dashboard JSON changes.
```

Check API logs:

```bash
docker logs mlops-logistic-regression-api --tail 20
```

Check Loki labels:

```bash
curl http://127.0.0.1:3100/loki/api/v1/labels
```

Expected Loki labels include:

```text
job
filename
service_name
```

Commit changes:

```bash
git add .
git commit -m "feat: add loki logs dashboard panel"
```