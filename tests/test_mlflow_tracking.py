from pathlib import Path

import mlflow

from mlops_lr.data import generate_raw_data
from mlops_lr.evaluate import evaluate_model
from mlops_lr.features import build_features
from mlops_lr.mlflow_utils import configure_mlflow
from mlops_lr.train import train_model
from mlops_lr.config import load_config


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
