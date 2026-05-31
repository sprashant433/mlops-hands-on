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

    _, run_id = train_model()
    metrics = evaluate_model(run_id=run_id)

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
