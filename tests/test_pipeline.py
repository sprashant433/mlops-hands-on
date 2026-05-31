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
