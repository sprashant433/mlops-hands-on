from sklearn.linear_model import LogisticRegression

from mlops_lr.config import load_config
from mlops_lr.data import generate_raw_data
from mlops_lr.features import build_features
from mlops_lr.tune import tune_model


def test_tune_model():
    config = load_config()

    raw_data = generate_raw_data(n_samples=300)
    processed_data = build_features(raw_data)
    processed_data.to_csv(config.data.processed_path, index=False)

    model, metrics = tune_model()

    assert isinstance(model, LogisticRegression)
    assert "f1" in metrics
    assert 0 <= metrics["f1"] <= 1
