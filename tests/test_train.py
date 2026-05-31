from pathlib import Path

from sklearn.linear_model import LogisticRegression

from mlops_lr.data import generate_raw_data
from mlops_lr.features import build_features
from mlops_lr.train import train_model
from mlops_lr.config import load_config


def test_train_model():
    config = load_config()

    raw_data = generate_raw_data(n_samples=200)
    processed_data = build_features(raw_data)
    processed_data.to_csv(config.data.processed_path, index=False)

    model = train_model()

    assert isinstance(model, LogisticRegression)
    assert Path(config.model.output_path).exists()
