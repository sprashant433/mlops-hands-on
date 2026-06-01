from mlops_lr.config import load_config
from mlops_lr.data import generate_raw_data
from mlops_lr.features import build_features
from mlops_lr.mlflow_utils import (
    get_latest_model_version,
    get_mlflow_client,
)
from mlops_lr.tune import tune_model


def test_get_mlflow_client():
    client = get_mlflow_client()

    assert client is not None


def test_get_latest_model_version():
    config = load_config()

    raw_data = generate_raw_data(n_samples=300)
    processed_data = build_features(raw_data)
    processed_data.to_csv(config.data.processed_path, index=False)

    tune_model()

    latest_version = get_latest_model_version(config.mlflow.registered_model_name)

    assert latest_version.name == config.mlflow.registered_model_name
    assert int(latest_version.version) >= 1
