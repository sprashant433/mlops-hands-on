import mlflow

from mlops_lr.config import load_config
from mlops_lr.mlflow_utils import configure_mlflow


def test_configure_mlflow():
    config = load_config()

    configure_mlflow()

    assert mlflow.get_tracking_uri() == config.mlflow.tracking_uri
