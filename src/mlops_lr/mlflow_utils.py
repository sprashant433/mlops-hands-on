import mlflow

from mlops_lr.config import load_config


def configure_mlflow() -> None:
    config = load_config()

    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)
