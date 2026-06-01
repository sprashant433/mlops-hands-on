import mlflow

from mlops_lr.config import load_config
from mlflow.tracking import MlflowClient


def configure_mlflow() -> None:
    config = load_config()

    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_registry_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)


def get_mlflow_client() -> MlflowClient:
    configure_mlflow()
    return MlflowClient()


def get_latest_model_version(model_name: str):
    client = get_mlflow_client()
    versions = client.search_model_versions(f"name='{model_name}'")

    if not versions:
        raise ValueError(f"No model versions found for: {model_name}")

    return max(versions, key=lambda version: int(version.version))
