import time
from pathlib import Path

import mlflow
import yaml
from yaml.representer import RepresenterError

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


def promote_latest_model_to_stage(model_name: str, stage: str):
    client = get_mlflow_client()
    latest_version = get_latest_model_version(model_name)

    try:
        client.transition_model_version_stage(
            name=model_name,
            version=latest_version.version,
            stage=stage,
            archive_existing_versions=False,
        )
    except RepresenterError:
        _transition_file_store_model_version_stage(
            model_name=model_name,
            version=latest_version.version,
            stage=stage,
        )

    return client.get_model_version(
        name=model_name,
        version=latest_version.version,
    )


def _transition_file_store_model_version_stage(
    model_name: str,
    version: str,
    stage: str,
) -> None:
    config = load_config()
    tracking_path = config.mlflow.tracking_uri.replace("file:", "", 1)
    metadata_path = (
        Path(tracking_path) / "models" / model_name / f"version-{version}" / "meta.yaml"
    )

    if not metadata_path.exists():
        raise FileNotFoundError(f"Model version metadata not found: {metadata_path}")

    with metadata_path.open("r") as file:
        metadata = yaml.safe_load(file)

    metadata["current_stage"] = stage
    metadata["last_updated_timestamp"] = int(time.time() * 1000)

    with metadata_path.open("w") as file:
        yaml.safe_dump(metadata, file, sort_keys=False)
