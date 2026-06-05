from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


class ProjectConfig(BaseModel):
    name: str
    version: str


class DataConfig(BaseModel):
    raw_path: str
    processed_path: str
    target_column: str
    test_size: float
    random_state: int


class ModelConfig(BaseModel):
    name: str
    max_iter: int
    output_path: str
    metrics_path: str


class MLflowConfig(BaseModel):
    tracking_uri: str
    experiment_name: str
    registered_model_name: str


class TuningConfig(BaseModel):
    max_evals: int


class ServingConfig(BaseModel):
    host: str
    port: int
    model_stage: str


class MonitoringConfig(BaseModel):
    prediction_log_path: str
    drift_alert_path: str


class TracingConfig(BaseModel):
    otlp_endpoint: str


class AppConfig(BaseModel):
    project: ProjectConfig
    data: DataConfig
    model: ModelConfig
    mlflow: MLflowConfig
    tuning: TuningConfig
    serving: ServingConfig
    monitoring: MonitoringConfig
    tracing: TracingConfig


def load_config(config_path: str = "configs/config.yaml") -> AppConfig:
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with path.open("r") as file:
        config_data: dict[str, Any] = yaml.safe_load(file)

    return AppConfig(**config_data)
