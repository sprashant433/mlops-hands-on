from pathlib import Path

import yaml


def test_docker_compose_has_mlflow_service():
    compose = yaml.safe_load(Path("docker-compose.yml").read_text())

    mlflow = compose["services"]["mlflow"]

    assert mlflow["container_name"] == "mlops-mlflow"
    assert "5000:5000" in mlflow["ports"]
    assert "./mlruns:/app/mlruns" in mlflow["volumes"]


def test_mlflow_service_runs_tracking_server():
    compose = yaml.safe_load(Path("docker-compose.yml").read_text())

    command = compose["services"]["mlflow"]["command"]

    assert "mlflow" in command
    assert "server" in command
    assert "--backend-store-uri" in command
    assert "/app/mlruns" in command
    assert "--default-artifact-root" in command
    assert "--port" in command
    assert "5000" in command


def test_api_depends_on_mlflow():
    compose = yaml.safe_load(Path("docker-compose.yml").read_text())

    assert "mlflow" in compose["services"]["api"]["depends_on"]
