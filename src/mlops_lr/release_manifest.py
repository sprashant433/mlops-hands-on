import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from mlops_lr.config import load_config


def _run_git_command(command: list[str]) -> Optional[str]:
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None
    except FileNotFoundError:
        return None


def get_git_commit() -> Optional[str]:
    return _run_git_command(["git", "rev-parse", "HEAD"])


def get_git_branch() -> Optional[str]:
    return _run_git_command(["git", "rev-parse", "--abbrev-ref", "HEAD"])


def build_release_manifest(
    image_name: str = "mlops-logistic-regression-api",
    image_tag: Optional[str] = None,
) -> dict:
    config = load_config()

    git_commit = get_git_commit()
    git_branch = get_git_branch()
    resolved_image_tag = image_tag or git_commit or "local"

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project": config.project.name,
        "git": {
            "branch": git_branch,
            "commit": git_commit,
        },
        "docker": {
            "image": image_name,
            "tag": resolved_image_tag,
            "full_name": f"{image_name}:{resolved_image_tag}",
        },
        "mlflow": {
            "tracking_uri": config.mlflow.tracking_uri,
            "experiment_name": config.mlflow.experiment_name,
            "registered_model_name": config.mlflow.registered_model_name,
            "serving_stage": config.serving.model_stage,
        },
        "services": {
            "api": "http://127.0.0.1:8000",
            "mlflow": "http://127.0.0.1:5000",
            "prometheus": "http://127.0.0.1:9090",
            "grafana": "http://127.0.0.1:3000",
            "jaeger": "http://127.0.0.1:16686",
            "loki": "http://127.0.0.1:3100",
        },
    }


def save_release_manifest(
    output_path: str = "reports/release_manifest.json",
    image_name: str = "mlops-logistic-regression-api",
    image_tag: Optional[str] = None,
) -> dict:
    manifest = build_release_manifest(
        image_name=image_name,
        image_tag=image_tag,
    )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return manifest


if __name__ == "__main__":
    save_release_manifest()
