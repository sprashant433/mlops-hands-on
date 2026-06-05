import argparse
import subprocess
from typing import Sequence


def run_command(command: Sequence[str]) -> None:
    subprocess.run(command, check=True)


def run_quality_checks() -> None:
    run_command(["black", "src", "tests"])
    run_command(["flake8", "src", "tests"])
    run_command(["pytest"])


def run_ml_pipeline() -> None:
    run_command(["python", "src/mlops_lr/pipeline.py"])


def run_tuning_pipeline() -> None:
    run_command(["python", "src/mlops_lr/tuning_pipeline.py"])


def run_drift_pipeline() -> None:
    run_command(["python", "src/mlops_lr/drift_pipeline.py"])


def run_retraining_pipeline() -> None:
    run_command(["python", "src/mlops_lr/retraining_pipeline.py"])


def run_release_manifest() -> None:
    run_command(["python", "src/mlops_lr/release_manifest.py"])


def build_docker_image(image_tag: str) -> None:
    run_command(
        [
            "docker",
            "build",
            "-t",
            f"mlops-logistic-regression-api:{image_tag}",
            ".",
        ]
    )


def run_production_flow(image_tag: str = "local", skip_docker: bool = False) -> None:
    run_quality_checks()
    run_ml_pipeline()
    run_tuning_pipeline()
    run_drift_pipeline()
    run_retraining_pipeline()
    run_release_manifest()

    if not skip_docker:
        build_docker_image(image_tag=image_tag)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-tag", default="local")
    parser.add_argument("--skip-docker", action="store_true")
    args = parser.parse_args()

    run_production_flow(
        image_tag=args.image_tag,
        skip_docker=args.skip_docker,
    )


if __name__ == "__main__":
    main()