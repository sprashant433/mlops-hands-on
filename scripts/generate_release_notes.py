import argparse
import json
from pathlib import Path


def load_release_manifest(manifest_path: str) -> dict:
    return json.loads(Path(manifest_path).read_text(encoding="utf-8"))


def build_release_notes(manifest: dict) -> str:
    project = manifest["project"]
    generated_at = manifest["generated_at"]
    git = manifest["git"]
    docker = manifest["docker"]
    mlflow = manifest["mlflow"]
    services = manifest["services"]

    return f"""# Release Notes

## Summary

Project `{project}` production release generated at `{generated_at}`.

## Git

- Branch: `{git["branch"]}`
- Commit: `{git["commit"]}`

## Docker

- Image: `{docker["image"]}`
- Tag: `{docker["tag"]}`
- Full image: `{docker["full_name"]}`

## MLflow

- Tracking URI: `{mlflow["tracking_uri"]}`
- Experiment: `{mlflow["experiment_name"]}`
- Registered model: `{mlflow["registered_model_name"]}`
- Serving stage: `{mlflow["serving_stage"]}`

## Services

- API: `{services["api"]}`
- MLflow: `{services["mlflow"]}`
- Prometheus: `{services["prometheus"]}`
- Grafana: `{services["grafana"]}`
- Jaeger: `{services["jaeger"]}`
- Loki: `{services["loki"]}`
"""


def save_release_notes(
    manifest_path: str = "reports/release_manifest.json",
    output_path: str = "reports/release_notes.md",
) -> str:
    manifest = load_release_manifest(manifest_path)
    notes = build_release_notes(manifest)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(notes, encoding="utf-8")

    return notes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest-path",
        default="reports/release_manifest.json",
    )
    parser.add_argument(
        "--output-path",
        default="reports/release_notes.md",
    )
    args = parser.parse_args()

    save_release_notes(
        manifest_path=args.manifest_path,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()