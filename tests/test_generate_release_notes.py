import json

from scripts.generate_release_notes import build_release_notes, save_release_notes


def _sample_manifest() -> dict:
    return {
        "generated_at": "2026-06-05T00:00:00+00:00",
        "project": "mlops-logistic-regression",
        "git": {
            "branch": "develop",
            "commit": "abc123",
        },
        "docker": {
            "image": "mlops-logistic-regression-api",
            "tag": "abc123",
            "full_name": "mlops-logistic-regression-api:abc123",
        },
        "mlflow": {
            "tracking_uri": "file:./mlruns",
            "experiment_name": "loan-approval-logistic-regression",
            "registered_model_name": "LoanApprovalModel",
            "serving_stage": "Production",
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


def test_build_release_notes_contains_release_metadata():
    notes = build_release_notes(_sample_manifest())

    assert "# Release Notes" in notes
    assert "mlops-logistic-regression" in notes
    assert "abc123" in notes
    assert "LoanApprovalModel" in notes
    assert "http://127.0.0.1:3000" in notes


def test_save_release_notes(tmp_path):
    manifest_path = tmp_path / "release_manifest.json"
    output_path = tmp_path / "release_notes.md"

    manifest_path.write_text(json.dumps(_sample_manifest()), encoding="utf-8")

    notes = save_release_notes(
        manifest_path=str(manifest_path),
        output_path=str(output_path),
    )

    assert output_path.exists()
    assert "Release Notes" in notes
    assert "mlops-logistic-regression-api:abc123" in output_path.read_text(
        encoding="utf-8"
    )
