from pathlib import Path

from mlops_lr.release_manifest import build_release_manifest, save_release_manifest


def test_build_release_manifest_contains_production_traceability():
    manifest = build_release_manifest(
        image_name="test-api",
        image_tag="test-tag",
    )

    assert manifest["project"]
    assert manifest["docker"]["image"] == "test-api"
    assert manifest["docker"]["tag"] == "test-tag"
    assert manifest["docker"]["full_name"] == "test-api:test-tag"

    assert "git" in manifest
    assert "mlflow" in manifest
    assert "services" in manifest

    assert manifest["mlflow"]["registered_model_name"]
    assert manifest["mlflow"]["serving_stage"]


def test_save_release_manifest(tmp_path):
    output_path = tmp_path / "release_manifest.json"

    manifest = save_release_manifest(
        output_path=str(output_path),
        image_name="test-api",
        image_tag="test-tag",
    )

    assert output_path.exists()
    assert manifest["docker"]["full_name"] == "test-api:test-tag"