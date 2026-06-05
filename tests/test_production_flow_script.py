from pathlib import Path


def test_production_flow_script_exists():
    script = Path("scripts/run_production_flow.sh")

    assert script.exists()
    assert script.read_text().startswith("#!/usr/bin/env bash")


def test_production_flow_script_runs_core_stages():
    script = Path("scripts/run_production_flow.sh").read_text()

    assert "pytest" in script
    assert "src/mlops_lr/pipeline.py" in script
    assert "src/mlops_lr/tuning_pipeline.py" in script
    assert "src/mlops_lr/drift_pipeline.py" in script
    assert "src/mlops_lr/retraining_pipeline.py" in script
    assert "src/mlops_lr/release_manifest.py" in script
    assert "docker build" in script