from pathlib import Path


def test_production_flow_script_exists():
    script = Path("scripts/run_production_flow.sh")

    assert script.exists()
    assert script.read_text().startswith("#!/usr/bin/env bash")


def test_production_flow_script_calls_python_orchestrator():
    script = Path("scripts/run_production_flow.sh").read_text()

    assert "src/mlops_lr/production_flow.py" in script
    assert "--image-tag" in script
    assert "IMAGE_TAG" in script
