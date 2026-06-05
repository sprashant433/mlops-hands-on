from pathlib import Path

import yaml


def test_production_flow_workflow_exists():
    workflow_path = Path(".github/workflows/production-flow.yml")

    assert workflow_path.exists()


def test_production_flow_workflow_has_manual_trigger():
    workflow = yaml.safe_load(Path(".github/workflows/production-flow.yml").read_text())

    assert workflow["name"] == "Production Flow"
    assert "workflow_dispatch" in workflow[True]


def test_production_flow_workflow_runs_core_steps():
    workflow_text = Path(".github/workflows/production-flow.yml").read_text()

    assert "src/mlops_lr/pipeline.py" in workflow_text
    assert "src/mlops_lr/tuning_pipeline.py" in workflow_text
    assert "src/mlops_lr/drift_pipeline.py" in workflow_text
    assert "src/mlops_lr/retraining_pipeline.py" in workflow_text
    assert "src/mlops_lr/release_manifest.py" in workflow_text
    assert "docker build" in workflow_text
    assert "actions/upload-artifact@v4" in workflow_text
