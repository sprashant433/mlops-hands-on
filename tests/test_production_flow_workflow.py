from pathlib import Path

import yaml


def test_production_flow_workflow_exists():
    workflow_path = Path(".github/workflows/production-flow.yml")

    assert workflow_path.exists()


def test_production_flow_workflow_has_manual_trigger():
    workflow = yaml.safe_load(Path(".github/workflows/production-flow.yml").read_text())

    assert workflow["name"] == "Production Flow"
    assert "workflow_dispatch" in workflow[True]


def test_production_flow_workflow_runs_python_orchestrator():
    workflow_text = Path(".github/workflows/production-flow.yml").read_text()

    assert "src/mlops_lr/production_flow.py" in workflow_text
    assert "--image-tag" in workflow_text
    assert "${{ github.sha }}" in workflow_text
