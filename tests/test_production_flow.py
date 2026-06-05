from mlops_lr.production_flow import run_production_flow


def test_run_production_flow_runs_all_stages(monkeypatch):
    commands = []

    def fake_run_command(command):
        commands.append(command)

    monkeypatch.setattr("mlops_lr.production_flow.run_command", fake_run_command)

    run_production_flow(image_tag="test-tag")

    assert ["black", "src", "tests"] in commands
    assert ["flake8", "src", "tests"] in commands
    assert ["pytest"] in commands
    assert ["python", "src/mlops_lr/pipeline.py"] in commands
    assert ["python", "src/mlops_lr/tuning_pipeline.py"] in commands
    assert ["python", "src/mlops_lr/drift_pipeline.py"] in commands
    assert ["python", "src/mlops_lr/retraining_pipeline.py"] in commands
    assert ["python", "src/mlops_lr/release_manifest.py"] in commands
    assert [
        "docker",
        "build",
        "-t",
        "mlops-logistic-regression-api:test-tag",
        ".",
    ] in commands


def test_run_production_flow_can_skip_docker(monkeypatch):
    commands = []

    def fake_run_command(command):
        commands.append(command)

    monkeypatch.setattr("mlops_lr.production_flow.run_command", fake_run_command)

    run_production_flow(skip_docker=True)

    assert not any(command[0] == "docker" for command in commands)
