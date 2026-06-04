import json

from mlops_lr.retraining_pipeline import run_retraining_pipeline
from mlops_lr.retraining_pipeline import (
    run_retraining_pipeline,
    save_retraining_result,
)


def test_run_retraining_pipeline_skips_when_no_drift(tmp_path):
    alert_path = tmp_path / "drift_alert.json"
    alert_path.write_text(
        json.dumps(
            {
                "drift_detected": False,
                "status": "ok",
            }
        )
    )

    result = run_retraining_pipeline(alert_path=str(alert_path))

    assert result == {
        "status": "skipped",
        "reason": "drift_not_detected",
    }


def test_run_retraining_pipeline_runs_when_drift_detected(monkeypatch, tmp_path):
    alert_path = tmp_path / "drift_alert.json"
    alert_path.write_text(
        json.dumps(
            {
                "drift_detected": True,
                "status": "alert",
            }
        )
    )

    def fake_run_pipeline():
        return {
            "accuracy": 0.9,
            "precision": 0.8,
        }

    monkeypatch.setattr(
        "mlops_lr.retraining_pipeline.run_pipeline",
        fake_run_pipeline,
    )

    result = run_retraining_pipeline(alert_path=str(alert_path))

    assert result == {
        "status": "retrained",
        "metrics": {
            "accuracy": 0.9,
            "precision": 0.8,
        },
    }


def test_save_retraining_result(tmp_path):
    output_path = tmp_path / "retraining_result.json"
    result = {
        "status": "skipped",
        "reason": "drift_not_detected",
    }

    save_retraining_result(result, str(output_path))

    saved = json.loads(output_path.read_text())

    assert saved == result
