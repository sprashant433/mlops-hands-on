import json

from mlops_lr.retraining_trigger import (
    evaluate_retraining_trigger,
    load_drift_alert,
    should_trigger_retraining,
    write_retraining_trigger,
)


def test_load_drift_alert_returns_missing_when_file_does_not_exist(tmp_path):
    alert_path = tmp_path / "missing.json"

    alert = load_drift_alert(str(alert_path))

    assert alert["drift_detected"] is False
    assert alert["status"] == "missing"


def test_should_trigger_retraining_for_drift_detected():
    alert = {
        "drift_detected": True,
        "status": "alert",
    }

    assert should_trigger_retraining(alert) is True


def test_should_not_trigger_retraining_for_ok_alert():
    alert = {
        "drift_detected": False,
        "status": "ok",
    }

    assert should_trigger_retraining(alert) is False


def test_write_retraining_trigger(tmp_path):
    output_path = tmp_path / "retraining_trigger.json"

    write_retraining_trigger(
        should_retrain=True,
        output_path=str(output_path),
    )

    trigger = json.loads(output_path.read_text())

    assert trigger["should_retrain"] is True
    assert trigger["status"] == "triggered"


def test_evaluate_retraining_trigger(tmp_path):
    alert_path = tmp_path / "drift_alert.json"
    output_path = tmp_path / "retraining_trigger.json"

    alert_path.write_text(
        json.dumps(
            {
                "drift_detected": True,
                "status": "alert",
            }
        )
    )

    trigger = evaluate_retraining_trigger(
        alert_path=str(alert_path),
        output_path=str(output_path),
    )

    assert trigger["should_retrain"] is True
    assert trigger["status"] == "triggered"
    assert output_path.exists()
