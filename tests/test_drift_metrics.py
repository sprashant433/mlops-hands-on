import json

from mlops_lr.drift_metrics import read_drift_alert_status, update_drift_metric


def test_read_drift_alert_status_returns_zero_when_missing(tmp_path):
    alert_path = tmp_path / "missing_alert.json"

    assert read_drift_alert_status(str(alert_path)) == 0.0


def test_read_drift_alert_status_returns_one_for_alert(tmp_path):
    alert_path = tmp_path / "drift_alert.json"
    alert_path.write_text(
        json.dumps(
            {
                "drift_detected": True,
                "status": "alert",
            }
        )
    )

    assert read_drift_alert_status(str(alert_path)) == 1.0


def test_read_drift_alert_status_returns_zero_for_ok(tmp_path):
    alert_path = tmp_path / "drift_alert.json"
    alert_path.write_text(
        json.dumps(
            {
                "drift_detected": False,
                "status": "ok",
            }
        )
    )

    assert read_drift_alert_status(str(alert_path)) == 0.0


def test_update_drift_metric(tmp_path):
    alert_path = tmp_path / "drift_alert.json"
    alert_path.write_text(
        json.dumps(
            {
                "drift_detected": True,
                "status": "alert",
            }
        )
    )

    assert update_drift_metric(str(alert_path)) == 1.0
