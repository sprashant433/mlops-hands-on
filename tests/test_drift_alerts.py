import json

from mlops_lr.drift_alerts import (
    evaluate_drift_alert,
    extract_dataset_drift,
    load_drift_report,
    write_drift_alert,
)


def test_load_drift_report(tmp_path):
    report_path = tmp_path / "data_drift.json"
    report_path.write_text(json.dumps({"dataset_drift": True}))

    report = load_drift_report(str(report_path))

    assert report["dataset_drift"] is True


def test_extract_dataset_drift_true():
    report = {
        "metrics": [
            {
                "result": {
                    "dataset_drift": True,
                }
            }
        ]
    }

    assert extract_dataset_drift(report) is True


def test_extract_dataset_drift_false_when_missing():
    report = {"metrics": []}

    assert extract_dataset_drift(report) is False


def test_write_drift_alert(tmp_path):
    output_path = tmp_path / "drift_alert.json"

    write_drift_alert(
        drift_detected=True,
        output_path=str(output_path),
    )

    alert = json.loads(output_path.read_text())

    assert alert["drift_detected"] is True
    assert alert["status"] == "alert"


def test_evaluate_drift_alert(tmp_path):
    report_path = tmp_path / "data_drift.json"
    output_path = tmp_path / "drift_alert.json"

    report_path.write_text(json.dumps({"dataset_drift": True}))

    alert = evaluate_drift_alert(
        report_path=str(report_path),
        output_path=str(output_path),
    )

    assert alert["drift_detected"] is True
    assert alert["status"] == "alert"
    assert output_path.exists()
