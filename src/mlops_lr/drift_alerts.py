import json
from pathlib import Path


def load_drift_report(report_path: str) -> dict:
    return json.loads(Path(report_path).read_text())


def extract_dataset_drift(report: dict) -> bool:
    report_text = json.dumps(report).lower()

    if "dataset_drift" not in report_text:
        return False

    return "true" in report_text


def write_drift_alert(
    drift_detected: bool,
    output_path: str,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    alert = {
        "drift_detected": drift_detected,
        "status": "alert" if drift_detected else "ok",
    }

    path.write_text(json.dumps(alert, indent=2))


def evaluate_drift_alert(
    report_path: str,
    output_path: str,
) -> dict:
    report = load_drift_report(report_path)
    drift_detected = extract_dataset_drift(report)

    write_drift_alert(
        drift_detected=drift_detected,
        output_path=output_path,
    )

    return {
        "drift_detected": drift_detected,
        "status": "alert" if drift_detected else "ok",
    }
