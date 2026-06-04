import json
from pathlib import Path


def load_drift_alert(alert_path: str) -> dict:
    path = Path(alert_path)

    if not path.exists():
        return {
            "drift_detected": False,
            "status": "missing",
        }

    return json.loads(path.read_text())


def should_trigger_retraining(alert: dict) -> bool:
    return alert.get("drift_detected") is True or alert.get("status") == "alert"


def write_retraining_trigger(
    should_retrain: bool,
    output_path: str,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    trigger = {
        "should_retrain": should_retrain,
        "status": "triggered" if should_retrain else "skipped",
    }

    path.write_text(json.dumps(trigger, indent=2))


def evaluate_retraining_trigger(
    alert_path: str,
    output_path: str,
) -> dict:
    alert = load_drift_alert(alert_path)
    should_retrain = should_trigger_retraining(alert)

    write_retraining_trigger(
        should_retrain=should_retrain,
        output_path=output_path,
    )

    return {
        "should_retrain": should_retrain,
        "status": "triggered" if should_retrain else "skipped",
    }
