import json
from pathlib import Path

from prometheus_client import Gauge


DRIFT_DETECTED = Gauge(
    "model_drift_detected",
    "Whether model/data drift has been detected. 1 means drift alert, 0 means ok.",
)


def read_drift_alert_status(alert_path: str) -> float:
    path = Path(alert_path)

    if not path.exists():
        return 0.0

    alert = json.loads(path.read_text())

    if alert.get("drift_detected") is True:
        return 1.0

    if alert.get("status") == "alert":
        return 1.0

    return 0.0


def update_drift_metric(alert_path: str) -> float:
    drift_value = read_drift_alert_status(alert_path)
    DRIFT_DETECTED.set(drift_value)

    return drift_value
