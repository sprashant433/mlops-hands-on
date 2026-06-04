from mlops_lr.pipeline import run_pipeline
from mlops_lr.retraining_trigger import load_drift_alert, should_trigger_retraining


def run_retraining_pipeline(
    alert_path: str = "reports/drift_alert.json",
) -> dict:
    alert = load_drift_alert(alert_path)

    if not should_trigger_retraining(alert):
        return {
            "status": "skipped",
            "reason": "drift_not_detected",
        }

    metrics = run_pipeline()

    return {
        "status": "retrained",
        "metrics": metrics,
    }


if __name__ == "__main__":
    result = run_retraining_pipeline()
    print(result)
