from mlops_lr.pipeline import run_pipeline
from mlops_lr.retraining_trigger import load_drift_alert, should_trigger_retraining
import json
from pathlib import Path


def save_retraining_result(
    result: dict,
    output_path: str,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(json.dumps(result, indent=2))


def run_retraining_pipeline(
    alert_path: str = "reports/drift_alert.json",
    output_path: str = "reports/retraining_result.json",
) -> dict:
    alert = load_drift_alert(alert_path)

    if not should_trigger_retraining(alert):
        result = {
            "status": "skipped",
            "reason": "drift_not_detected",
        }
        save_retraining_result(result, output_path)
        return result

    metrics = run_pipeline()

    result = {
        "status": "retrained",
        "metrics": metrics,
    }
    save_retraining_result(result, output_path)

    return result


if __name__ == "__main__":
    result = run_retraining_pipeline()
    print(result)
