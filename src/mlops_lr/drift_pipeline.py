from mlops_lr.evidently_drift import (
    generate_data_drift_report,
    generate_prediction_drift_report,
)
from mlops_lr.input_statistics import compute_input_statistics, save_input_statistics
from mlops_lr.monitoring_dataset import create_reference_monitoring_dataset
from mlops_lr.prediction_drift import (
    compute_prediction_statistics,
    save_prediction_statistics,
)
from mlops_lr.drift_alerts import evaluate_drift_alert


def run_drift_monitoring_pipeline() -> None:
    create_reference_monitoring_dataset(
        processed_data_path="data/processed.csv",
        output_path="data/reference_monitoring.csv",
    )

    input_statistics = compute_input_statistics("data/predictions.csv")
    save_input_statistics(
        input_statistics,
        "reports/input_statistics.csv",
    )

    prediction_statistics = compute_prediction_statistics("data/predictions.csv")
    save_prediction_statistics(
        prediction_statistics,
        "reports/prediction_statistics.csv",
    )

    generate_data_drift_report(
        reference_path="data/reference_monitoring.csv",
        current_path="data/predictions.csv",
        html_output_path="reports/data_drift.html",
        json_output_path="reports/data_drift.json",
    )

    generate_prediction_drift_report(
        reference_path="data/predictions.csv",
        current_path="data/predictions.csv",
        html_output_path="reports/prediction_drift.html",
        json_output_path="reports/prediction_drift.json",
    )


if __name__ == "__main__":
    run_drift_monitoring_pipeline()
    evaluate_drift_alert(
        report_path="reports/data_drift.json",
        output_path="reports/drift_alert.json",
    )
