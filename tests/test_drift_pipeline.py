from mlops_lr.drift_pipeline import run_drift_monitoring_pipeline


def test_run_drift_monitoring_pipeline(monkeypatch):
    calls = []

    def fake_create_reference_monitoring_dataset(
        processed_data_path,
        output_path,
    ):
        calls.append(("reference", processed_data_path, output_path))

    def fake_compute_input_statistics(prediction_log_path):
        calls.append(("input_stats", prediction_log_path))
        return {"age": {"mean": 35}}

    def fake_save_input_statistics(statistics, output_path):
        calls.append(("save_input_stats", statistics, output_path))

    def fake_compute_prediction_statistics(prediction_log_path):
        calls.append(("prediction_stats", prediction_log_path))
        return {"prediction_rate": 0.5}

    def fake_save_prediction_statistics(statistics, output_path):
        calls.append(("save_prediction_stats", statistics, output_path))

    def fake_evaluate_drift_alert(report_path, output_path):
        calls.append(("drift_alert", report_path, output_path))

    def fake_evaluate_retraining_trigger(alert_path, output_path):
        calls.append(("retraining_trigger", alert_path, output_path))

    def fake_generate_data_drift_report(
        reference_path,
        current_path,
        html_output_path,
        json_output_path,
    ):
        calls.append(
            (
                "data_drift",
                reference_path,
                current_path,
                html_output_path,
                json_output_path,
            )
        )

    def fake_generate_prediction_drift_report(
        reference_path,
        current_path,
        html_output_path,
        json_output_path,
    ):
        calls.append(
            (
                "prediction_drift",
                reference_path,
                current_path,
                html_output_path,
                json_output_path,
            )
        )

    monkeypatch.setattr(
        "mlops_lr.drift_pipeline.create_reference_monitoring_dataset",
        fake_create_reference_monitoring_dataset,
    )
    monkeypatch.setattr(
        "mlops_lr.drift_pipeline.compute_input_statistics",
        fake_compute_input_statistics,
    )
    monkeypatch.setattr(
        "mlops_lr.drift_pipeline.save_input_statistics",
        fake_save_input_statistics,
    )
    monkeypatch.setattr(
        "mlops_lr.drift_pipeline.compute_prediction_statistics",
        fake_compute_prediction_statistics,
    )
    monkeypatch.setattr(
        "mlops_lr.drift_pipeline.save_prediction_statistics",
        fake_save_prediction_statistics,
    )
    monkeypatch.setattr(
        "mlops_lr.drift_pipeline.generate_data_drift_report",
        fake_generate_data_drift_report,
    )
    monkeypatch.setattr(
        "mlops_lr.drift_pipeline.generate_prediction_drift_report",
        fake_generate_prediction_drift_report,
    )

    monkeypatch.setattr(
        "mlops_lr.drift_pipeline.evaluate_drift_alert",
        fake_evaluate_drift_alert,
    )

    monkeypatch.setattr(
        "mlops_lr.drift_pipeline.evaluate_retraining_trigger",
        fake_evaluate_retraining_trigger,
    )

    run_drift_monitoring_pipeline()

    assert calls[0] == (
        "reference",
        "data/processed.csv",
        "data/reference_monitoring.csv",
    )
    assert calls[-2] == (
        "drift_alert",
        "reports/data_drift.json",
        "reports/drift_alert.json",
    )

    assert calls[-1] == (
        "retraining_trigger",
        "reports/drift_alert.json",
        "reports/retraining_trigger.json",
    )
