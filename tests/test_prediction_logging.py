import pandas as pd

from mlops_lr.prediction_logging import append_prediction_log, prediction_to_record
from mlops_lr.schemas import PredictionRequest


def test_prediction_to_record():
    request = PredictionRequest(
        age=35,
        income=75000,
        loan_amount=25000,
        credit_score=700,
        employment_years=5,
        debt_to_income=0.3,
    )

    record = prediction_to_record(
        request=request,
        prediction=1,
        probability=0.91,
        request_id="request-123",
        trace_id="trace-123",
    )

    assert record["request_id"] == "request-123"
    assert record["trace_id"] == "trace-123"
    assert record["credit_score"] == 700
    assert record["prediction"] == 1
    assert record["probability"] == 0.91


def test_append_prediction_log(tmp_path):
    output_path = tmp_path / "predictions.csv"

    record = {
        "request_id": "request-123",
        "trace_id": "trace-123",
        "age": 35,
        "income": 75000,
        "loan_amount": 25000,
        "credit_score": 700,
        "employment_years": 5,
        "debt_to_income": 0.3,
        "prediction": 1,
        "probability": 0.91,
    }

    append_prediction_log(record, str(output_path))
    append_prediction_log(record, str(output_path))

    logs = pd.read_csv(output_path)

    assert len(logs) == 2
    assert logs.iloc[0]["request_id"] == "request-123"