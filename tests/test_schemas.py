import pytest
from pydantic import ValidationError

from mlops_lr.schemas import PredictionRequest, PredictionResponse


def test_prediction_request_valid():
    request = PredictionRequest(
        age=35,
        income=75000,
        loan_amount=25000,
        credit_score=700,
        employment_years=5,
        debt_to_income=0.3,
    )

    assert request.age == 35
    assert request.credit_score == 700


def test_prediction_request_invalid_credit_score():
    with pytest.raises(ValidationError):
        PredictionRequest(
            age=35,
            income=75000,
            loan_amount=25000,
            credit_score=1000,
            employment_years=5,
            debt_to_income=0.3,
        )


def test_prediction_response_valid():
    response = PredictionResponse(
        loan_approved=1,
        probability=0.82,
    )

    assert response.loan_approved == 1
    assert response.probability == 0.82
