from mlops_lr.model_service import ModelService
from mlops_lr.schemas import PredictionRequest


def test_model_service_initial_state():
    service = ModelService()

    assert service.is_ready() is False


def test_model_service_predict(monkeypatch):
    class FakeModel:
        def predict(self, input_data):
            return [1]

    service = ModelService()
    service.model = FakeModel()

    request = PredictionRequest(
        age=35,
        income=75000,
        loan_amount=25000,
        credit_score=700,
        employment_years=5,
        debt_to_income=0.3,
    )

    prediction, probability = service.predict(request)

    assert prediction == 1
    assert probability == 1.0
