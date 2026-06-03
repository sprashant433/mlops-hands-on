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


def test_resolve_local_model_path(monkeypatch, tmp_path):
    class FakeConfig:
        class MLflow:
            registered_model_name = "LoanApprovalModel"
            tracking_uri = f"file:{tmp_path}"

        class Serving:
            model_stage = "Production"

        mlflow = MLflow()
        serving = Serving()

    class FakeVersion:
        version = "1"
        current_stage = "Production"
        source = "models:/m-test"

    class FakeClient:
        def search_model_versions(self, query):
            return [FakeVersion()]

    artifact_path = tmp_path / "1" / "models" / "m-test" / "artifacts"
    artifact_path.mkdir(parents=True)

    service = ModelService()
    service.config = FakeConfig()

    monkeypatch.setattr(
        "mlops_lr.model_service.get_mlflow_client",
        lambda: FakeClient(),
    )

    assert service._resolve_local_model_path() == str(artifact_path)
