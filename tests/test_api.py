from fastapi.testclient import TestClient

from mlops_lr.api import app, model_service


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready():
    response = client.get("/ready")

    assert response.status_code == 200
    assert "ready" in response.json()


def test_model_info():
    response = client.get("/model-info")

    assert response.status_code == 200
    assert response.json()["model_name"] == "LoanApprovalModel"


def test_predict(monkeypatch):
    def fake_predict(request):
        return 1, 0.82

    monkeypatch.setattr(model_service, "predict", fake_predict)

    response = client.post(
        "/predict",
        json={
            "age": 35,
            "income": 75000,
            "loan_amount": 25000,
            "credit_score": 700,
            "employment_years": 5,
            "debt_to_income": 0.3,
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "loan_approved": 1,
        "probability": 0.82,
    }


def test_metrics():
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "http_requests_total" in response.text or "http_request" in response.text
