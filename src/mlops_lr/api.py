from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram
from mlops_lr.tracing import configure_tracing

from mlops_lr.config import load_config
from mlops_lr.model_service import ModelService
from mlops_lr.schemas import PredictionRequest, PredictionResponse


app = FastAPI(
    title="Loan Approval Prediction API",
    version="0.1.0",
)

model_service = ModelService()
Instrumentator().instrument(app).expose(app)
configure_tracing(app)
PREDICTION_COUNT = Counter(
    "prediction_requests_total",
    "Total number of prediction requests",
)

PREDICTION_ERRORS = Counter(
    "prediction_errors_total",
    "Total number of prediction errors",
)

PREDICTION_PROBABILITY = Histogram(
    "prediction_probability",
    "Predicted probability distribution",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, bool]:
    return {"ready": model_service.is_ready()}


@app.get("/model-info")
def model_info() -> dict[str, str]:
    config = load_config()

    return {
        "model_name": config.mlflow.registered_model_name,
        "model_stage": config.serving.model_stage,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    PREDICTION_COUNT.inc()

    try:
        prediction, probability = model_service.predict(request)
        PREDICTION_PROBABILITY.observe(probability)
    except Exception as error:
        PREDICTION_ERRORS.inc()
        raise HTTPException(status_code=500, detail=str(error)) from error

    return PredictionResponse(
        loan_approved=prediction,
        probability=probability,
    )
