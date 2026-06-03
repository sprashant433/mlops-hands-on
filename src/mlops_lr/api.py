from fastapi import FastAPI, HTTPException

from mlops_lr.config import load_config
from mlops_lr.model_service import ModelService
from mlops_lr.schemas import PredictionRequest, PredictionResponse


app = FastAPI(
    title="Loan Approval Prediction API",
    version="0.1.0",
)

model_service = ModelService()


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
    try:
        prediction, probability = model_service.predict(request)
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    return PredictionResponse(
        loan_approved=prediction,
        probability=probability,
    )
