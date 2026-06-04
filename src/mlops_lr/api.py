from fastapi import FastAPI, HTTPException, Request
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram
from mlops_lr.tracing import configure_tracing
from opentelemetry import trace
from uuid import uuid4


from mlops_lr.config import load_config
from mlops_lr.model_service import ModelService
from mlops_lr.schemas import PredictionRequest, PredictionResponse
import logging

from mlops_lr.json_logging import configure_json_logging

configure_json_logging()
logger = logging.getLogger(__name__)

tracer = trace.get_tracer(__name__)


def get_current_trace_context() -> dict[str, str]:
    span = trace.get_current_span()
    span_context = span.get_span_context()

    return {
        "trace_id": format(span_context.trace_id, "032x"),
        "span_id": format(span_context.span_id, "016x"),
    }


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


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("x-request-id", str(uuid4()))
    request.state.request_id = request_id

    response = await call_next(request)
    response.headers["x-request-id"] = request_id

    return response


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        **get_current_trace_context(),
    }


@app.get("/ready")
def ready() -> dict[str, bool]:
    return {"ready": model_service.is_ready()}


@app.get("/model-info")
def model_info() -> dict[str, str]:
    config = load_config()

    return {
        "model_name": config.mlflow.registered_model_name,
        "model_stage": config.serving.model_stage,
        **get_current_trace_context(),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest, http_request: Request) -> PredictionResponse:
    PREDICTION_COUNT.inc()
    request_id = http_request.state.request_id
    try:
        with tracer.start_as_current_span("model_prediction") as span:
            config = load_config()

            span.set_attribute("model.name", config.mlflow.registered_model_name)
            span.set_attribute("model.stage", config.serving.model_stage)
            span.set_attribute("request.credit_score", payload.credit_score)
            span.set_attribute("request.debt_to_income", payload.debt_to_income)

            prediction, probability = model_service.predict(payload)

            span.set_attribute("prediction.loan_approved", prediction)
            span.set_attribute("prediction.probability", probability)

            PREDICTION_PROBABILITY.observe(probability)
            logger.info(
                "prediction_completed",
                extra={
                    "request_id": request_id,
                    "loan_approved": prediction,
                    "probability": probability,
                    "credit_score": payload.credit_score,
                    "debt_to_income": payload.debt_to_income,
                    **get_current_trace_context(),
                },
            )
    except Exception as error:
        PREDICTION_ERRORS.inc()
        logger.exception(
            "prediction_failed",
            extra={
                "request_id": request_id,
                "credit_score": payload.credit_score,
                "debt_to_income": payload.debt_to_income,
                **get_current_trace_context(),
            },
        )
        raise HTTPException(status_code=500, detail=str(error)) from error

    return PredictionResponse(
        loan_approved=prediction,
        probability=probability,
    )
