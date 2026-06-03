import mlflow.pyfunc
import pandas as pd

from mlops_lr.config import load_config
from mlops_lr.mlflow_utils import configure_mlflow
from mlops_lr.schemas import PredictionRequest


class ModelService:
    def __init__(self) -> None:
        self.config = load_config()
        configure_mlflow()
        self.model = None

    def load_model(self) -> None:
        model_uri = (
            f"models:/{self.config.mlflow.registered_model_name}"
            f"/{self.config.serving.model_stage}"
        )
        self.model = mlflow.pyfunc.load_model(model_uri)

    def is_ready(self) -> bool:
        return self.model is not None

    def predict(self, request: PredictionRequest) -> tuple[int, float]:
        if self.model is None:
            self.load_model()

        input_data = pd.DataFrame([request.model_dump()])
        predictions = self.model.predict(input_data)

        prediction = int(predictions[0])

        probability = float(prediction)

        if hasattr(self.model, "_model_impl"):
            sklearn_model = getattr(self.model._model_impl, "sklearn_model", None)
            if sklearn_model is not None and hasattr(sklearn_model, "predict_proba"):
                probability = float(sklearn_model.predict_proba(input_data)[0][1])

        return prediction, probability
