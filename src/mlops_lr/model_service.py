from pathlib import Path

import mlflow.pyfunc
import pandas as pd

from mlops_lr.config import load_config
from mlops_lr.mlflow_utils import configure_mlflow, get_mlflow_client
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

        try:
            self.model = mlflow.pyfunc.load_model(model_uri)
        except OSError:
            self.model = mlflow.pyfunc.load_model(self._resolve_local_model_path())

    def _resolve_local_model_path(self) -> str:
        client = get_mlflow_client()
        model_versions = client.search_model_versions(
            f"name='{self.config.mlflow.registered_model_name}'"
        )

        matching_versions = [
            version
            for version in model_versions
            if version.current_stage == self.config.serving.model_stage
        ]

        if not matching_versions:
            raise ValueError(
                "No model version found for "
                f"{self.config.mlflow.registered_model_name} "
                f"at stage {self.config.serving.model_stage}"
            )

        latest_version = max(
            matching_versions, key=lambda version: int(version.version)
        )
        model_id = latest_version.source.replace("models:/", "")
        tracking_path = Path(
            self.config.mlflow.tracking_uri.replace("file:", "", 1)
        ).resolve()

        matches = list(tracking_path.glob(f"*/models/{model_id}/artifacts"))

        if not matches:
            raise FileNotFoundError(f"Local model artifacts not found for: {model_id}")

        return str(matches[0])

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
