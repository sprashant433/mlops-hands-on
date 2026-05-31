from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

from mlops_lr.config import load_config
from mlops_lr.logger import get_logger

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlops_lr.mlflow_utils import configure_mlflow


logger = get_logger(__name__)


def train_model() -> tuple[LogisticRegression, str]:
    config = load_config()
    configure_mlflow()
    target_column = config.data.target_column

    data = pd.read_csv(config.data.processed_path)

    X = data.drop(columns=[target_column])
    y = data[target_column]

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=config.data.test_size,
        random_state=config.data.random_state,
        stratify=y,
    )

    logger.info("Training model")

    with mlflow.start_run(run_name="logistic-regression-training"):
        mlflow.log_param("model_name", config.model.name)
        mlflow.log_param("max_iter", config.model.max_iter)
        mlflow.log_param("test_size", config.data.test_size)
        mlflow.log_param("random_state", config.data.random_state)

        model = LogisticRegression(max_iter=config.model.max_iter)
        model.fit(X_train, y_train)

        input_example = X_train.iloc[[0]]
        model_signature = infer_signature(X_train, model.predict(X_train))

        output_path = Path(config.model.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, output_path)

        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=model_signature,
            input_example=input_example,
        )
        mlflow.log_artifact(str(output_path))

        run_id = mlflow.active_run().info.run_id

    logger.info("Model saved to %s", output_path)

    return model, run_id
