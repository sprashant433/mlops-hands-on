import json
from pathlib import Path
from typing import Optional

import joblib
import matplotlib
import mlflow
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


from mlops_lr.config import load_config
from mlops_lr.logger import get_logger
from mlops_lr.mlflow_utils import configure_mlflow

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix  # noqa: E402


logger = get_logger(__name__)


def evaluate_model(run_id: Optional[str] = None) -> dict[str, float]:
    logger.info("Evaluating model")
    config = load_config()
    configure_mlflow()
    target_column = config.data.target_column

    data = pd.read_csv(config.data.processed_path)

    X = data.drop(columns=[target_column])
    y = data[target_column]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=config.data.test_size,
        random_state=config.data.random_state,
        stratify=y,
    )

    model = joblib.load(config.model.output_path)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

    output_path = Path(config.model.metrics_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    confusion_matrix_path = output_path.parent / "confusion_matrix.png"

    cm = confusion_matrix(y_test, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=cm)
    display.plot()
    plt.title("Confusion Matrix")
    plt.savefig(confusion_matrix_path, bbox_inches="tight")
    plt.close()

    with output_path.open("w") as file:
        json.dump(metrics, file, indent=2)

    if run_id:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(output_path))
            mlflow.log_artifact(str(confusion_matrix_path))
    else:
        with mlflow.start_run(run_name="logistic-regression-evaluation"):
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(output_path))
            mlflow.log_artifact(str(confusion_matrix_path))

    logger.info("Evaluation metrics saved to %s", output_path)
    return metrics
