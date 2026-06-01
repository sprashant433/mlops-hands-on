from pathlib import Path
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from mlflow.models.signature import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from mlops_lr.config import load_config
from mlops_lr.logger import get_logger
from mlops_lr.mlflow_utils import configure_mlflow


logger = get_logger(__name__)


def tune_model() -> tuple[LogisticRegression, dict[str, float], dict]:
    config = load_config()
    configure_mlflow()

    target_column = config.data.target_column
    data = pd.read_csv(config.data.processed_path)

    X = data.drop(columns=[target_column])
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.data.test_size,
        random_state=config.data.random_state,
        stratify=y,
    )

    solver_options = ["liblinear", "lbfgs"]
    max_iter_options = [100, 500, 1000]

    search_space = {
        "C": hp.loguniform("C", -4, 2),
        "solver": hp.choice("solver", solver_options),
        "max_iter": hp.choice("max_iter", max_iter_options),
    }

    best_model = None
    best_metrics = None
    best_f1 = -1.0

    def objective(params: dict) -> dict:
        nonlocal best_model, best_metrics, best_f1

        with mlflow.start_run(
            run_name="hyperopt-logistic-regression-trial", nested=True
        ):
            model = LogisticRegression(
                C=params["C"],
                solver=params["solver"],
                max_iter=params["max_iter"],
                random_state=config.data.random_state,
            )
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_prob),
            }

            mlflow.log_params(params)
            mlflow.log_metrics(metrics)

            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_model = model
                best_metrics = metrics

            return {"loss": -metrics["f1"], "status": STATUS_OK}

    trials = Trials()

    with mlflow.start_run(run_name="hyperopt-logistic-regression"):
        best_params = fmin(
            fn=objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=config.tuning.max_evals,
            trials=trials,
            rstate=np.random.default_rng(config.data.random_state),
        )

        best_decoded_params = {
            "best_C": best_params["C"],
            "best_solver": solver_options[best_params["solver"]],
            "best_max_iter": max_iter_options[best_params["max_iter"]],
        }

        mlflow.log_params(best_decoded_params)

        mlflow.log_metrics(
            {f"best_{key}": value for key, value in best_metrics.items()}
        )

        output_path = Path(config.model.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, output_path)

        input_example = X_train.iloc[[0]]
        model_signature = infer_signature(X_train, best_model.predict(X_train))

        mlflow.sklearn.log_model(
            best_model,
            name="best_model",
            signature=model_signature,
            input_example=input_example,
            registered_model_name=config.mlflow.registered_model_name,
        )
        mlflow.log_artifact(str(output_path))

    logger.info("Best tuning metrics: %s", best_metrics)

    return best_model, best_metrics, best_decoded_params


if __name__ == "__main__":
    tune_model()
