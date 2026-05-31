from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

from mlops_lr.config import load_config
from mlops_lr.logger import get_logger


logger = get_logger(__name__)


def train_model() -> LogisticRegression:
    logger.info("Training model")
    config = load_config()
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

    model = LogisticRegression(max_iter=config.model.max_iter)
    model.fit(X_train, y_train)

    output_path = Path(config.model.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    logger.info("Model saved to %s", output_path)
    return model
