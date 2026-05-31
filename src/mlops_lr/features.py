from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

from mlops_lr.config import load_config
from mlops_lr.data import validate_raw_data
from mlops_lr.logger import get_logger


logger = get_logger(__name__)


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    logger.info("Building features")
    config = load_config()
    target_column = config.data.target_column

    validate_raw_data(data)

    feature_columns = [
        "age",
        "income",
        "loan_amount",
        "credit_score",
        "employment_years",
        "debt_to_income",
    ]

    features = data[feature_columns].copy()
    target = data[target_column].copy()

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    processed_data = pd.DataFrame(scaled_features, columns=feature_columns)
    processed_data[target_column] = target.values
    logger.info("Feature engineering completed")
    return processed_data


def create_processed_data() -> pd.DataFrame:
    config = load_config()

    raw_data = pd.read_csv(config.data.raw_path)
    processed_data = build_features(raw_data)

    output_path = Path(config.data.processed_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_data.to_csv(output_path, index=False)
    logger.info("Processed data saved to %s", output_path)
    return processed_data
