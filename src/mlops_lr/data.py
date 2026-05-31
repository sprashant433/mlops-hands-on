from pathlib import Path

import numpy as np
import pandas as pd

from mlops_lr.config import load_config
from mlops_lr.logger import get_logger


logger = get_logger(__name__)


def generate_raw_data(n_samples: int = 1000) -> pd.DataFrame:
    logger.info("Generating raw data with %s samples", n_samples)
    config = load_config()
    rng = np.random.default_rng(config.data.random_state)

    age = rng.integers(21, 65, size=n_samples)
    income = rng.normal(75000, 25000, size=n_samples).clip(20000, 200000)
    loan_amount = rng.normal(25000, 10000, size=n_samples).clip(1000, 100000)
    credit_score = rng.normal(680, 80, size=n_samples).clip(300, 850)
    employment_years = rng.integers(0, 40, size=n_samples)
    debt_to_income = rng.uniform(0.05, 0.65, size=n_samples)

    approval_score = (
        0.004 * (credit_score - 650)
        + 0.000015 * (income - loan_amount)
        + 0.03 * employment_years
        - 3.0 * debt_to_income
        + rng.normal(0, 0.5, size=n_samples)
    )

    approval_probability = 1 / (1 + np.exp(-approval_score))
    loan_approved = (approval_probability >= 0.5).astype(int)

    data = pd.DataFrame(
        {
            "age": age,
            "income": income.round(2),
            "loan_amount": loan_amount.round(2),
            "credit_score": credit_score.round(0).astype(int),
            "employment_years": employment_years,
            "debt_to_income": debt_to_income.round(3),
            "loan_approved": loan_approved,
        }
    )

    output_path = Path(config.data.raw_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    logger.info("Raw data saved to %s", output_path)
    return data


def validate_raw_data(data: pd.DataFrame) -> bool:
    logger.info("Validating raw data")
    config = load_config()
    target_column = config.data.target_column

    required_columns = {
        "age",
        "income",
        "loan_amount",
        "credit_score",
        "employment_years",
        "debt_to_income",
        target_column,
    }

    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if data.empty:
        raise ValueError("Dataset is empty")

    if data.isnull().sum().sum() > 0:
        raise ValueError("Dataset contains missing values")

    if not data[target_column].isin([0, 1]).all():
        raise ValueError(f"Target column must contain only 0 and 1: {target_column}")

    if not data["age"].between(21, 65).all():
        raise ValueError("Age values are outside expected range")

    if not data["credit_score"].between(300, 850).all():
        raise ValueError("Credit score values are outside expected range")

    if not data["debt_to_income"].between(0, 1).all():
        raise ValueError("Debt-to-income values are outside expected range")
    logger.info("Raw data validation passed")
    return True
