import pandas as pd

from mlops_lr.data import generate_raw_data
from mlops_lr.features import build_features


def test_build_features():
    raw_data = generate_raw_data(n_samples=100)
    processed_data = build_features(raw_data)

    assert isinstance(processed_data, pd.DataFrame)
    assert len(processed_data) == 100
    assert "loan_approved" in processed_data.columns


def test_build_features_scales_numeric_columns():
    raw_data = generate_raw_data(n_samples=100)
    processed_data = build_features(raw_data)

    feature_columns = [
        "age",
        "income",
        "loan_amount",
        "credit_score",
        "employment_years",
        "debt_to_income",
    ]

    means = processed_data[feature_columns].mean().round(1)

    assert all(means == 0)
