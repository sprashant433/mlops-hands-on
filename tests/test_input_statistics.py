import pandas as pd
import pytest

from mlops_lr.input_statistics import (
    compute_input_statistics,
    save_input_statistics,
)


def test_compute_input_statistics(tmp_path):
    prediction_log_path = tmp_path / "predictions.csv"

    pd.DataFrame(
        [
            {
                "age": 30,
                "income": 60000,
                "loan_amount": 20000,
                "credit_score": 680,
                "employment_years": 4,
                "debt_to_income": 0.25,
            },
            {
                "age": 40,
                "income": 80000,
                "loan_amount": 30000,
                "credit_score": 720,
                "employment_years": 6,
                "debt_to_income": 0.35,
            },
        ]
    ).to_csv(prediction_log_path, index=False)

    statistics = compute_input_statistics(str(prediction_log_path))

    assert statistics["age"]["mean"] == 35
    assert statistics["income"]["min"] == 60000
    assert statistics["credit_score"]["max"] == 720


def test_compute_input_statistics_fails_for_missing_columns(tmp_path):
    prediction_log_path = tmp_path / "predictions.csv"

    pd.DataFrame(
        [
            {
                "age": 30,
                "income": 60000,
            }
        ]
    ).to_csv(prediction_log_path, index=False)

    with pytest.raises(ValueError, match="Missing feature columns"):
        compute_input_statistics(str(prediction_log_path))


def test_save_input_statistics(tmp_path):
    output_path = tmp_path / "input_statistics.csv"

    statistics = {
        "age": {
            "mean": 35,
            "std": 7.07,
            "min": 30,
            "max": 40,
        }
    }

    save_input_statistics(statistics, str(output_path))

    saved = pd.read_csv(output_path)

    assert saved.iloc[0]["feature"] == "age"
    assert saved.iloc[0]["mean"] == 35
