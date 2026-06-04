import pandas as pd
import pytest

from mlops_lr.evidently_drift import (
    FEATURE_COLUMNS,
    generate_data_drift_report,
    load_drift_data,
)


def test_load_drift_data(tmp_path):
    reference_path = tmp_path / "reference.csv"
    current_path = tmp_path / "current.csv"

    data = pd.DataFrame(
        [
            {
                "age": 35,
                "income": 75000,
                "loan_amount": 25000,
                "credit_score": 700,
                "employment_years": 5,
                "debt_to_income": 0.3,
                "extra_column": "ignored",
            }
        ]
    )

    data.to_csv(reference_path, index=False)
    data.to_csv(current_path, index=False)

    reference_data, current_data = load_drift_data(
        reference_path=str(reference_path),
        current_path=str(current_path),
    )

    assert list(reference_data.columns) == FEATURE_COLUMNS
    assert list(current_data.columns) == FEATURE_COLUMNS


def test_load_drift_data_fails_for_missing_current_columns(tmp_path):
    reference_path = tmp_path / "reference.csv"
    current_path = tmp_path / "current.csv"

    reference = pd.DataFrame(
        [
            {
                "age": 35,
                "income": 75000,
                "loan_amount": 25000,
                "credit_score": 700,
                "employment_years": 5,
                "debt_to_income": 0.3,
            }
        ]
    )
    current = pd.DataFrame(
        [
            {
                "age": 35,
                "income": 75000,
            }
        ]
    )

    reference.to_csv(reference_path, index=False)
    current.to_csv(current_path, index=False)

    with pytest.raises(ValueError, match="Missing current columns"):
        load_drift_data(
            reference_path=str(reference_path),
            current_path=str(current_path),
        )


def test_generate_data_drift_report(tmp_path):
    reference_path = tmp_path / "reference.csv"
    current_path = tmp_path / "current.csv"
    html_output_path = tmp_path / "data_drift.html"
    json_output_path = tmp_path / "data_drift.json"

    data = pd.DataFrame(
        [
            {
                "age": 35,
                "income": 75000,
                "loan_amount": 25000,
                "credit_score": 700,
                "employment_years": 5,
                "debt_to_income": 0.3,
            },
            {
                "age": 45,
                "income": 95000,
                "loan_amount": 30000,
                "credit_score": 720,
                "employment_years": 10,
                "debt_to_income": 0.2,
            },
        ]
    )

    data.to_csv(reference_path, index=False)
    data.to_csv(current_path, index=False)

    generate_data_drift_report(
        reference_path=str(reference_path),
        current_path=str(current_path),
        html_output_path=str(html_output_path),
        json_output_path=str(json_output_path),
    )

    assert html_output_path.exists()
    assert json_output_path.exists()
