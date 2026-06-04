import pandas as pd
import pytest

from mlops_lr.monitoring_dataset import create_reference_monitoring_dataset


def test_create_reference_monitoring_dataset(tmp_path):
    processed_data_path = tmp_path / "processed.csv"
    output_path = tmp_path / "reference_monitoring.csv"

    pd.DataFrame(
        [
            {
                "age": 35,
                "income": 75000,
                "loan_amount": 25000,
                "credit_score": 700,
                "employment_years": 5,
                "debt_to_income": 0.3,
                "loan_approved": 1,
            }
        ]
    ).to_csv(processed_data_path, index=False)

    reference = create_reference_monitoring_dataset(
        processed_data_path=str(processed_data_path),
        output_path=str(output_path),
    )

    assert output_path.exists()
    assert list(reference.columns) == [
        "age",
        "income",
        "loan_amount",
        "credit_score",
        "employment_years",
        "debt_to_income",
    ]


def test_create_reference_monitoring_dataset_fails_for_missing_columns(tmp_path):
    processed_data_path = tmp_path / "processed.csv"
    output_path = tmp_path / "reference_monitoring.csv"

    pd.DataFrame(
        [
            {
                "age": 35,
                "income": 75000,
            }
        ]
    ).to_csv(processed_data_path, index=False)

    with pytest.raises(ValueError, match="Missing feature columns"):
        create_reference_monitoring_dataset(
            processed_data_path=str(processed_data_path),
            output_path=str(output_path),
        )
