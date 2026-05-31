import pandas as pd
import pytest

from mlops_lr.data import generate_raw_data, validate_raw_data


def test_generate_raw_data():
    data = generate_raw_data(n_samples=100)

    assert isinstance(data, pd.DataFrame)
    assert len(data) == 100
    assert "loan_approved" in data.columns


def test_validate_raw_data():
    data = generate_raw_data(n_samples=100)

    assert validate_raw_data(data) is True


def test_validate_raw_data_missing_column():
    data = generate_raw_data(n_samples=100)
    data = data.drop(columns=["age"])

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_raw_data(data)
