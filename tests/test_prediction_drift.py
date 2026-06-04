import pandas as pd
import pytest

from mlops_lr.prediction_drift import (
    compute_prediction_statistics,
    save_prediction_statistics,
)


def test_compute_prediction_statistics(tmp_path):
    prediction_log_path = tmp_path / "predictions.csv"

    pd.DataFrame(
        [
            {
                "prediction": 1,
                "probability": 0.90,
            },
            {
                "prediction": 0,
                "probability": 0.20,
            },
            {
                "prediction": 1,
                "probability": 0.80,
            },
        ]
    ).to_csv(prediction_log_path, index=False)

    statistics = compute_prediction_statistics(str(prediction_log_path))

    assert statistics["prediction_rate"] == pytest.approx(2 / 3)
    assert statistics["average_probability"] == pytest.approx(0.6333333333)
    assert statistics["min_probability"] == 0.20
    assert statistics["max_probability"] == 0.90


def test_compute_prediction_statistics_fails_for_missing_columns(tmp_path):
    prediction_log_path = tmp_path / "predictions.csv"

    pd.DataFrame(
        [
            {
                "prediction": 1,
            }
        ]
    ).to_csv(prediction_log_path, index=False)

    with pytest.raises(ValueError, match="Missing prediction columns"):
        compute_prediction_statistics(str(prediction_log_path))


def test_save_prediction_statistics(tmp_path):
    output_path = tmp_path / "prediction_statistics.csv"
    statistics = {
        "prediction_rate": 0.5,
        "average_probability": 0.6,
        "min_probability": 0.1,
        "max_probability": 0.9,
    }

    save_prediction_statistics(statistics, str(output_path))

    saved = pd.read_csv(output_path)

    assert saved.iloc[0]["prediction_rate"] == 0.5
    assert saved.iloc[0]["average_probability"] == 0.6
