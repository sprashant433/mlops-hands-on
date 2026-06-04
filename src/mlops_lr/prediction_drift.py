from pathlib import Path

import pandas as pd


def compute_prediction_statistics(
    prediction_log_path: str,
) -> dict[str, float]:
    logs = pd.read_csv(prediction_log_path)

    required_columns = ["prediction", "probability"]

    missing_columns = [
        column for column in required_columns if column not in logs.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing prediction columns: {missing_columns}")

    return {
        "prediction_rate": float(logs["prediction"].mean()),
        "average_probability": float(logs["probability"].mean()),
        "min_probability": float(logs["probability"].min()),
        "max_probability": float(logs["probability"].max()),
    }


def save_prediction_statistics(
    statistics: dict[str, float],
    output_path: str,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([statistics]).to_csv(path, index=False)
