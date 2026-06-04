from pathlib import Path

import pandas as pd


FEATURE_COLUMNS = [
    "age",
    "income",
    "loan_amount",
    "credit_score",
    "employment_years",
    "debt_to_income",
]


def compute_input_statistics(prediction_log_path: str) -> dict[str, dict[str, float]]:
    logs = pd.read_csv(prediction_log_path)

    missing_columns = [
        column for column in FEATURE_COLUMNS if column not in logs.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing feature columns: {missing_columns}")

    statistics: dict[str, dict[str, float]] = {}

    for column in FEATURE_COLUMNS:
        statistics[column] = {
            "mean": float(logs[column].mean()),
            "std": float(logs[column].std()),
            "min": float(logs[column].min()),
            "max": float(logs[column].max()),
        }

    return statistics


def save_input_statistics(
    statistics: dict[str, dict[str, float]],
    output_path: str,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for feature_name, feature_stats in statistics.items():
        row = {"feature": feature_name}
        row.update(feature_stats)
        rows.append(row)

    pd.DataFrame(rows).to_csv(path, index=False)
