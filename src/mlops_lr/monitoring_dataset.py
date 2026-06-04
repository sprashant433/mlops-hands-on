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


def create_reference_monitoring_dataset(
    processed_data_path: str,
    output_path: str,
) -> pd.DataFrame:
    data = pd.read_csv(processed_data_path)

    missing_columns = [
        column for column in FEATURE_COLUMNS if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing feature columns: {missing_columns}")

    reference = data[FEATURE_COLUMNS].copy()

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    reference.to_csv(path, index=False)

    return reference
