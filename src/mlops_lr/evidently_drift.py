from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset


FEATURE_COLUMNS = [
    "age",
    "income",
    "loan_amount",
    "credit_score",
    "employment_years",
    "debt_to_income",
]


def load_drift_data(
    reference_path: str,
    current_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference_data = pd.read_csv(reference_path)
    current_data = pd.read_csv(current_path)

    missing_reference_columns = [
        column for column in FEATURE_COLUMNS if column not in reference_data.columns
    ]
    missing_current_columns = [
        column for column in FEATURE_COLUMNS if column not in current_data.columns
    ]

    if missing_reference_columns:
        raise ValueError(f"Missing reference columns: {missing_reference_columns}")

    if missing_current_columns:
        raise ValueError(f"Missing current columns: {missing_current_columns}")

    return (
        reference_data[FEATURE_COLUMNS].copy(),
        current_data[FEATURE_COLUMNS].copy(),
    )


def generate_data_drift_report(
    reference_path: str,
    current_path: str,
    html_output_path: str,
    json_output_path: str,
) -> None:
    reference_data, current_data = load_drift_data(
        reference_path=reference_path,
        current_path=current_path,
    )

    report = Report([DataDriftPreset()])
    snapshot = report.run(
        reference_data=reference_data,
        current_data=current_data,
    )

    html_path = Path(html_output_path)
    json_path = Path(json_output_path)

    html_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    snapshot.save_html(str(html_path))
    snapshot.save_json(str(json_path))
