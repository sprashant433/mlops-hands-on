from pathlib import Path
from typing import Union

import pandas as pd

from mlops_lr.schemas import PredictionRequest


def prediction_to_record(
    request: PredictionRequest,
    prediction: int,
    probability: float,
    request_id: str,
    trace_id: str,
) -> dict[str, Union[float, int, str]]:
    return {
        "request_id": request_id,
        "trace_id": trace_id,
        "age": request.age,
        "income": request.income,
        "loan_amount": request.loan_amount,
        "credit_score": request.credit_score,
        "employment_years": request.employment_years,
        "debt_to_income": request.debt_to_income,
        "prediction": prediction,
        "probability": probability,
    }


def append_prediction_log(record: dict, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    row = pd.DataFrame([record])

    if path.exists():
        row.to_csv(path, mode="a", header=False, index=False)
    else:
        row.to_csv(path, index=False)
