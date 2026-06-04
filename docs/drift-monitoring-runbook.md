# Drift Monitoring Runbook

## Purpose

This runbook explains how to run local data and prediction drift monitoring.

Drift monitoring compares reference training data against current prediction logs.

## Inputs

Reference data:

```text
data/reference_monitoring.csv
```

Current prediction logs:

```text
data/predictions.csv
```

Processed training data:

```text
data/processed.csv
```

## Outputs

Input statistics:

```text
reports/input_statistics.csv
```

Prediction statistics:

```text
reports/prediction_statistics.csv
```

Evidently data drift report:

```text
reports/data_drift.html
reports/data_drift.json
```

Evidently prediction drift report:

```text
reports/prediction_drift.html
reports/prediction_drift.json
```

## Run Full Drift Pipeline

```bash
PYTHONPATH=src python src/mlops_lr/drift_pipeline.py
```

## Open HTML Reports

Open these files in a browser:

```text
reports/data_drift.html
reports/prediction_drift.html
```

## Generate Fresh Prediction Logs

Start API:

```bash
docker compose up -d --build api
```

Send prediction requests:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -H "x-request-id: drift-runbook-001" \
  -d '{
    "age": 35,
    "income": 75000,
    "loan_amount": 25000,
    "credit_score": 700,
    "employment_years": 5,
    "debt_to_income": 0.3
  }'
```

Check prediction logs:

```bash
tail -n 5 data/predictions.csv
```

Run drift pipeline again:

```bash
PYTHONPATH=src python src/mlops_lr/drift_pipeline.py
```

## What To Review

In the Evidently reports, review:

```text
dataset drift
column drift
feature distributions
prediction distribution
probability distribution
```

## Troubleshooting

If `data/predictions.csv` is missing:

```text
1. Start the API.
2. Send at least one /predict request.
3. Confirm prediction logging is enabled in the API.
```

If `data/reference_monitoring.csv` is missing:

```text
1. Confirm data/processed.csv exists.
2. Run the drift pipeline.
3. The pipeline recreates reference_monitoring.csv.
```

If reports are missing:

```text
1. Run PYTHONPATH=src python src/mlops_lr/drift_pipeline.py.
2. Check error output.
3. Confirm Evidently is installed.
```