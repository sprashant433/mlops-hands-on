# Production Deployment Runbook

This runbook explains how to execute the full production-level MLOps flow locally.

## 1. Start From develop

```bash
git checkout develop
git pull origin develop
```

## 2. Run Local Quality Checks

```bash
black src tests
flake8 src tests
PYTHONPATH=src pytest
```

## 3. Run Production Flow Locally

Run without Docker build:

```bash
PYTHONPATH=src python src/mlops_lr/production_flow.py --skip-docker
```

Run with Docker build:

```bash
PYTHONPATH=src python src/mlops_lr/production_flow.py --image-tag local
```

Or through the shell wrapper:

```bash
IMAGE_TAG=local ./scripts/run_production_flow.sh
```

## 4. Start Local Platform

```bash
docker compose up --build
```

Verify services:

```bash
docker compose ps
```

Open services:

```text
API:        http://127.0.0.1:8000
MLflow:     http://127.0.0.1:5000
Prometheus: http://127.0.0.1:9090
Grafana:    http://127.0.0.1:3000
Jaeger:     http://127.0.0.1:16686
Loki:       http://127.0.0.1:3100
```

## 5. Run API Smoke Test

```bash
./scripts/smoke_test_api.sh
```

## 6. Check MLflow

Confirm:

- Experiment exists
- Latest run exists
- Metrics are logged
- Artifacts are logged
- Best model is registered
- Production model version exists

## 7. Check Monitoring

Prometheus:

```text
http://127.0.0.1:9090
```

Useful metrics:

```text
http_requests_total
http_request_duration_seconds
model_drift_detected
```

Grafana:

```text
http://127.0.0.1:3000
```

Confirm dashboards show:

- Request count
- Latency
- Errors
- Drift alert
- Logs

## 8. Check Tracing

Jaeger:

```text
http://127.0.0.1:16686
```

Confirm service appears:

```text
mlops-logistic-regression-api
```

## 9. Check Logs

Grafana Explore:

```text
Datasource: Loki
```

Example LogQL:

```text
{container_name="mlops-logistic-regression-api"}
```

## 10. Run Drift Pipeline

```bash
PYTHONPATH=src python src/mlops_lr/drift_pipeline.py
```

Expected outputs:

```text
reports/input_statistics.json
reports/prediction_statistics.json
reports/data_drift.html
reports/data_drift.json
reports/prediction_drift.html
reports/prediction_drift.json
reports/drift_alert.json
reports/retraining_trigger.json
```

## 11. Run Kubernetes Deployment

```bash
./scripts/deploy_k8s.sh
```

Check resources:

```bash
kubectl get all -n mlops-local
```

Run Kubernetes smoke test:

```bash
kubectl port-forward -n mlops-local service/mlops-api-service 8000:8000
./scripts/smoke_test_k8s_api.sh
```

## 12. Merge and Tag Release

```bash
git checkout main
git merge --no-ff develop -m "merge: develop into main"
git tag v1.5-production-mlops
git checkout develop
```

Push:

```bash
git push origin develop
git push origin main
git push origin v1.5-production-mlops
```

## 13. Rollback

If the release fails:

```bash
git checkout main
git log --oneline
```

Find the previous known-good tag:

```bash
git tag
```

Return to the previous tag locally:

```bash
git checkout <previous-tag>
```

For MLflow model rollback, use the rollback utility:

```bash
PYTHONPATH=src python -c "from mlops_lr.mlflow_utils import rollback_model_to_previous_production; rollback_model_to_previous_production('LoanApprovalModel')"
```

## 14. Cleanup

Docker:

```bash
docker compose down
```

Kubernetes:

```bash
./scripts/delete_k8s.sh
```