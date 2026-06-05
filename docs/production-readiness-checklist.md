# Production Readiness Checklist

Use this checklist before promoting a model or deployment to production.

## 1. Source Control

- [ ] Feature branch was created from `develop`
- [ ] Pull request was reviewed
- [ ] CI passed
- [ ] Changes were merged into `develop`
- [ ] `develop` was merged into `main`
- [ ] Release tag was created

## 2. Data Pipeline

- [ ] Raw data generation completed
- [ ] Data validation passed
- [ ] Feature engineering completed
- [ ] Processed dataset was created
- [ ] Target column is present
- [ ] No unexpected null values were introduced

## 3. Model Training

- [ ] Training pipeline completed
- [ ] MLflow run was created
- [ ] Parameters were logged
- [ ] Metrics were logged
- [ ] Confusion matrix artifact was logged
- [ ] Model artifact was logged
- [ ] Model signature is available

## 4. Hyperparameter Tuning

- [ ] Tuning pipeline completed
- [ ] Every trial was tracked in MLflow
- [ ] Best trial was selected
- [ ] Best metrics are acceptable
- [ ] Best model was registered

## 5. Model Registry

- [ ] Registered model exists
- [ ] Correct model version is selected
- [ ] Model was promoted to Staging
- [ ] Staging validation completed
- [ ] Model was promoted to Production
- [ ] Rollback version is known

## 6. API Serving

- [ ] FastAPI app starts successfully
- [ ] `/health` returns healthy response
- [ ] `/ready` returns ready response
- [ ] `/model-info` returns model metadata
- [ ] `/predict` returns prediction response
- [ ] Request ID is included in responses

## 7. Docker

- [ ] Docker image builds successfully
- [ ] API container starts successfully
- [ ] MLflow container starts successfully
- [ ] Docker Compose stack starts successfully
- [ ] Smoke test passes against containerized API

## 8. CI/CD

- [ ] CI workflow passes
- [ ] CD workflow passes
- [ ] Production flow workflow passes
- [ ] Reports are uploaded as artifacts
- [ ] Docker image is tagged with Git SHA

## 9. Monitoring

- [ ] Prometheus is scraping API metrics
- [ ] Grafana dashboard loads
- [ ] Request count is visible
- [ ] Latency is visible
- [ ] Error rate is visible
- [ ] Drift alert metric is visible

## 10. Observability

- [ ] OpenTelemetry collector is running
- [ ] Jaeger UI is available
- [ ] API traces are visible
- [ ] Trace ID appears in logs
- [ ] Request ID appears in logs

## 11. Logging

- [ ] JSON logs are emitted
- [ ] Loki is running
- [ ] Promtail is running
- [ ] Grafana Loki datasource works
- [ ] API logs are searchable in Grafana

## 12. Load Testing

- [ ] 10-user load test completed
- [ ] 100-user load test completed
- [ ] 500-user load test completed
- [ ] 1000-user load test completed
- [ ] Latency threshold passed
- [ ] Failure threshold passed

## 13. Drift Monitoring

- [ ] Prediction logging is enabled
- [ ] Input statistics are generated
- [ ] Prediction statistics are generated
- [ ] Evidently data drift report is generated
- [ ] Evidently prediction drift report is generated
- [ ] Drift alert file is generated
- [ ] Retraining trigger file is generated

## 14. Kubernetes

- [ ] Kubernetes context is correct
- [ ] Namespace exists
- [ ] API deployment is ready
- [ ] API service is reachable
- [ ] ConfigMap is mounted correctly
- [ ] PVCs are bound
- [ ] Prometheus is running
- [ ] Grafana is running
- [ ] Jaeger is running
- [ ] Loki is running
- [ ] Promtail is running
- [ ] Kubernetes smoke test passes

## 15. Release Evidence

- [ ] Release manifest was generated
- [ ] Git commit is recorded
- [ ] Docker image tag is recorded
- [ ] MLflow model name is recorded
- [ ] Serving stage is recorded
- [ ] Reports are stored under `reports/`