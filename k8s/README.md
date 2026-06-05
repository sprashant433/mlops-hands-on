# Kubernetes Manifests

This folder contains local Kubernetes manifests for the MLOps platform.

Namespace:

```text
mlops-local
```

Apply namespace:

```bash
kubectl apply -f k8s/namespace.yaml
```

Check namespace:

```bash
kubectl get namespace mlops-local
```

## API Deployment

Apply API deployment:

```bash
kubectl apply -f k8s/api-deployment.yaml
```

Check pods:

```bash
kubectl get pods -n mlops-local
```

Check logs:

```bash
kubectl logs -n mlops-local deployment/mlops-api
```

## API Service

Apply API service:

```bash
kubectl apply -f k8s/api-service.yaml
```

Check services:

```bash
kubectl get service -n mlops-local
```

Port forward:

```bash
kubectl port-forward -n mlops-local service/mlops-api-service 8000:8000
```

Test:

```bash
curl http://127.0.0.1:8000/health
```

## API ConfigMap

Apply API config:

```bash
kubectl apply -f k8s/api-configmap.yaml
```

Apply updated deployment:

```bash
kubectl apply -f k8s/api-deployment.yaml
```

Restart deployment:

```bash
kubectl rollout restart deployment/mlops-api -n mlops-local
kubectl rollout status deployment/mlops-api -n mlops-local
```

Verify mounted config:

```bash
kubectl exec -n mlops-local deployment/mlops-api -- cat /app/configs/config.yaml
```

## API Persistent Volumes

Apply API PVCs:

```bash
kubectl apply -f k8s/api-pvc.yaml
```

Apply updated deployment:

```bash
kubectl apply -f k8s/api-deployment.yaml
```

Restart deployment:

```bash
kubectl rollout restart deployment/mlops-api -n mlops-local
kubectl rollout status deployment/mlops-api -n mlops-local
```

Check PVCs:

```bash
kubectl get pvc -n mlops-local
```

Verify mounts:

```bash
kubectl exec -n mlops-local deployment/mlops-api -- ls /app/data
kubectl exec -n mlops-local deployment/mlops-api -- ls /app/reports
kubectl exec -n mlops-local deployment/mlops-api -- ls /app/mlruns
```

## API Health Probes

Apply updated deployment:

```bash
kubectl apply -f k8s/api-deployment.yaml
```

Restart deployment:

```bash
kubectl rollout restart deployment/mlops-api -n mlops-local
kubectl rollout status deployment/mlops-api -n mlops-local
```

Describe deployment:

```bash
kubectl describe deployment mlops-api -n mlops-local
```

## API Ingress

Apply API ingress:

```bash
kubectl apply -f k8s/api-ingress.yaml
```

Check ingress:

```bash
kubectl get ingress -n mlops-local
```

Add local host mapping:

```text
127.0.0.1 mlops-api.local
```

Test:

```bash
curl http://mlops-api.local/health
```

If no ingress controller is installed, use port-forward:

```bash
kubectl port-forward -n mlops-local service/mlops-api-service 8000:8000
curl http://127.0.0.1:8000/health
```

## Prometheus

Apply Prometheus manifests:

```bash
kubectl apply -f k8s/prometheus-configmap.yaml
kubectl apply -f k8s/prometheus-deployment.yaml
kubectl apply -f k8s/prometheus-service.yaml
```

Check rollout:

```bash
kubectl rollout status deployment/prometheus -n mlops-local
kubectl get pods -n mlops-local
```

Port forward:

```bash
kubectl port-forward -n mlops-local service/prometheus-service 9090:9090
```

Open:

```text
http://127.0.0.1:9090
```

Query:

```promql
up
```

## Grafana

Apply Grafana manifests:

```bash
kubectl apply -f k8s/grafana-datasource-configmap.yaml
kubectl apply -f k8s/grafana-deployment.yaml
kubectl apply -f k8s/grafana-service.yaml
```

Check rollout:

```bash
kubectl rollout status deployment/grafana -n mlops-local
kubectl get pods -n mlops-local
```

Port forward:

```bash
kubectl port-forward -n mlops-local service/grafana-service 3000:3000
```

Open:

```text
http://127.0.0.1:3000
```

Login:

```text
username: admin
password: admin
```

## Grafana Dashboard Provisioning

Apply dashboard ConfigMap:

```bash
kubectl apply -f k8s/grafana-dashboard-configmap.yaml
```

Apply updated Grafana deployment:

```bash
kubectl apply -f k8s/grafana-deployment.yaml
```

Restart Grafana:

```bash
kubectl rollout restart deployment/grafana -n mlops-local
kubectl rollout status deployment/grafana -n mlops-local
```

Open Grafana:

```bash
kubectl port-forward -n mlops-local service/grafana-service 3000:3000
```

Go to:

```text
Dashboards → MLOps → MLOps API Monitoring - Kubernetes
```

## Jaeger

Apply Jaeger manifests:

```bash
kubectl apply -f k8s/jaeger-deployment.yaml
kubectl apply -f k8s/jaeger-service.yaml
```

Check rollout:

```bash
kubectl rollout status deployment/jaeger -n mlops-local
kubectl get pods -n mlops-local
```

Port forward:

```bash
kubectl port-forward -n mlops-local service/jaeger-service 16686:16686
```

Open:

```text
http://127.0.0.1:16686
```

## OpenTelemetry Collector

Apply collector manifests:

```bash
kubectl apply -f k8s/otel-collector-configmap.yaml
kubectl apply -f k8s/otel-collector-deployment.yaml
kubectl apply -f k8s/otel-collector-service.yaml
```

Check rollout:

```bash
kubectl rollout status deployment/otel-collector -n mlops-local
kubectl logs -n mlops-local deployment/otel-collector
```

## API Tracing to OTel Collector

Apply updated API config:

```bash
kubectl apply -f k8s/api-configmap.yaml
```

Restart API:

```bash
kubectl rollout restart deployment/mlops-api -n mlops-local
kubectl rollout status deployment/mlops-api -n mlops-local
```

Generate trace:

```bash
kubectl port-forward -n mlops-local service/mlops-api-service 8000:8000
curl http://127.0.0.1:8000/health
```

Open Jaeger:

```bash
kubectl port-forward -n mlops-local service/jaeger-service 16686:16686
```

Check service:

```text
mlops-logistic-regression-api
```

## Loki

Apply Loki manifests:

```bash
kubectl apply -f k8s/loki-deployment.yaml
kubectl apply -f k8s/loki-service.yaml
```

Check rollout:

```bash
kubectl rollout status deployment/loki -n mlops-local
kubectl get pods -n mlops-local
```

Port forward:

```bash
kubectl port-forward -n mlops-local service/loki-service 3100:3100
```

Check readiness:

```bash
curl http://127.0.0.1:3100/ready
```

## Promtail

Apply Promtail manifests:

```bash
kubectl apply -f k8s/promtail-configmap.yaml
kubectl apply -f k8s/promtail-daemonset.yaml
```

Check DaemonSet:

```bash
kubectl get daemonset -n mlops-local
kubectl get pods -n mlops-local -l app=promtail
```

Check logs:

```bash
kubectl logs -n mlops-local -l app=promtail --tail 50
```

## Grafana Loki Datasource

Apply updated datasource ConfigMap:

```bash
kubectl apply -f k8s/grafana-datasource-configmap.yaml
```

Restart Grafana:

```bash
kubectl rollout restart deployment/grafana -n mlops-local
kubectl rollout status deployment/grafana -n mlops-local
```

Open Grafana:

```bash
kubectl port-forward -n mlops-local service/grafana-service 3000:3000
```

Check:

```text
Connections → Data sources → Loki
```

Loki query:

```logql
{namespace="mlops-local"}
```

## Final Kubernetes Logging Check

Apply Promtail updates:

```bash
kubectl apply -f k8s/promtail-configmap.yaml
kubectl apply -f k8s/promtail-daemonset.yaml
kubectl rollout restart daemonset/promtail -n mlops-local
kubectl rollout status daemonset/promtail -n mlops-local
```

Check Promtail logs:

```bash
kubectl logs -n mlops-local -l app=promtail --tail 100
```

Generate API logs:

```bash
kubectl port-forward -n mlops-local service/mlops-api-service 8000:8000
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/model-info
```

Check Loki:

```bash
kubectl port-forward -n mlops-local service/loki-service 3100:3100
curl http://127.0.0.1:3100/loki/api/v1/labels
```

Grafana Explore queries:

```logql
{namespace="mlops-local"}
```

```logql
{job="kubernetes-containers"}
```

## Kubernetes API Smoke Test

Port forward API service:

```bash
kubectl port-forward -n mlops-local service/mlops-api-service 8000:8000
```

Run smoke test:

```bash
./scripts/smoke_test_k8s_api.sh
```

## Full Kubernetes Deployment

Run full deployment:

```bash
./scripts/deploy_k8s.sh
```

Check resources:

```bash
kubectl get all -n mlops-local
```

## Kubernetes Teardown

Delete full stack:

```bash
./scripts/delete_k8s.sh
```

Redeploy full stack:

```bash
./scripts/deploy_k8s.sh
```