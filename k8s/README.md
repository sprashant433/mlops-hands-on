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