# Kubernetes Runbook

## Purpose

This runbook explains how to deploy, verify, debug, and delete the local Kubernetes MLOps stack.

## Prerequisites

Use one local Kubernetes runtime:

```text
Docker Desktop Kubernetes
or
Minikube
```

Check cluster:

```bash
kubectl config current-context
kubectl get nodes
```

## Deploy Stack

```bash
./scripts/deploy_k8s.sh
```

## Check Resources

```bash
kubectl get all -n mlops-local
kubectl get pvc -n mlops-local
kubectl get ingress -n mlops-local
```

## API Access

Port forward API:

```bash
kubectl port-forward -n mlops-local service/mlops-api-service 8000:8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Prediction check:

```bash
./scripts/smoke_test_k8s_api.sh
```

## Prometheus Access

Port forward Prometheus:

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

## Grafana Access

Port forward Grafana:

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

## Jaeger Access

Port forward Jaeger:

```bash
kubectl port-forward -n mlops-local service/jaeger-service 16686:16686
```

Open:

```text
http://127.0.0.1:16686
```

Search service:

```text
mlops-logistic-regression-api
```

## Loki Access

Port forward Loki:

```bash
kubectl port-forward -n mlops-local service/loki-service 3100:3100
```

Check labels:

```bash
curl http://127.0.0.1:3100/loki/api/v1/labels
```

Grafana Loki queries:

```logql
{namespace="mlops-local"}
```

```logql
{job="kubernetes-containers"}
```

## Common Debug Commands

Check pods:

```bash
kubectl get pods -n mlops-local
```

Describe failing pod:

```bash
kubectl describe pod <pod-name> -n mlops-local
```

Check deployment logs:

```bash
kubectl logs -n mlops-local deployment/mlops-api --tail 100
```

Check previous crashed logs:

```bash
kubectl logs -n mlops-local deployment/mlops-api --previous
```

Check events:

```bash
kubectl get events -n mlops-local --sort-by=.lastTimestamp
```

## Restart Components

Restart API:

```bash
kubectl rollout restart deployment/mlops-api -n mlops-local
kubectl rollout status deployment/mlops-api -n mlops-local
```

Restart Grafana:

```bash
kubectl rollout restart deployment/grafana -n mlops-local
kubectl rollout status deployment/grafana -n mlops-local
```

Restart Promtail:

```bash
kubectl rollout restart daemonset/promtail -n mlops-local
kubectl rollout status daemonset/promtail -n mlops-local
```

## Delete Stack

```bash
./scripts/delete_k8s.sh
```

## Redeploy Stack

```bash
./scripts/deploy_k8s.sh
```