#!/usr/bin/env bash

set -euo pipefail

docker build -t mlops-logistic-regression-api:latest .

kubectl apply -f k8s/namespace.yaml

kubectl apply -f k8s/api-configmap.yaml
kubectl apply -f k8s/api-pvc.yaml
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/api-service.yaml
kubectl apply -f k8s/api-ingress.yaml

kubectl apply -f k8s/prometheus-configmap.yaml
kubectl apply -f k8s/prometheus-deployment.yaml
kubectl apply -f k8s/prometheus-service.yaml

kubectl apply -f k8s/grafana-datasource-configmap.yaml
kubectl apply -f k8s/grafana-dashboard-configmap.yaml
kubectl apply -f k8s/grafana-deployment.yaml
kubectl apply -f k8s/grafana-service.yaml

kubectl apply -f k8s/jaeger-deployment.yaml
kubectl apply -f k8s/jaeger-service.yaml

kubectl apply -f k8s/otel-collector-configmap.yaml
kubectl apply -f k8s/otel-collector-deployment.yaml
kubectl apply -f k8s/otel-collector-service.yaml

kubectl apply -f k8s/loki-deployment.yaml
kubectl apply -f k8s/loki-service.yaml

kubectl apply -f k8s/promtail-rbac.yaml
kubectl apply -f k8s/promtail-configmap.yaml
kubectl apply -f k8s/promtail-daemonset.yaml

kubectl rollout status deployment/mlops-api -n mlops-local
kubectl rollout status deployment/prometheus -n mlops-local
kubectl rollout status deployment/grafana -n mlops-local
kubectl rollout status deployment/jaeger -n mlops-local
kubectl rollout status deployment/otel-collector -n mlops-local
kubectl rollout status deployment/loki -n mlops-local
kubectl rollout status daemonset/promtail -n mlops-local

kubectl get pods -n mlops-local
kubectl get services -n mlops-local