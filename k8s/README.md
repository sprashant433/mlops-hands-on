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