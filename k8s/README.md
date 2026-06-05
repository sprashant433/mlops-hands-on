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