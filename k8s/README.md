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