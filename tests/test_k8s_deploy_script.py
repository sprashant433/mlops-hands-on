from pathlib import Path


def test_k8s_deploy_script_exists():
    script = Path("scripts/deploy_k8s.sh")

    assert script.exists()


def test_k8s_deploy_script_applies_core_manifests():
    content = Path("scripts/deploy_k8s.sh").read_text()

    assert "docker build -t mlops-logistic-regression-api:latest ." in content
    assert "kubectl apply -f k8s/namespace.yaml" in content
    assert "kubectl apply -f k8s/api-deployment.yaml" in content
    assert "kubectl apply -f k8s/prometheus-deployment.yaml" in content
    assert "kubectl apply -f k8s/grafana-deployment.yaml" in content
    assert "kubectl apply -f k8s/promtail-daemonset.yaml" in content
