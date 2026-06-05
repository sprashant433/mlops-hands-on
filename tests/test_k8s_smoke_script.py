from pathlib import Path


def test_k8s_smoke_script_exists():
    script = Path("scripts/smoke_test_k8s_api.sh")

    assert script.exists()


def test_k8s_smoke_script_contains_health_and_predict_checks():
    content = Path("scripts/smoke_test_k8s_api.sh").read_text()

    assert "/health" in content
    assert "/predict" in content
    assert "k8s-smoke-test" in content
