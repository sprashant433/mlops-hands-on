from pathlib import Path


def test_k8s_delete_script_exists():
    script = Path("scripts/delete_k8s.sh")

    assert script.exists()


def test_k8s_delete_script_deletes_namespace():
    content = Path("scripts/delete_k8s.sh").read_text()

    assert "kubectl delete namespace mlops-local" in content
    assert "--ignore-not-found=true" in content
