from pathlib import Path


def test_locustfile_exists():
    locustfile = Path("locustfile.py")

    assert locustfile.exists()


def test_locustfile_defines_loan_approval_user():
    content = Path("locustfile.py").read_text()

    assert "class LoanApprovalUser" in content
    assert "def predict" in content
    assert "def health" in content
    assert "locust-load-test" in content
