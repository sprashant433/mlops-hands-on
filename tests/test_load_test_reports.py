from pathlib import Path


def test_load_test_reports_directory_exists():
    reports_dir = Path("reports/load_tests")

    assert reports_dir.exists()
    assert reports_dir.is_dir()
