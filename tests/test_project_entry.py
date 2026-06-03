from mlops_lr.project_entry import main


def test_project_entry_imports():
    assert callable(main)
