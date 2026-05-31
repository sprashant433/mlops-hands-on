from mlops_lr.config import load_config


def test_load_config():
    config = load_config()

    assert config.project.name == "mlops-logistic-regression"
    assert config.data.target_column == "loan_approved"
    assert config.model.name == "LogisticRegression"
