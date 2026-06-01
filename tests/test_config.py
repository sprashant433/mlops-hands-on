from mlops_lr.config import load_config


def test_load_config():
    config = load_config()

    assert config.project.name == "mlops-logistic-regression"
    assert config.data.target_column == "loan_approved"
    assert config.model.name == "LogisticRegression"

    assert config.mlflow.tracking_uri == "file:./mlruns"
    assert config.mlflow.experiment_name == "loan-approval-logistic-regression"
    assert config.tuning.max_evals == 10
    assert config.mlflow.registered_model_name == "LoanApprovalModel"
