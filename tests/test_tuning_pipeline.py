from sklearn.linear_model import LogisticRegression

from mlops_lr.tuning_pipeline import run_tuning_pipeline


def test_run_tuning_pipeline():
    model, metrics, best_params = run_tuning_pipeline()

    assert isinstance(model, LogisticRegression)
    assert "f1" in metrics
    assert "best_solver" in best_params
