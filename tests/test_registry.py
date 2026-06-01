from mlflow.tracking import MlflowClient

from mlops_lr.config import load_config
from mlops_lr.data import generate_raw_data
from mlops_lr.features import build_features
from mlops_lr.mlflow_utils import configure_mlflow
from mlops_lr.tune import tune_model


def test_register_best_tuned_model():
    config = load_config()
    configure_mlflow()

    raw_data = generate_raw_data(n_samples=300)
    processed_data = build_features(raw_data)
    processed_data.to_csv(config.data.processed_path, index=False)

    tune_model()

    client = MlflowClient()
    registered_model = client.get_registered_model(config.mlflow.registered_model_name)

    assert registered_model.name == "LoanApprovalModel"
