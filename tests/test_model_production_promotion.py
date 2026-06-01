from mlops_lr.config import load_config
from mlops_lr.data import generate_raw_data
from mlops_lr.features import build_features
from mlops_lr.mlflow_utils import get_mlflow_client, promote_latest_model_to_stage
from mlops_lr.tune import tune_model


def test_promote_latest_model_to_production():
    config = load_config()

    raw_data = generate_raw_data(n_samples=300)
    processed_data = build_features(raw_data)
    processed_data.to_csv(config.data.processed_path, index=False)

    tune_model()

    promoted_version = promote_latest_model_to_stage(
        config.mlflow.registered_model_name,
        "Production",
    )

    assert promoted_version.current_stage == "Production"

    client = get_mlflow_client()
    aliased_version = client.get_model_version_by_alias(
        config.mlflow.registered_model_name,
        "production",
    )

    assert aliased_version.version == promoted_version.version
