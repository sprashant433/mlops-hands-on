from mlops_lr.config import load_config
from mlops_lr.data import generate_raw_data
from mlops_lr.features import build_features
from mlops_lr.mlflow_utils import (
    get_latest_model_version,
    get_mlflow_client,
    promote_model_version_to_stage,
)
from mlops_lr.tune import tune_model


def test_promote_specific_model_version_to_production():
    config = load_config()

    raw_data = generate_raw_data(n_samples=300)
    processed_data = build_features(raw_data)
    processed_data.to_csv(config.data.processed_path, index=False)

    tune_model()
    version = get_latest_model_version(config.mlflow.registered_model_name)

    promoted_version = promote_model_version_to_stage(
        model_name=config.mlflow.registered_model_name,
        version=version.version,
        stage="Production",
    )

    assert promoted_version.version == version.version
    assert promoted_version.current_stage == "Production"

    client = get_mlflow_client()
    aliased_version = client.get_model_version_by_alias(
        config.mlflow.registered_model_name,
        "production",
    )

    assert aliased_version.version == version.version
