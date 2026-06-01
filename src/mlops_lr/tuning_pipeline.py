from mlops_lr.data import generate_raw_data
from mlops_lr.features import create_processed_data
from mlops_lr.logger import get_logger
from mlops_lr.tune import tune_model


logger = get_logger(__name__)


def run_tuning_pipeline() -> tuple:
    logger.info("Starting tuning pipeline")

    generate_raw_data()
    create_processed_data()
    model, metrics, best_params = tune_model()

    logger.info("Tuning pipeline completed")
    return model, metrics, best_params


if __name__ == "__main__":
    run_tuning_pipeline()
