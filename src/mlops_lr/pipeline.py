from mlops_lr.data import generate_raw_data
from mlops_lr.evaluate import evaluate_model
from mlops_lr.features import create_processed_data
from mlops_lr.logger import get_logger
from mlops_lr.train import train_model


logger = get_logger(__name__)


def run_pipeline() -> dict[str, float]:
    logger.info("Starting ML pipeline")

    generate_raw_data()
    create_processed_data()
    _, run_id = train_model()
    metrics = evaluate_model(run_id=run_id)

    logger.info("ML pipeline completed")
    return metrics


if __name__ == "__main__":
    run_pipeline()
