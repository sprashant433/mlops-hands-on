import logging

from mlops_lr.logger import get_logger


def test_get_logger():
    logger = get_logger("test_logger")

    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_logger"
    assert logger.level == logging.INFO
