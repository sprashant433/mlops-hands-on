import pytest

from scripts.summarize_load_test import validate_thresholds


def test_validate_thresholds_passes():
    summary = {
        "failure_count": 0,
        "average_response_time_ms": 100,
    }

    validate_thresholds(
        summary,
        max_failure_count=0,
        max_average_response_time_ms=500,
    )


def test_validate_thresholds_fails_for_failures():
    summary = {
        "failure_count": 1,
        "average_response_time_ms": 100,
    }

    with pytest.raises(ValueError, match="failure_count"):
        validate_thresholds(
            summary,
            max_failure_count=0,
            max_average_response_time_ms=500,
        )


def test_validate_thresholds_fails_for_slow_response():
    summary = {
        "failure_count": 0,
        "average_response_time_ms": 800,
    }

    with pytest.raises(ValueError, match="average_response_time_ms"):
        validate_thresholds(
            summary,
            max_failure_count=0,
            max_average_response_time_ms=500,
        )
