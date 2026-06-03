import json
import logging

from mlops_lr.json_logging import JsonFormatter


def test_json_formatter_outputs_valid_json():
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="mlops_lr.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="prediction_completed",
        args=(),
        exc_info=None,
    )
    record.trace_id = "abc123"
    record.loan_approved = 1

    output = formatter.format(record)
    payload = json.loads(output)

    assert payload["level"] == "INFO"
    assert payload["logger"] == "mlops_lr.test"
    assert payload["message"] == "prediction_completed"
    assert payload["trace_id"] == "abc123"
    assert payload["loan_approved"] == 1
