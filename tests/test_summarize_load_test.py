import json

import pandas as pd

from scripts.summarize_load_test import save_summary, summarize_load_test


def test_summarize_load_test(tmp_path):
    stats_path = tmp_path / "locust_stats.csv"

    pd.DataFrame(
        [
            {
                "Type": "POST",
                "Name": "/predict",
                "Request Count": 10,
                "Failure Count": 0,
                "Median Response Time": 20,
                "Average Response Time": 25,
                "Min Response Time": 10,
                "Max Response Time": 50,
                "Requests/s": 2.5,
                "Failures/s": 0.0,
            },
            {
                "Type": "",
                "Name": "Aggregated",
                "Request Count": 20,
                "Failure Count": 1,
                "Median Response Time": 30,
                "Average Response Time": 35,
                "Min Response Time": 10,
                "Max Response Time": 80,
                "Requests/s": 4.0,
                "Failures/s": 0.1,
            },
        ]
    ).to_csv(stats_path, index=False)

    summary = summarize_load_test(str(stats_path))

    assert summary["request_count"] == 20
    assert summary["failure_count"] == 1
    assert summary["median_response_time_ms"] == 30
    assert summary["average_response_time_ms"] == 35
    assert summary["requests_per_second"] == 4.0
    assert summary["failures_per_second"] == 0.1


def test_save_summary(tmp_path):
    output_path = tmp_path / "summary.json"
    summary = {
        "request_count": 20,
        "failure_count": 1,
    }

    save_summary(summary, str(output_path))

    saved = json.loads(output_path.read_text())

    assert saved == summary
