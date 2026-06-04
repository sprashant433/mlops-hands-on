import argparse
import json
from pathlib import Path

import pandas as pd


def summarize_load_test(stats_path: str) -> dict[str, float]:
    stats = pd.read_csv(stats_path)

    aggregated = stats[stats["Name"] == "Aggregated"]

    if aggregated.empty:
        raise ValueError("Aggregated row not found in Locust stats file")

    row = aggregated.iloc[0]

    return {
        "request_count": float(row["Request Count"]),
        "failure_count": float(row["Failure Count"]),
        "median_response_time_ms": float(row["Median Response Time"]),
        "average_response_time_ms": float(row["Average Response Time"]),
        "min_response_time_ms": float(row["Min Response Time"]),
        "max_response_time_ms": float(row["Max Response Time"]),
        "requests_per_second": float(row["Requests/s"]),
        "failures_per_second": float(row["Failures/s"]),
    }


def save_summary(summary: dict[str, float], output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2))


def validate_thresholds(
    summary: dict[str, float],
    max_failure_count: float,
    max_average_response_time_ms: float,
) -> None:
    if summary["failure_count"] > max_failure_count:
        raise ValueError(
            "Load test failed: "
            f"failure_count={summary['failure_count']} "
            f"> max_failure_count={max_failure_count}"
        )

    if summary["average_response_time_ms"] > max_average_response_time_ms:
        raise ValueError(
            "Load test failed: "
            f"average_response_time_ms={summary['average_response_time_ms']} "
            f"> max_average_response_time_ms={max_average_response_time_ms}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--max-failure-count", type=float, default=0)
    parser.add_argument("--max-average-response-time-ms", type=float, default=500)
    args = parser.parse_args()

    summary = summarize_load_test(args.stats_path)
    validate_thresholds(
        summary,
        max_failure_count=args.max_failure_count,
        max_average_response_time_ms=args.max_average_response_time_ms,
    )
    save_summary(summary, args.output_path)


if __name__ == "__main__":
    main()
