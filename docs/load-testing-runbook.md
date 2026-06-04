# Load Testing Runbook

## Purpose

This runbook explains how to run and evaluate local Locust load tests for the FastAPI inference service.

## Start Stack

```bash
docker compose up -d
docker compose ps
```

## Health Check

```bash
curl http://127.0.0.1:8000/health
```

## Run UI Load Test

```bash
locust -f locustfile.py --host http://127.0.0.1:8000
```

Open:

```text
http://127.0.0.1:8089
```

Start with:

```text
Number of users: 10
Spawn rate: 2
Host: http://127.0.0.1:8000
```

## Run Headless Load Test

```bash
locust -f locustfile.py \
  --host http://127.0.0.1:8000 \
  --users 10 \
  --spawn-rate 2 \
  --run-time 1m \
  --headless \
  --csv reports/load_tests/locust_10_users
```

## Summarize Results

```bash
python scripts/summarize_load_test.py \
  --stats-path reports/load_tests/locust_10_users_stats.csv \
  --output-path reports/load_tests/locust_10_users_summary.json \
  --max-failure-count 0 \
  --max-average-response-time-ms 500
```

## Review Summary

```bash
cat reports/load_tests/locust_10_users_summary.json
```

## Grafana Checks

Open:

```text
http://127.0.0.1:3000
```

Check dashboard:

```text
Dashboards → MLOps API Monitoring
```

Look for:

```text
Prediction Requests
Prediction Errors
Request Latency
Prediction Probability
API Prediction Logs
```

## Loki Checks

Open:

```text
Grafana → Explore → Loki
```

Query:

```logql
{job="docker"} |= "locust-load-test"
```

## Jaeger Checks

Open:

```text
http://127.0.0.1:16686
```

Search service:

```text
mlops-logistic-regression-api
```

Look for operation:

```text
model_prediction
```

## Pass Criteria

A local load test passes when:

```text
failure_count <= 0
average_response_time_ms <= 500
API remains healthy
Grafana metrics update
Loki logs appear
Jaeger traces appear
```

## Troubleshooting

If Locust cannot connect:

```text
1. Confirm docker compose ps shows api running.
2. Confirm http://127.0.0.1:8000/health works.
3. Confirm Locust host is http://127.0.0.1:8000.
```

If Loki logs are missing:

```text
1. Check docker logs mlops-promtail --tail 50.
2. Check docker logs mlops-loki --tail 50.
3. Use query {job="docker"}.
4. Confirm Docker logs contain locust-load-test.
```

If Grafana dashboard is empty:

```text
1. Set time range to Last 15 minutes.
2. Generate fresh traffic.
3. Restart Grafana after dashboard JSON changes.
```

If Jaeger traces are missing:

```text
1. Check docker logs mlops-otel-collector --tail 50.
2. Check docker logs mlops-jaeger --tail 50.
3. Generate a fresh /predict request.
```