#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${1:-mlops-logistic-regression-api:latest}"
CONTAINER_NAME="mlops-api-smoke-test"

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

docker run -d \
  --name "${CONTAINER_NAME}" \
  -p 8000:8000 \
  "${IMAGE_NAME}"

cleanup() {
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
}

trap cleanup EXIT

sleep 10

curl --fail http://127.0.0.1:8000/health