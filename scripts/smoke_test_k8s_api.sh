#!/usr/bin/env bash

set -euo pipefail

API_URL="${API_URL:-http://127.0.0.1:8000}"

curl -s "$API_URL/health"

curl -s -X POST "$API_URL/predict" \
  -H "Content-Type: application/json" \
  -H "x-request-id: k8s-smoke-test" \
  -d '{
    "age": 35,
    "income": 75000,
    "loan_amount": 25000,
    "credit_score": 700,
    "employment_years": 5,
    "debt_to_income": 0.3
  }'