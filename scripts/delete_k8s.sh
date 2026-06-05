#!/usr/bin/env bash

set -euo pipefail

kubectl delete namespace mlops-local --ignore-not-found=true