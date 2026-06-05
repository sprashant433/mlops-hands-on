#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-local}"

PYTHONPATH=src python src/mlops_lr/production_flow.py --image-tag "$IMAGE_TAG"