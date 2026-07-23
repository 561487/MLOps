#!/usr/bin/env bash
set -e

IMAGE_NAME=${1:-10.121.177.20:8082/mlops/decision-tree:20260717}

echo "Building image: ${IMAGE_NAME}"

cd "$(dirname "$0")"
docker build -t "${IMAGE_NAME}" .

echo "Build success: ${IMAGE_NAME}"
