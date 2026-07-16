#!/usr/bin/env bash
set -e

IMAGE=${1:-10.121.177.20:8082/mlops/hyperparam-search:20260710-swanlab-gpu-v1}

echo "Building image: ${IMAGE}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONTEXT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"   # job-template/

echo "Build context: ${CONTEXT_DIR}"

docker build --network=host --no-cache -t "${IMAGE}" -f "${SCRIPT_DIR}/Dockerfile" "${CONTEXT_DIR}"

echo "Build success: ${IMAGE}"
echo "Pushing ..."
docker push "${IMAGE}"
echo "Push success: ${IMAGE}"
