#!/usr/bin/env bash
set -e

IMAGE=${1:-10.121.177.20:8082/mlops/lightgbm:20260716-gpu-swanlab-cloud-v1}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONTEXT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"   # job-template/

echo "Building ${IMAGE} from context ${CONTEXT_DIR}"
docker build --network=host -t "${IMAGE}" -f "${SCRIPT_DIR}/Dockerfile" "${CONTEXT_DIR}"

echo "Pushing ${IMAGE}..."
docker push "${IMAGE}"
echo "Done: ${IMAGE}"
