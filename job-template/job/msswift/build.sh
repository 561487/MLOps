#!/bin/bash
set -ex
IMAGE=${IMAGE:-10.121.177.20:8082/mlops/msswift:20260715-v5}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
docker build --network=host -t "${IMAGE}" -f "${SCRIPT_DIR}/Dockerfile" "${REPO_ROOT}"
docker push "${IMAGE}"
echo "Built and pushed: ${IMAGE}"
