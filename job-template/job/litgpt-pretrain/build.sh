#!/bin/bash
set -ex

REGISTRY="${REGISTRY:-10.121.177.20:8082}"
IMAGE_NAME="${IMAGE_NAME:-mlops/litgpt-pretrain}"
IMAGE_TAG=$(date +%Y%m%d-%H%M)
JOB_NAME="litgpt-pretrain"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

cd "${PROJECT_ROOT}"

docker build --network=host \
    -t "${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}" \
    -f "job-template/job/${JOB_NAME}/Dockerfile" \
    -t "${REGISTRY}/${IMAGE_NAME}:latest" \
    .

docker push "${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
docker push "${REGISTRY}/${IMAGE_NAME}:latest"

echo "Done: ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
