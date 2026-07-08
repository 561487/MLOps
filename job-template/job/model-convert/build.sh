#!/bin/bash
set -ex
REGISTRY="${REGISTRY:-10.121.177.20:8082}"
IMAGE_NAME="${IMAGE_NAME:-mlops/model-convert}"
IMAGE_TAG=$(date +%Y%m%d)
JOB_NAME="${IMAGE_NAME##*/}"
DOCKERFILE_PATH="job-template/job/${JOB_NAME}/Dockerfile"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"
echo "Building: ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
docker build --network=host \
    -t "${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}" \
    -f "${DOCKERFILE_PATH}" .
docker tag "${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}" "${REGISTRY}/${IMAGE_NAME}:latest"
docker push "${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
docker push "${REGISTRY}/${IMAGE_NAME}:latest"
echo "Done: ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
