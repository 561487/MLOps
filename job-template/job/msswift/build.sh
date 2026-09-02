#!/bin/bash
set -ex

REGISTRY="10.121.177.20:8082"
PROJECT="notebook"
IMAGE_NAME="msswift"
IMAGE_VERSION="4.5.0-py311-cu128-r1"
IMAGE="${REGISTRY}/${PROJECT}/${IMAGE_NAME}:${IMAGE_VERSION}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "${REPO_ROOT}"

docker build --network=host -t "${IMAGE}" \
  -f job-template/job/msswift/Dockerfile .
docker push "${IMAGE}"

echo "Published: ${IMAGE}"
