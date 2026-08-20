#!/bin/bash
set -ex

REGISTRY="10.121.177.20:8082"
PROJECT="mlops"
IMAGE_NAME="msswift-base"
IMAGE_VERSION="4.5.0-py311-cu128-r2"
IMAGE="${REGISTRY}/${PROJECT}/${IMAGE_NAME}:${IMAGE_VERSION}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

docker build --network=host -t "${IMAGE}" .
docker push "${IMAGE}"

echo "Published: ${IMAGE}"
