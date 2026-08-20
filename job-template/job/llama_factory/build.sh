#!/bin/bash

set -ex

REGISTRY="10.121.177.20:8082"
PROJECT="mlops"
IMAGE_NAME="llama_factory"
IMAGE_VERSION="0.9.5-py313-cu128-r1"
IMAGE="${REGISTRY}/${PROJECT}/${IMAGE_NAME}:${IMAGE_VERSION}"

docker build --network=host -t "${IMAGE}" -f job/llama_factory/Dockerfile .
docker push "${IMAGE}"

echo "Published: ${IMAGE}"
