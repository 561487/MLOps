#!/bin/bash
# 构建上下文为 job-template/job（包含 pkgs/ 和 model_offline_predict/）
set -e
cd "$(dirname "$0")/.."

REGISTRY="10.121.177.20:8082"
PROJECT="mlops"
IMAGE_NAME="offline-predict-launcher"
IMAGE_VERSION="1.0.0-20260821-r1"
IMAGE="${REGISTRY}/${PROJECT}/${IMAGE_NAME}:${IMAGE_VERSION}"

docker build -t "${IMAGE}" -f model_offline_predict/Dockerfile .
docker push "${IMAGE}"

echo "Published: ${IMAGE}"
