#!/bin/bash
# 构建 worker 镜像（独立脚本，也可用 build_all.sh 一键构建 launcher+worker）
set -e
cd "$(dirname "$0")"

# 镜像配置（按规范第 4 节统一变量名）
REGISTRY="10.121.177.20:8082"
PROJECT="mlops"
IMAGE_NAME="model-offline-predict"
IMAGE_VERSION="1.0.0-20260825-r1"
IMAGE="${REGISTRY}/${PROJECT}/${IMAGE_NAME}:${IMAGE_VERSION}"

docker build -t "${IMAGE}" .
docker push "${IMAGE}"

echo "Published: ${IMAGE}"
