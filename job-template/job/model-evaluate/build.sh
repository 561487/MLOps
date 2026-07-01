#!/bin/bash
# job-template/job/model-evaluate/build.sh
# 构建并推送 Docker 镜像到私有仓库
#
# 用法: cd /path/to/project/root && bash job-template/job/model-evaluate/build.sh

set -ex

# ── 仓库配置（可通过环境变量覆盖） ──
REGISTRY="${REGISTRY:-10.121.177.20:8082}"
IMAGE_NAME="${IMAGE_NAME:-mlops/model-evaluate}"
IMAGE_TAG=$(date +%Y%m%d-%H%M%S)
# IMAGE_NAME 可能带 registry 前缀（如 mlops/model-evaluate），
# Dockerfile 路径只用最后一段作为目录名
JOB_NAME="${IMAGE_NAME##*/}"
DOCKERFILE_PATH="job-template/job/${JOB_NAME}/Dockerfile"

# ── 切换到项目根目录 ──
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

echo "=========================================="
echo "构建镜像: ${REGISTRY}/${IMAGE_NAME}"
echo "Tag:      ${IMAGE_TAG}"
echo "Context:  ${PROJECT_ROOT}"
echo "Dockerfile: ${DOCKERFILE_PATH}"
echo "=========================================="

# ── 构建 ──
docker build \
    --network=host \
    -t "${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}" \
    -f "${DOCKERFILE_PATH}" \
    .

# ── 打 latest 标签 ──
docker tag "${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}" "${REGISTRY}/${IMAGE_NAME}:latest"

# ── 推送 ──
docker push "${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
docker push "${REGISTRY}/${IMAGE_NAME}:latest"

echo "=========================================="
echo "构建推送完成！"
echo "镜像地址: ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
echo "Latest:   ${REGISTRY}/${IMAGE_NAME}:latest"
echo "=========================================="
