#!/bin/bash
# images/tensorboard/build.sh
# 构建并推送带 TensorBoard 的 Notebook 镜像到私有仓库
#
# 用法: cd /path/to/project/root && bash images/tensorboard/build.sh

set -ex

REGISTRY="${REGISTRY:-10.121.177.20:8082}"
IMAGE_PREFIX="${IMAGE_PREFIX:-mlops/notebook}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

echo "=========================================="
echo "构建带 TensorBoard 的 Notebook 镜像"
echo "Registry: ${REGISTRY}/${IMAGE_PREFIX}"
echo "=========================================="

# Jupyter CPU
docker build --network=host \
    -t "${REGISTRY}/${IMAGE_PREFIX}:tensorboard-jupyter-cpu" \
    -f images/tensorboard/Dockerfile-jupyter-cpu \
    .
docker push "${REGISTRY}/${IMAGE_PREFIX}:tensorboard-jupyter-cpu"
echo "✅ tensorboard-jupyter-cpu"

# Jupyter Bigdata
docker build --network=host \
    -t "${REGISTRY}/${IMAGE_PREFIX}:tensorboard-jupyter-bigdata" \
    -f images/tensorboard/Dockerfile-jupyter-bigdata \
    .
docker push "${REGISTRY}/${IMAGE_PREFIX}:tensorboard-jupyter-bigdata"
echo "✅ tensorboard-jupyter-bigdata"

# VS Code CPU
docker build --network=host \
    -t "${REGISTRY}/${IMAGE_PREFIX}:tensorboard-vscode-cpu" \
    -f images/tensorboard/Dockerfile-vscode-cpu \
    .
docker push "${REGISTRY}/${IMAGE_PREFIX}:tensorboard-vscode-cpu"
echo "✅ tensorboard-vscode-cpu"

# VS Code GPU
docker build --network=host \
    -t "${REGISTRY}/${IMAGE_PREFIX}:tensorboard-vscode-gpu" \
    -f images/tensorboard/Dockerfile-vscode-gpu \
    .
docker push "${REGISTRY}/${IMAGE_PREFIX}:tensorboard-vscode-gpu"
echo "✅ tensorboard-vscode-gpu"

echo ""
echo "=========================================="
echo "构建推送完成！"
echo "=========================================="
echo ""
echo "在 config.py NOTEBOOK_IMAGES 中新增以下条目:"
echo "  ['${REGISTRY}/${IMAGE_PREFIX}:tensorboard-jupyter-cpu', 'jupyter-tensorboard（cpu）'],"
echo "  ['${REGISTRY}/${IMAGE_PREFIX}:tensorboard-jupyter-bigdata', 'jupyter-tensorboard（bigdata）'],"
echo "  ['${REGISTRY}/${IMAGE_PREFIX}:tensorboard-vscode-cpu', 'vscode-tensorboard（cpu）'],"
echo "  ['${REGISTRY}/${IMAGE_PREFIX}:tensorboard-vscode-gpu', 'vscode-tensorboard（gpu）'],"
