#!/bin/bash
# job-template/job/model-evaluate/build.sh
# 构建并推送业务镜像（基于 model-evaluate:base-py310-cu128，仅含入口脚本层，日常迭代走此脚本）
#
# Tag 规范: Runtime 类 - <framework-version>-py<python>-cu<cuda>-r<revision>
# 说明: 重依赖在 BASE 镜像；本脚本只重建 /app 业务层，通常几十秒内完成。
#
# 用法: cd /path/to/project/root && bash job-template/job/model-evaluate/build.sh
# 首次或升级依赖: 先 bash job-template/job/model-evaluate/build-base.sh

set -euo pipefail

REGISTRY="10.121.177.20:8082"
PROJECT="mlops"
IMAGE_NAME="model-evaluate"
BASE_IMAGE_NAME="model-evaluate"
BASE_IMAGE_VERSION="base-py310-cu128"
IMAGE_VERSION="main-py310-cu128-r8"

BASE_IMAGE="${REGISTRY}/${PROJECT}/${BASE_IMAGE_NAME}:${BASE_IMAGE_VERSION}"
IMAGE="${REGISTRY}/${PROJECT}/${IMAGE_NAME}:${IMAGE_VERSION}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

DOCKERFILE_PATH="job-template/job/${IMAGE_NAME}/Dockerfile"

echo "=========================================="
echo "构建业务镜像: ${IMAGE}"
echo "BASE 镜像:    ${BASE_IMAGE}"
echo "Context:      ${PROJECT_ROOT}"
echo "Dockerfile:   ${DOCKERFILE_PATH}"
echo "=========================================="

if ! docker image inspect "${BASE_IMAGE}" >/dev/null 2>&1; then
    echo "[WARN] 本地未找到 BASE 镜像 ${BASE_IMAGE}"
    echo "[HINT] 尝试 docker pull；若不存在请先执行 build-base.sh"
    docker pull "${BASE_IMAGE}" || {
        echo "[ERROR] 无法拉取 BASE 镜像，请先运行: bash job-template/job/model-evaluate/build-base.sh"
        exit 1
    }
fi

docker build --network=host \
    --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
    -t "${IMAGE}" \
    -f "${DOCKERFILE_PATH}" \
    .

echo ""
echo "=== 冒烟测试 ==="

docker run --rm --entrypoint python3 "${IMAGE}" -c "
import opencompass, transformers, torch
assert transformers.__version__.split('.')[0] == '5'
assert torch.__version__.split('.')[0] == '2'
print('opencompass', opencompass.__version__,
      '| transformers', transformers.__version__,
      '| torch', torch.__version__)
" || { echo "[FAIL] 版本断言失败"; exit 1; }

docker run --rm --entrypoint python3 "${IMAGE}" -m py_compile \
    /app/run_evaluation.py /app/parse_and_save.py /app/sitecustomize.py \
    || { echo "[FAIL] 入口脚本语法错误"; exit 1; }

docker run --rm --entrypoint python3 "${IMAGE}" -c "
import sitecustomize  # noqa: F401 — 由 site-packages 自动加载
import site, pathlib
sp = pathlib.Path(site.getsitepackages()[0]) / 'sitecustomize.py'
assert sp.exists(), f'sitecustomize 未安装: {sp}'
print('[OK] sitecustomize 已安装到', sp)
" || { echo "[FAIL] sitecustomize 安装检查失败"; exit 1; }

echo "[OK] 冒烟测试通过"

echo ""
echo "=== 推送镜像 ==="
docker push "${IMAGE}"

echo ""
echo "=========================================="
echo "构建推送完成！"
echo "业务镜像: ${IMAGE}"
echo "BASE 镜像: ${BASE_IMAGE} (未变更则无需重建)"
echo "=========================================="
