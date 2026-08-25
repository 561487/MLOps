#!/bin/bash
# job-template/job/model-evaluate/build.sh
# 构建并推送 Docker 镜像到 Harbor（遵循平台镜像构建发布规范）
#
# Tag 规范: Runtime 类 - <framework-version>-py<python>-cu<cuda>-r<revision>
#           framework=main(git 主分支), python=3.10(ubuntu22.04), cuda=12.4
# 说明: 本镜像为 Qwen3.5 支持版（OpenCompass main + transformers>=5.2 + torch 2.6），
#       详细方案见 PLAN-qwen35-upgrade.md
#
# 用法: cd /path/to/project/root && bash job-template/job/model-evaluate/build.sh

set -euo pipefail

# ── 仓库配置（遵循平台镜像规范） ──
REGISTRY="10.121.177.20:8082"
PROJECT="mlops"
IMAGE_NAME="model-evaluate"
IMAGE_VERSION="main-py310-cu128-r1"
IMAGE="${REGISTRY}/${PROJECT}/${IMAGE_NAME}:${IMAGE_VERSION}"

# ── 切换到项目根目录 ──
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

DOCKERFILE_PATH="job-template/job/${IMAGE_NAME}/Dockerfile"

echo "=========================================="
echo "构建镜像: ${IMAGE}"
echo "Context:  ${PROJECT_ROOT}"
echo "Dockerfile: ${DOCKERFILE_PATH}"
echo "=========================================="

# ── 构建 ──
docker build --network=host \
    -t "${IMAGE}" \
    -f "${DOCKERFILE_PATH}" \
    .

# ── 冒烟测试：验证版本/补丁/入口脚本 ──
echo ""
echo "=== 冒烟测试 ==="

# 1. 版本断言: transformers 5.x + torch 2.7 + opencompass 可导入
docker run --rm --entrypoint python3 "${IMAGE}" -c "
import opencompass, transformers, torch
assert transformers.__version__.split('.')[0] == '5', \
    f'transformers 版本不符: {transformers.__version__}'
assert torch.__version__.split('.')[0] == '2' and torch.__version__.split('.')[1] == '7', \
    f'torch 版本不符: {torch.__version__}'
# sm_120 (RTX 5090 Blackwell) kernel 支持断言
if torch.cuda.is_available():
    archs = torch.cuda.get_arch_list()
    assert any('120' in a for a in archs), f'GPU 环境下 arch 不含 sm_120: {archs}'
    arch_info = 'GPU arch: ' + str(archs)
else:
    assert torch.version.cuda and torch.version.cuda.startswith('12.8'), \
        f'无 GPU 环境, CUDA 版本不符: {torch.version.cuda}'
    arch_info = '无 GPU 构建机, CUDA ' + str(torch.version.cuda) + ' (cu128 wheel 含 sm_120)'
print('opencompass', opencompass.__version__,
      '| transformers', transformers.__version__,
      '| torch', torch.__version__, '|', arch_info)
" || { echo "[FAIL] 版本断言失败"; exit 1; }

# 2. 补丁生效断言: opencompass 内不应再有 batch_encode_plus 调用
#    （按调用模式匹配，注释中的字样不算误报）
docker run --rm --entrypoint python3 "${IMAGE}" -c "
import opencompass, pathlib
pkg = pathlib.Path(opencompass.__file__).parent
hits = []
for f in pkg.rglob('*.py'):
    if 'tokenizer.batch_encode_plus(' in f.read_text(errors='ignore'):
        hits.append(str(f))
assert not hits, f'补丁未生效，残留调用: {hits}'
print('[OK] batch_encode_plus 补丁验证通过')
" || { echo "[FAIL] 补丁验证失败"; exit 1; }

# 3. 入口脚本语法检查
docker run --rm --entrypoint python3 "${IMAGE}" -m py_compile \
    /app/run_evaluation.py /app/parse_and_save.py \
    || { echo "[FAIL] 入口脚本语法错误"; exit 1; }

echo "[OK] 冒烟测试通过"

# ── 推送 ──
echo ""
echo "=== 推送镜像 ==="
docker push "${IMAGE}"

echo ""
echo "=========================================="
echo "构建推送完成！"
echo "镜像地址: ${IMAGE}"
echo "=========================================="
