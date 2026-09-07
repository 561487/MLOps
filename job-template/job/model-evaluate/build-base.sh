#!/bin/bash
# job-template/job/model-evaluate/build-base.sh
# 构建并推送 model-evaluate 基础镜像（重依赖层，变更频率低）
#
# Tag 规范: base-py310-cu128-r<N>（依赖发生变化时递增 revision）
# 注意: 递增 revision 而非覆盖，保留历史 Base Runtime，不覆盖已用于流水线的历史 tag。
# 仅当升级 PyTorch / OpenCompass / transformers / 系统补丁时需重新执行。
#
# 用法: cd /path/to/project/root && bash job-template/job/model-evaluate/build-base.sh

set -euo pipefail

REGISTRY="10.121.177.20:8082"
PROJECT="mlops"
BASE_IMAGE_NAME="model-evaluate"
BASE_IMAGE_VERSION="base-py310-cu128-r1"
BASE_IMAGE="${REGISTRY}/${PROJECT}/${BASE_IMAGE_NAME}:${BASE_IMAGE_VERSION}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

DOCKERFILE_PATH="job-template/job/model-evaluate/Dockerfile.base"

echo "=========================================="
echo "构建 BASE 镜像: ${BASE_IMAGE}"
echo "Context:  ${PROJECT_ROOT}"
echo "Dockerfile: ${DOCKERFILE_PATH}"
echo "=========================================="

docker build --network=host \
    -t "${BASE_IMAGE}" \
    -f "${DOCKERFILE_PATH}" \
    .

echo ""
echo "=== BASE 冒烟测试 ==="

docker run --rm --entrypoint python3 "${BASE_IMAGE}" -c "
import opencompass, transformers, torch, faiss
assert transformers.__version__.split('.')[0] == '5', \
    f'transformers 版本不符: {transformers.__version__}'
assert torch.__version__.split('.')[0] == '2' and torch.__version__.split('.')[1] == '7', \
    f'torch 版本不符: {torch.__version__}'
assert faiss.__version__ == '1.15.0', \
    f'faiss 版本不符: {faiss.__version__}'
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
      '| torch', torch.__version__,
      '| faiss', faiss.__version__,
      '|', arch_info)
" || { echo "[FAIL] 版本断言失败"; exit 1; }

docker run --rm --entrypoint python3 "${BASE_IMAGE}" -c "
import opencompass, pathlib
pkg = pathlib.Path(opencompass.__file__).parent
hits = []
for f in pkg.rglob('*.py'):
    text = f.read_text(errors='ignore')
    if 'tokenizer.batch_encode_plus(' in text or 'tokenizer.encode_plus(' in text:
        hits.append(str(f))
assert not hits, f'补丁未生效，残留调用: {hits}'
print('[OK] encode_plus / batch_encode_plus 补丁验证通过')
" || { echo "[FAIL] 补丁验证失败"; exit 1; }

echo "[OK] BASE 冒烟测试通过"

echo ""
echo "=== 推送 BASE 镜像 ==="
docker push "${BASE_IMAGE}"

echo ""
echo "=========================================="
echo "BASE 构建推送完成！"
echo "镜像地址: ${BASE_IMAGE}"
echo ""
echo "下一步: 更新 Dockerfile 中 BASE_IMAGE / build.sh 中 BASE_IMAGE_VERSION（如有变更），"
echo "        然后执行 bash job-template/job/model-evaluate/build.sh 构建业务镜像。"
echo "=========================================="
