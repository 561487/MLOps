#!/bin/bash
set -ex
IMAGE=${IMAGE:-10.121.177.20:8082/mlops/msswift-base:20260710-v1}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

docker build --network=host -t "${IMAGE}" -f - "${SCRIPT_DIR}" <<'EOF'
FROM modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.8.1-py311-torch2.9.0-vllm0.13.0-modelscope1.33.0-swift3.12.5

ENV TZ=Asia/Shanghai
ENV DEBIAN_FRONTEND=noninteractive
RUN curl -fsSL https://github.com/stern/stern/releases/download/v1.26.0/stern_1.26.0_linux_amd64.tar.gz | tar xz -C /usr/local/bin stern && chmod +x /usr/local/bin/stern


RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple
RUN pip install kubernetes==25.3.0 psutil requests
RUN pip install deepspeed
RUN pip install pybind11 'transformer_engine[pytorch]' --no-build-isolation
RUN pip install 'megatron-core==0.15.0'
RUN pip install flash-attn --no-build-isolation
EOF

docker push "${IMAGE}"
echo "Built and pushed: ${IMAGE}"



