#!/bin/bash
set -ex
IMAGE=${IMAGE:-10.121.177.20:8082/mlops/msswift-base:20260819-qwen35-v4}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

docker build --network=host -t "${IMAGE}" -f - "${SCRIPT_DIR}" <<'EOF'
# v2 contains the validated Qwen3.5 stack and compiled CUDA extensions.
FROM 10.121.177.20:8082/mlops/msswift-base:20260818-qwen35-v2

ENV TZ=Asia/Shanghai
ENV DEBIAN_FRONTEND=noninteractive
RUN pip install --no-cache-dir 'mistral-common==1.11.7'
RUN pip check
EOF

docker push "${IMAGE}"
echo "Built and pushed: ${IMAGE}"



