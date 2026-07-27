#!/bin/bash
set -euo pipefail

REGISTRY="${REGISTRY:-10.121.177.20:8082}"
IMAGE="${IMAGE:-${REGISTRY}/mlops/gptqmodel:3.2.0}"

if [[ -n "${HARBOR_USERNAME:-}" && -n "${HARBOR_PASSWORD:-}" ]]; then
  printf '%s' "${HARBOR_PASSWORD}" | docker login "${REGISTRY}" \
    --username "${HARBOR_USERNAME}" --password-stdin
else
  echo "未提供 HARBOR_USERNAME/HARBOR_PASSWORD，使用现有 Docker 登录会话"
fi

docker build --network=host --pull -t "${IMAGE}" -f Dockerfile .
docker push "${IMAGE}"
echo "构建完成: ${IMAGE}"
