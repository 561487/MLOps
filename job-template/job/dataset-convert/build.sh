#!/bin/bash

set -ex

IMAGE=${IMAGE:-10.121.177.20:8082/mlops/dataset-convert:20260828}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
docker build --network=host -t "${IMAGE}" -f "${SCRIPT_DIR}/Dockerfile" "${REPO_ROOT}"
docker push "${IMAGE}"

# 冒烟测试：镜像内自检（版本与基础链路）
docker run --rm --entrypoint python3 "${IMAGE}" -c "
import launcher, dataset_io, schema, converter
print('dataset-convert smoke OK: modules importable')
"
