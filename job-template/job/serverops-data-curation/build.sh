#!/bin/bash

set -euo pipefail

IMAGE="${IMAGE:-10.121.177.20:8082/mlops/serverops-data-curation:20260723}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_CONTEXT="$(cd "$SCRIPT_DIR/../.." && pwd)"

docker build --network=host -t "${IMAGE}" -f "${SCRIPT_DIR}/Dockerfile" "${BUILD_CONTEXT}"
docker push "${IMAGE}"
