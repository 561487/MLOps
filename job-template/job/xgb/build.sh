#!/usr/bin/env bash

set -euo pipefail

component_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
image_name="${1:-10.121.177.20:8082/mlops/xgb:20260721}"

test -f "${component_dir}/launcher.py"
test -f "${component_dir}/Dockerfile"

docker build -t "${image_name}" "${component_dir}"
docker run --rm "${image_name}" python3 launcher.py --help

echo "XGBoost image built and checked: ${image_name}"
