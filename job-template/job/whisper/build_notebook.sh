#!/usr/bin/env bash
set -euo pipefail

IMAGE="${WHISPER_NOTEBOOK_IMAGE:-10.121.177.20:8082/mlops/whisper-notebook:20260811}"

docker build --network=host -t "${IMAGE}" -f Dockerfile.notebook .
docker push "${IMAGE}"
