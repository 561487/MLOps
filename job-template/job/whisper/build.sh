#!/usr/bin/env bash
set -euo pipefail

IMAGE="${WHISPER_IMAGE:-10.121.177.20:8082/mlops/whisper:20260811}"

docker build --network=host -t "${IMAGE}" -f Dockerfile .
docker push "${IMAGE}"
