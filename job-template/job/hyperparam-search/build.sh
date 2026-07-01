#!/usr/bin/env bash
set -e

IMAGE=${1:-10.121.177.20:8082/mlops/hyperparam-search:20260626-v3}

echo "Building image: ${IMAGE}"

cd "$(dirname "$0")"

if [ ! -f Dockerfile ]; then
    echo "ERROR: Dockerfile not found in $(pwd)"
    exit 1
fi
if [ ! -f launcher.py ]; then
    echo "ERROR: launcher.py not found in $(pwd)"
    exit 1
fi

echo "Build context: $(pwd)"
echo "Files in context:"
ls -lah

docker build --network=host --no-cache -t "${IMAGE}" -f Dockerfile .

echo "Build success: ${IMAGE}"
echo "Pushing ..."
docker push "${IMAGE}"
echo "Push success: ${IMAGE}"
