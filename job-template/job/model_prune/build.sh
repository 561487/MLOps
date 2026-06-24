#!/bin/bash
set -ex

IMAGE=${IMAGE:-10.121.177.20:8082/mlops/torch-pruning:1.0.0}

docker login 10.121.177.20:8082 -u admin -p Harbor@12345

docker build --no-cache --network=host -t "${IMAGE}" -f Dockerfile .
docker push "${IMAGE}"

echo "构建完成: ${IMAGE}"
