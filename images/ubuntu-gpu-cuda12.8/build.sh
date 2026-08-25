#!/bin/bash
set -ex

# RTX 5090 D / Blackwell (sm_120) 基础镜像
# 若构建机拉不到 Docker Hub，先把 nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
# 导入内网仓库，再改 FROM_IMAGES。

FROM_IMAGES=${FROM_IMAGES:-nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04}
HUB=${HUB:-10.121.177.20:8082/mlops}
TAG=ubuntu-gpu:cuda12.8.1-cudnn-python3.11

cd "$(dirname "$0")"

docker build --network=host \
  --build-arg FROM_IMAGES="${FROM_IMAGES}" \
  -t "${HUB}/${TAG}" \
  -f Dockerfile .

docker push "${HUB}/${TAG}"

echo "image: ${HUB}/${TAG}"
