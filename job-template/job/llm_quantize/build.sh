#!/bin/bash
set -ex

# 镜像名称（可以外部覆盖）
IMAGE=${IMAGE:-10.121.177.20:8082/mlops/gptqmodel:3.0.6}

# 登录镜像仓库
docker login 10.121.177.20:8082 -u admin -p Harbor@12345

# 构建镜像（--network=host 使用宿主机网络）
docker build --network=host -t "${IMAGE}" -f Dockerfile .

# 推送到仓库
docker push "${IMAGE}"

echo "构建完成: ${IMAGE}"
