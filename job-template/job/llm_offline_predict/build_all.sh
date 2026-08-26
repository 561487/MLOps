#!/bin/bash
# 一键构建推送 launcher + worker
set -e
cd "$(dirname "$0")"

# 镜像配置（按规范第 4 节统一变量名）
REGISTRY="10.121.177.20:8082"
PROJECT="mlops"
IMAGE_VERSION="1.0.0-20260825-r1"

LAUNCHER_NAME="model-offline-predict-launcher"
WORKER_NAME="model-offline-predict"

OFFLINE_PREDICT_LAUNCHER="${REGISTRY}/${PROJECT}/${LAUNCHER_NAME}:${IMAGE_VERSION}"
LLM_OFFLINE_PREDICT="${REGISTRY}/${PROJECT}/${WORKER_NAME}:${IMAGE_VERSION}"

echo ">>> 同步 init-job-template.json..."
sed -i "s|\"image_name\": \"10.121.177.20:8082/mlops/model-offline-predict-launcher:[^\"]*\"|\"image_name\": \"$OFFLINE_PREDICT_LAUNCHER\"|g" ../../../myapp/init/init-job-template.json

echo ">>> 构建推送 launcher: $OFFLINE_PREDICT_LAUNCHER"
cd ..
docker build -t $OFFLINE_PREDICT_LAUNCHER -f model_offline_predict/Dockerfile .
docker push $OFFLINE_PREDICT_LAUNCHER

echo ">>> 构建推送 worker: $LLM_OFFLINE_PREDICT"
cd llm_offline_predict
docker build -t $LLM_OFFLINE_PREDICT .
docker push $LLM_OFFLINE_PREDICT

echo ">>> 完成!"
echo "launcher: $OFFLINE_PREDICT_LAUNCHER"
echo "worker:   $LLM_OFFLINE_PREDICT"
