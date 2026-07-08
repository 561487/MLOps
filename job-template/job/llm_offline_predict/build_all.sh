#!/bin/bash
# 一键构建推送（改完 image_tags.conf 后执行这一行即可）
set -e
cd "$(dirname "$0")"
source image_tags.conf

echo ">>> 同步 init-job-template.json..."
sed -i "s|\"image_name\": \"10.121.177.20:8082/mlops/offline-predict-launcher:[^\"]*\"|\"image_name\": \"$OFFLINE_PREDICT_LAUNCHER\"|g" ../../../myapp/init/init-job-template.json

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
