#!/bin/bash
# 构建上下文为 job-template/job（包含 pkgs/ 和 model_offline_predict/）
cd "$(dirname "$0")/.."
source llm_offline_predict/image_tags.conf
docker build -t $OFFLINE_PREDICT_LAUNCHER -f model_offline_predict/Dockerfile .
docker push $OFFLINE_PREDICT_LAUNCHER
echo "Pushed: $OFFLINE_PREDICT_LAUNCHER"
