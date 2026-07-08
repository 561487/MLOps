#!/bin/bash
cd "$(dirname "$0")"
source image_tags.conf
docker build -t $LLM_OFFLINE_PREDICT .
docker push $LLM_OFFLINE_PREDICT
echo "Pushed: $LLM_OFFLINE_PREDICT"
