#!/bin/bash

set -ex

docker build --network=host -t ccr.ccs.tencentyun.com/cube-studio/nlp-filter-data:20250601 -f job/nlp-filter-data/Dockerfile .
docker push ccr.ccs.tencentyun.com/cube-studio/nlp-filter-data:20250601

# docker buildx build --platform linux/amd64,linux/arm64 -t ccr.ccs.tencentyun.com/cube-studio/nlp-filter-data:20250601 -f job/nlp-filter-data/Dockerfile . --push
