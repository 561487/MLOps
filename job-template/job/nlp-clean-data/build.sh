#!/bin/bash

set -ex

docker build --network=host -t ccr.ccs.tencentyun.com/cube-studio/nlp-clean-data:20250601 -f job/nlp-clean-data/Dockerfile .
docker push ccr.ccs.tencentyun.com/cube-studio/nlp-clean-data:20250601

# docker buildx build --platform linux/amd64,linux/arm64 -t ccr.ccs.tencentyun.com/cube-studio/nlp-clean-data:20250601 -f job/nlp-clean-data/Dockerfile . --push
