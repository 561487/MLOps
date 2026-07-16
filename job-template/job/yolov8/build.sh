#!/bin/bash

set -ex

docker build --network=host -t 10.121.177.20:8082/mlops/yolov8:20250801 -f Dockerfile  .
docker push 10.121.177.20:8082/mlops/yolov8:20250801

# docker buildx build --platform linux/amd64,linux/arm64 -t 10.121.177.20:8082/mlops/yolov8:20250801 -f Dockerfile . --push

