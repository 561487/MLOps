#!/bin/bash

set -ex

REGISTRY=${REGISTRY:-10.121.177.20:8082/mlops}
TAG=${TAG:-20260720}

docker build --network=host -t ${REGISTRY}/paddle-job:${TAG} -f job/paddle/Dockerfile .
docker push ${REGISTRY}/paddle-job:${TAG}

