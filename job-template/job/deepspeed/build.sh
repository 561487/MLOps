#!/bin/bash

set -ex

REGISTRY=${REGISTRY:-10.121.177.20:8082/mlops}
TAG=${TAG:-20260716-cuda121-ds0144-hf4442-r3}

docker build --network=host -t ${REGISTRY}/deepspeed:${TAG} -f job/deepspeed/Dockerfile .
docker push ${REGISTRY}/deepspeed:${TAG}
