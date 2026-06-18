#!/bin/bash

set -ex

REGISTRY=10.121.177.20:8082/mlops
TAG=20250601

docker build --network=host -t ${REGISTRY}/llama_factory:${TAG} -f job/llama_factory/Dockerfile .
docker push ${REGISTRY}/llama_factory:${TAG}
