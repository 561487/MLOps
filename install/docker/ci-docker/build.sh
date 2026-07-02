#!/bin/bash

set -ex

IMAGE=${IMAGE:-10.121.177.20:8082/mlops/ci-tools:20260702}
docker build --network=host -t "${IMAGE}" -f Dockerfile .
docker push "${IMAGE}"