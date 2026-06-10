#!/bin/bash

set -ex

IMAGE=${IMAGE:-10.121.177.20:8082/mlops/vision-process:20260610}
docker build --network=host -t "${IMAGE}" -f Dockerfile .
docker push "${IMAGE}"

