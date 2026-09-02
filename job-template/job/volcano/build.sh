#!/bin/bash

set -ex

docker build --network=host -t 10.121.177.20:8082/mlops/volcano:20260821 -f job/volcano/Dockerfile .
docker push 10.121.177.20:8082/mlops/volcano:20260821



