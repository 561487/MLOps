#!/bin/bash

set -ex

docker build --network=host -t 10.121.177.20:8082/mlops/tf:20260821 -f job/tf/Dockerfile .
docker push 10.121.177.20:8082/mlops/tf:20260821


