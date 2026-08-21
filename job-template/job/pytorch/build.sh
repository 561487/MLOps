#!/bin/bash

set -ex

docker build --network=host -t 10.121.177.20:8082/mlops/pytorch:20260820 -f job/pytorch/Dockerfile .
docker push 10.121.177.20:8082/mlops/pytorch:20260820


