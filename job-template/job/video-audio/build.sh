#!/bin/bash

set -ex

docker build --network=host -t 10.121.177.20:8082/mlops/video-audio:20260821 -f job/video-audio/Dockerfile .
docker push 10.121.177.20:8082/mlops/video-audio:20260821





