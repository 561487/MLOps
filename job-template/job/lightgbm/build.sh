set -ex

IMAGE=${IMAGE:-10.121.177.20:8082/mlops/lightgbm:20260618}

docker build --network=host -t "${IMAGE}" -f Dockerfile .
docker push "${IMAGE}"
