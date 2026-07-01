#!/usr/bin/env bash
set -e

IMAGE=${1:-10.121.177.20:8082/mlops/hyperparam-search-nni:20260626-v3}

echo "Building image: ${IMAGE}"

cd "$(dirname "$0")"

echo "[INFO] build context: $(pwd)"
ls -lah

test -f Dockerfile
test -f launcher.py

docker build --no-cache --network=host -t "${IMAGE}" -f Dockerfile .

echo "Build success: ${IMAGE}"

docker push "${IMAGE}"

echo "Push success: ${IMAGE}"

echo "[INFO] Verify image content..."
docker run --rm --entrypoint bash "${IMAGE}" -lc '
pwd
ls -lah /app
test -f /app/launcher.py && echo "launcher.py exists"
python3 -c "import numpy, pandas, scipy, sklearn, joblib; print(\"deps ok\")"
python3 /app/launcher.py --help 2>&1 | head -5
'
