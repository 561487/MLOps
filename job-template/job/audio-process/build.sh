#!/bin/bash
set -euo pipefail

IMAGE=${IMAGE:-10.121.177.20:8082/mlops/audio-process:20260630}

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "${ROOT_DIR}"

echo "Project root: ${ROOT_DIR}"
echo "Building image: ${IMAGE}"

docker build --no-cache \
  -f job-template/job/audio-process/Dockerfile \
  -t "${IMAGE}" \
  .

echo ""
echo "=== Testing image ==="

docker run --rm --entrypoint python3 "${IMAGE}" --version

docker run --rm --entrypoint python3 "${IMAGE}" - <<'PY'
import sys
print("python:", sys.version)
modules = ["numpy", "pandas", "librosa", "soundfile", "scipy", "imageio_ffmpeg"]
for m in modules:
    try:
        __import__(m)
        print("[OK]", m)
    except Exception as e:
        print("[FAIL]", m, repr(e))
        raise
PY

docker run --rm --entrypoint ffmpeg "${IMAGE}" -version 2>&1 | head -1 || echo "ffmpeg check skipped"

docker run --rm "${IMAGE}" --help 2>&1 | head -3 || true

echo ""
echo "=== Pushing image ==="
docker push "${IMAGE}"

echo ""
echo "=== Done: ${IMAGE} ==="
