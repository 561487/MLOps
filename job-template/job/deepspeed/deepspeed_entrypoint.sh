#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DS_CONFIG_PATH="${DS_CONFIG_PATH:-/tmp/ds_config.json}"

echo "[deepspeed_entrypoint] generate DeepSpeed config: ${DS_CONFIG_PATH}"
python3 "${SCRIPT_DIR}/ds_config.py" "$@" --output "${DS_CONFIG_PATH}"

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
NODE_RANK="${RANK:-${NODE_RANK:-0}}"
NUM_NODES="${WORLD_SIZE:-${NUM_WORKER:-1}}"
NUM_GPUS="${GPU_NUM:-${NPROC_PER_NODE:-1}}"

if [ "${NUM_GPUS}" -lt 1 ]; then
  NUM_GPUS=1
fi

echo "[deepspeed_entrypoint] MASTER_ADDR=${MASTER_ADDR}"
echo "[deepspeed_entrypoint] MASTER_PORT=${MASTER_PORT}"
echo "[deepspeed_entrypoint] NODE_RANK=${NODE_RANK}"
echo "[deepspeed_entrypoint] NUM_NODES=${NUM_NODES}"
echo "[deepspeed_entrypoint] NUM_GPUS=${NUM_GPUS}"

exec torchrun \
  --nnodes "${NUM_NODES}" \
  --nproc-per-node "${NUM_GPUS}" \
  --node-rank "${NODE_RANK}" \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT}" \
  "${SCRIPT_DIR}/train_runner.py" \
  "$@" \
  --deepspeed_config "${DS_CONFIG_PATH}"
