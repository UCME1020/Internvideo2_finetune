#!/usr/bin/env bash
# BLiM fork of scripts/finetuning/stage2/6B/didemo/run.sh
# Run from: /data5/jyhong/BLiM/InternVideo2/multi_modality
#   bash scripts/finetuning/stage2/6B/didemo_blim/run.sh
#
# Env overrides:
#   NUM_GPUS=4 bash ...   # use 4 GPUs (matches official RTX4090 ×4 recipe)
#   CUDA_VISIBLE_DEVICES="0,1,2,3" bash ...   # pin GPUs

export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

# Locations (override via env if needed).
export INTERNVIDEO2_MODEL_PATH="${INTERNVIDEO2_MODEL_PATH:-/data5/jyhong/BLiM/InternVideo2/multi_modality/pretrained}"
# INTERNVIDEO2_DATA_PATH unused — data paths are inlined in config.py.

which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH="${PYTHONPATH}:."
echo "PYTHONPATH: ${PYTHONPATH}"

JOB_NAME=$(basename $0)_$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="$(dirname $0)/$JOB_NAME"
LOG_DIR="$(dirname $0)/logs/${JOB_NAME}"
mkdir -p "${LOG_DIR}"

NNODE=1
NUM_GPUS="${NUM_GPUS:-8}"

PRETRAINED_PT="${INTERNVIDEO2_MODEL_PATH}/internvideo2-s2_6b-224p-f4_with_audio_encoder.pt"
if [[ ! -f "${PRETRAINED_PT}" ]]; then
    echo "ERROR: pretrained checkpoint not found at ${PRETRAINED_PT}" >&2
    exit 1
fi
BEATS_PT="${INTERNVIDEO2_MODEL_PATH}/BEATs_iter3_plus_AS2M.pt"
if [[ ! -f "${BEATS_PT}" ]]; then
    echo "ERROR: BEATs checkpoint not found at ${BEATS_PT}" >&2
    exit 1
fi

echo "===================================================="
echo "JOB_NAME       = ${JOB_NAME}"
echo "NUM_GPUS       = ${NUM_GPUS}"
echo "PRETRAINED_PT  = ${PRETRAINED_PT}"
echo "BEATS_PT       = ${BEATS_PT}"
echo "OUTPUT_DIR     = ${OUTPUT_DIR}"
echo "===================================================="

torchrun \
    --nnodes=${NNODE} \
    --nproc_per_node=${NUM_GPUS} \
    --rdzv_id=12345 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:${MASTER_PORT} \
    tasks/pretrain.py \
    $(dirname $0)/config.py \
    output_dir "${OUTPUT_DIR}" \
    pretrained_path "${PRETRAINED_PT}" \
    2>&1 | tee "${LOG_DIR}/train.log"
