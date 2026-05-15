#!/usr/bin/env bash
set -o pipefail
# IV2-1B + DiDeMo finetune launcher (BLiM fork).
# Run from: <iv2 multi_modality root>
#   bash scripts/finetuning/stage2/1B/didemo_blim/run.sh
#
# Env overrides:
#   NUM_GPUS=8 bash ...        # override from default 4
#   CUDA_VISIBLE_DEVICES=...   # pin GPUs

export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

export INTERNVIDEO2_MODEL_PATH="${INTERNVIDEO2_MODEL_PATH:-/data5/ucjung/PoLaRT/pretrained/iv2_1b}"

which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH="${PYTHONPATH}:."
echo "PYTHONPATH: ${PYTHONPATH}"

JOB_NAME=$(basename $0)_$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="$(dirname $0)/$JOB_NAME"
LOG_DIR="$(dirname $0)/logs/${JOB_NAME}"
mkdir -p "${LOG_DIR}"

NNODE=1
NUM_GPUS="${NUM_GPUS:-4}"

PRETRAINED_PT="${INTERNVIDEO2_MODEL_PATH}/InternVideo2-stage2_1b-224p-f4.pt"
if [[ ! -f "${PRETRAINED_PT}" ]]; then
    echo "ERROR: pretrained checkpoint not found at ${PRETRAINED_PT}" >&2
    exit 1
fi

echo "===================================================="
echo "JOB_NAME       = ${JOB_NAME}"
echo "NUM_GPUS       = ${NUM_GPUS}"
echo "PRETRAINED_PT  = ${PRETRAINED_PT}"
echo "OUTPUT_DIR     = ${OUTPUT_DIR}"
echo "===================================================="

torchrun \
    --nnodes=${NNODE} \
    --nproc_per_node=${NUM_GPUS} \
    --rdzv_id=12346 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:${MASTER_PORT} \
    tasks/pretrain.py \
    $(dirname $0)/config.py \
    output_dir "${OUTPUT_DIR}" \
    pretrained_path "${PRETRAINED_PT}" \
    2>&1 | tee "${LOG_DIR}/train.log"
