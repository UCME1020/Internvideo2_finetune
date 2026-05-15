#!/usr/bin/env bash
# 1-GPU eval-only smoke test for IV2-1B + DiDeMo (BLiM fork).
# Verifies ckpt load + data load + model build + eval forward end-to-end
# before committing to a long full finetune.
#
# Run from: <iv2 multi_modality root>
#   bash scripts/finetuning/stage2/1B/didemo_blim/smoke.sh
set -o pipefail
export MASTER_PORT=$((22000 + $RANDOM % 5000))
export OMP_NUM_THREADS=1
export INTERNVIDEO2_MODEL_PATH="${INTERNVIDEO2_MODEL_PATH:-/data5/ucjung/PoLaRT/pretrained/iv2_1b}"
export PYTHONPATH="${PYTHONPATH}:."

JOB_NAME="smoke_$(date +"%Y%m%d_%H%M%S")"
OUTPUT_DIR="$(dirname $0)/${JOB_NAME}"
mkdir -p "${OUTPUT_DIR}"

PRETRAINED_PT="${INTERNVIDEO2_MODEL_PATH}/InternVideo2-stage2_1b-224p-f4.pt"
if [[ ! -f "${PRETRAINED_PT}" ]]; then
    echo "ERROR: pretrained checkpoint not found at ${PRETRAINED_PT}" >&2
    exit 1
fi

echo "Smoke test: 1 GPU, evaluate=True, batch=2, k_test=16"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" torchrun \
    --nnodes=1 --nproc_per_node=1 \
    --rdzv_id=99998 --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:${MASTER_PORT} \
    tasks/pretrain.py \
    $(dirname $0)/config.py \
    output_dir "${OUTPUT_DIR}" \
    pretrained_path "${PRETRAINED_PT}" \
    evaluate True \
    zero_shot True \
    batch_size 2 \
    batch_size_test 2 \
    evaluation.k_test 16 \
    2>&1 | tee "${OUTPUT_DIR}/smoke.log"
