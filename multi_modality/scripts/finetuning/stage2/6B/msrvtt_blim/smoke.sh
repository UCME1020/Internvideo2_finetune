#!/usr/bin/env bash
# 1-GPU eval-only smoke test:
#   bash scripts/finetuning/stage2/6B/didemo_blim/smoke.sh
# Verifies ckpt load + data load + model build + eval forward end-to-end
# before committing to a long 8-GPU run.
export MASTER_PORT=$((22000 + $RANDOM % 5000))
export OMP_NUM_THREADS=1
export INTERNVIDEO2_MODEL_PATH="${INTERNVIDEO2_MODEL_PATH:-/data5/jyhong/BLiM/InternVideo2/multi_modality/pretrained}"
export PYTHONPATH="${PYTHONPATH}:."

JOB_NAME="smoke_$(date +"%Y%m%d_%H%M%S")"
OUTPUT_DIR="$(dirname $0)/${JOB_NAME}"
mkdir -p "${OUTPUT_DIR}"

PRETRAINED_PT="${INTERNVIDEO2_MODEL_PATH}/internvideo2-s2_6b-224p-f4_with_audio_encoder.pt"

echo "Smoke test: 1 GPU, evaluate=True, batch=2, k_test=16"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" torchrun \
    --nnodes=1 --nproc_per_node=1 \
    --rdzv_id=99999 --rdzv_backend=c10d \
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
