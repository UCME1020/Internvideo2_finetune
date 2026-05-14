#!/usr/bin/env bash
# Zero-shot evaluation: InternVideo2-CLIP-6B ISO (flash_attn+FusedMLP) on DiDeMo test 1k.
# Run from: /data5/jyhong/BLiM/InternVideo2/multi_modality
#   CUDA_VISIBLE_DEVICES=0 bash scripts/evaluation/clip/zero_shot/6B/didemo_blim/eval.sh
#
# Required ckpts under ${INTERNVIDEO2_MODEL_PATH}:
#   InternVideo2_Stage2_6B.pth     (vision backbone, symlinked)
#   internvl/internvl_c_13b_224px.pth  (LLaMA text encoder)
#   InternVideo2_CLIP_6B.pth       (CLIP bridge head, symlinked from 6B_clip.pth)
#   chinese_alpaca_lora_7b/        (LlamaConfig + LlamaTokenizer metadata)

export MASTER_PORT=$((22000 + $RANDOM % 5000))
export OMP_NUM_THREADS=1
# Reduce fragmentation — ~6 GB sits "reserved but unallocated" otherwise.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export INTERNVIDEO2_MODEL_PATH="${INTERNVIDEO2_MODEL_PATH:-/data5/jyhong/BLiM/InternVideo2/multi_modality/pretrained}"
export PYTHONPATH="${PYTHONPATH}:."

JOB_NAME="zs_didemo_clip6b_iso_$(date +"%Y%m%d_%H%M%S")"
OUTPUT_DIR="$(dirname $0)/${JOB_NAME}"
mkdir -p "${OUTPUT_DIR}"

NUM_GPUS="${NUM_GPUS:-1}"

VISION_PT="${INTERNVIDEO2_MODEL_PATH}/InternVideo2_Stage2_6B.pth"
TEXT_PT="${INTERNVIDEO2_MODEL_PATH}/internvl/internvl_c_13b_224px.pth"
EXTRA_PT="${INTERNVIDEO2_MODEL_PATH}/InternVideo2_CLIP_6B.pth"
ALPACA_DIR="${INTERNVIDEO2_MODEL_PATH}/chinese_alpaca_lora_7b"

for f in "${VISION_PT}" "${TEXT_PT}" "${EXTRA_PT}"; do
    if [[ ! -e "$f" ]]; then
        echo "ERROR: ckpt not found at $f" >&2
        exit 1
    fi
done
if [[ ! -f "${ALPACA_DIR}/config.json" || ! -f "${ALPACA_DIR}/tokenizer.model" ]]; then
    echo "ERROR: chinese_alpaca_lora_7b incomplete at ${ALPACA_DIR}" >&2
    exit 1
fi

echo "===================================================="
echo "Zero-shot: InternVideo2-CLIP-6B on DiDeMo test 1k"
echo "JOB_NAME       = ${JOB_NAME}"
echo "NUM_GPUS       = ${NUM_GPUS}"
echo "VISION_PT      = ${VISION_PT}"
echo "TEXT_PT        = ${TEXT_PT}"
echo "EXTRA_PT       = ${EXTRA_PT}"
echo "OUTPUT_DIR     = ${OUTPUT_DIR}"
echo "===================================================="

torchrun \
    --nnodes=1 --nproc_per_node=${NUM_GPUS} \
    --rdzv_id=88889 --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:${MASTER_PORT} \
    tasks_clip/retrieval.py \
    $(dirname $0)/config.py \
    output_dir "${OUTPUT_DIR}" \
    evaluate True \
    2>&1 | tee "${OUTPUT_DIR}/eval.log"
