#!/usr/bin/env bash
# Launcher for IV2-1B DiDeMo finetune (BLiM fork).
#
# Prereq: run setup_iv2_6b_didemo.sh once first (sets up conda env,
# upstream repo clone, etc. — the 1B run reuses the same env).
#
# Usage:
#   bash train_iv2_1b_didemo.sh             # 4 GPUs (default; matches upstream 6B recipe)
#   NUM_GPUS=8 bash train_iv2_1b_didemo.sh  # override to 8 GPUs
#   SMOKE=1 bash train_iv2_1b_didemo.sh     # 1-GPU eval-only smoke test

set -euo pipefail

IV2_REPO="/data5/ucjung/PoLaRT/InternVideo"
IV2_MM="${IV2_REPO}/InternVideo2/multi_modality"
KIT_DIR="/data5/ucjung/PoLaRT/Internvideo2_finetune"

export INTERNVIDEO2_MODEL_PATH="${INTERNVIDEO2_MODEL_PATH:-/data5/ucjung/PoLaRT/pretrained/iv2_1b}"
export BLIM_ANNO_DIR="${BLIM_ANNO_DIR:-${KIT_DIR}/data/DiDeMo}"
export BLIM_VIDEO_DIR="${BLIM_VIDEO_DIR:-/data1/jyhong/workspace/ECCV2026/dataset/DiDeMo/DiDeMo}"

CONDA_ENV_NAME="iv2_6b"  # same env (torch 2.5.1 + cu121 + flash-attn 2.7.4)

if [[ ! -d "${IV2_MM}" ]]; then
    echo "ERROR: upstream not found at ${IV2_MM} — run setup_iv2_6b_didemo.sh first" >&2
    exit 1
fi

for f in \
    "${INTERNVIDEO2_MODEL_PATH}/InternVideo2-stage2_1b-224p-f4.pt" \
    "${BLIM_ANNO_DIR}/didemo_ret_train_filtered.json" \
    "${BLIM_ANNO_DIR}/didemo_ret_test_filtered.json"; do
    if [[ ! -f "${f}" ]]; then
        echo "ERROR: required file missing: ${f}" >&2
        exit 1
    fi
done

set +u
CONDA_BASE="$(conda info --base 2>/dev/null || echo /usr/local)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
export CUDA_HOME="${CONDA_PREFIX}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
set -u

echo "===================================================="
echo "  python   = $(which python)"
echo "  CUDA     = $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "  NUM_GPUS = ${NUM_GPUS:-4}"
echo "  SMOKE    = ${SMOKE:-0}"
echo "===================================================="

cd "${IV2_MM}"

if [[ "${SMOKE:-0}" == "1" ]]; then
    NUM_GPUS=1 CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
        bash scripts/finetuning/stage2/1B/didemo_blim/smoke.sh
else
    NUM_GPUS="${NUM_GPUS:-4}" \
        bash scripts/finetuning/stage2/1B/didemo_blim/run.sh
fi
