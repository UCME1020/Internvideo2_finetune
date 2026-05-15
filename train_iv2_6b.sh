#!/usr/bin/env bash
# Multi-dataset launcher for IV2-6B BLiM finetune.
#
# Prereq: bash setup_iv2_6b_didemo.sh (one-time)
#
# Usage:
#   bash train_iv2_6b.sh <dataset>                 # 4 GPU full finetune (matches upstream)
#   SMOKE=1 bash train_iv2_6b.sh <dataset>         # 1 GPU eval-only smoke
#   NUM_GPUS=8 bash train_iv2_6b.sh <dataset>      # override to 8 GPU
#
# <dataset> in: didemo | activitynet | msrvtt | lsmdc

set -euo pipefail

DATASET="${1:-didemo}"
case "${DATASET}" in
    didemo|activitynet|msrvtt|lsmdc) ;;
    *) echo "ERROR: unknown dataset '${DATASET}'. Pick: didemo|activitynet|msrvtt|lsmdc" >&2; exit 1 ;;
esac
SCRIPT_DIR="scripts/finetuning/stage2/6B/${DATASET}_blim"

# ───────────────────────── paths ─────────────────────────
IV2_REPO="/data5/ucjung/PoLaRT/InternVideo"
IV2_MM="${IV2_REPO}/InternVideo2/multi_modality"
KIT_DIR="/data5/ucjung/PoLaRT/Internvideo2_finetune"

export INTERNVIDEO2_MODEL_PATH="${INTERNVIDEO2_MODEL_PATH:-/data5/ucjung/PoLaRT/pretrained/iv2_6b}"

CONDA_ENV_NAME="iv2_6b"

# ───────────────────────── sanity ─────────────────────────
if [[ ! -d "${IV2_MM}/${SCRIPT_DIR}" ]]; then
    echo "ERROR: ${IV2_MM}/${SCRIPT_DIR} not found — did you run setup?" >&2
    exit 1
fi

for f in \
    "${INTERNVIDEO2_MODEL_PATH}/internvideo2-s2_6b-224p-f4_with_audio_encoder.pt" \
    "${INTERNVIDEO2_MODEL_PATH}/BEATs_iter3_plus_AS2M.pt"; do
    if [[ ! -f "${f}" ]]; then
        echo "ERROR: required ckpt missing: ${f}" >&2
        exit 1
    fi
done

# ───────────── activate conda env ─────────────
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
echo "  DATASET    = ${DATASET}"
echo "  SCRIPT_DIR = ${SCRIPT_DIR}"
echo "  python     = $(which python)"
echo "  CUDA       = $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "  NUM_GPUS   = ${NUM_GPUS:-4}"
echo "  SMOKE      = ${SMOKE:-0}"
echo "===================================================="

cd "${IV2_MM}"

if [[ "${SMOKE:-0}" == "1" ]]; then
    NUM_GPUS=1 CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
        bash "${SCRIPT_DIR}/smoke.sh"
else
    NUM_GPUS="${NUM_GPUS:-4}" \
        bash "${SCRIPT_DIR}/run.sh"
fi
