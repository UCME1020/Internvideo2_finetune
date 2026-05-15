#!/usr/bin/env bash
# One-shot setup for InternVideo2-6B DiDeMo finetune (BLiM fork).
#
# Idempotent: re-running skips steps that are already done.
#
# Steps:
#   1) Clone upstream OpenGVLab/InternVideo
#   2) Drop BLiM didemo_blim scripts onto upstream tree
#   3) Copy filtered DiDeMo JSONs to BLIM_ANNO_DIR (already in repo)
#   4) Create conda env "iv2_6b" (torch 2.5.1 + cu121, flash-attn 2.7.4.post1)
#   5) Download HF checkpoints: 6B Stage2 (+audio) + BEATs into ${INTERNVIDEO2_MODEL_PATH}
#
# After this, run: bash train_iv2_6b_didemo.sh

set -euo pipefail

# ───────────────────────── paths ─────────────────────────
KIT_DIR="/data5/ucjung/PoLaRT/Internvideo2_finetune"
IV2_REPO="/data5/ucjung/PoLaRT/InternVideo"
IV2_MM="${IV2_REPO}/InternVideo2/multi_modality"

export INTERNVIDEO2_MODEL_PATH="${INTERNVIDEO2_MODEL_PATH:-/data5/ucjung/PoLaRT/pretrained/iv2_6b}"
export BLIM_ANNO_DIR="${BLIM_ANNO_DIR:-${KIT_DIR}/data/DiDeMo}"
export BLIM_VIDEO_DIR="${BLIM_VIDEO_DIR:-/data1/jyhong/workspace/ECCV2026/dataset/DiDeMo/DiDeMo}"

CONDA_ENV_NAME="iv2_6b"

echo "===================================================="
echo "  IV2-6B DiDeMo finetune — setup"
echo "  IV2_REPO       = ${IV2_REPO}"
echo "  IV2_MM         = ${IV2_MM}"
echo "  MODEL_PATH     = ${INTERNVIDEO2_MODEL_PATH}"
echo "  ANNO_DIR       = ${BLIM_ANNO_DIR}"
echo "  VIDEO_DIR      = ${BLIM_VIDEO_DIR}"
echo "  CONDA_ENV      = ${CONDA_ENV_NAME}"
echo "===================================================="

# ───────────── step 1: clone upstream ─────────────
if [[ ! -d "${IV2_MM}" ]]; then
    echo "==> Cloning OpenGVLab/InternVideo into ${IV2_REPO}"
    git clone --depth=1 https://github.com/OpenGVLab/InternVideo.git "${IV2_REPO}"
else
    echo "==> upstream already cloned at ${IV2_REPO}"
fi

# ───────────── step 2: drop BLiM scripts onto upstream ─────────────
echo "==> Merging BLiM scripts into upstream tree"
cp -r "${KIT_DIR}/multi_modality/scripts/." "${IV2_MM}/scripts/"
chmod +x \
    "${IV2_MM}/scripts/finetuning/stage2/6B/didemo_blim/run.sh" \
    "${IV2_MM}/scripts/finetuning/stage2/6B/didemo_blim/smoke.sh" \
    "${IV2_MM}/scripts/evaluation/clip/zero_shot/6B/didemo_blim/eval.sh" 2>/dev/null || true

# ───────────── step 3: filtered DiDeMo JSONs ─────────────
mkdir -p "${BLIM_ANNO_DIR}"
if [[ "${BLIM_ANNO_DIR}" != "${KIT_DIR}/data/DiDeMo" ]]; then
    echo "==> Copying filtered JSONs to ${BLIM_ANNO_DIR}"
    cp -n "${KIT_DIR}/data/DiDeMo/"*.json "${BLIM_ANNO_DIR}/"
else
    echo "==> Filtered JSONs already at ${BLIM_ANNO_DIR}"
fi

# Sanity check: video dir
n_videos=$(ls "${BLIM_VIDEO_DIR}" 2>/dev/null | wc -l)
echo "==> BLIM_VIDEO_DIR contains ${n_videos} entries (expect ~10629)"
if [[ "${n_videos}" -lt 9000 ]]; then
    echo "WARN: too few videos in ${BLIM_VIDEO_DIR}; finetune will fail on missing clips" >&2
fi

# ───────────── step 4: conda env ─────────────
# Source conda so `conda activate` works in non-interactive shell.
# conda's activate.d scripts touch unbound vars (e.g. SYS_SYSROOT) so relax `set -u`.
set +u
CONDA_BASE="$(conda info --base 2>/dev/null || echo /usr/local)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV_NAME}"; then
    echo "==> conda env '${CONDA_ENV_NAME}' already exists — skipping create"
else
    echo "==> Creating conda env '${CONDA_ENV_NAME}' (python 3.10)"
    conda create -y -n "${CONDA_ENV_NAME}" python=3.10
fi
conda activate "${CONDA_ENV_NAME}"
set -u

echo "==> Installing torch 2.5.1 + cu121 (pip wheels — conda often picks CPU build)"
pip install --no-input torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

echo "==> Installing CUDA 12.1 toolkit (nvcc + dev libs) for DeepSpeed JIT"
conda install -y -c "nvidia/label/cuda-12.1.1" \
    cuda-nvcc cuda-cudart-dev cuda-libraries-dev

echo "==> Installing pyav via conda-forge (needs ffmpeg/pkg-config)"
conda install -y -c conda-forge "av=11.0.0"

echo "==> Installing python deps"
# setuptools<81 keeps pkg_resources available (wandb still imports it).
pip install --no-input "setuptools<81"
pip install --no-input \
    transformers==4.48.3 \
    deepspeed==0.15.4 \
    timm==0.5.4 \
    open_clip_torch \
    easydict \
    scikit-learn \
    matplotlib \
    einops==0.7.0 \
    decord==0.6.0 \
    imageio==2.31.1 \
    opencv-python-headless==4.8.0.76 \
    pandas==2.0.3 \
    Pillow==10.0.0 \
    PyYAML==6.0.1 \
    scipy==1.13.0 \
    soundfile==0.12.1 \
    librosa==0.10.1 \
    termcolor==2.4.0 \
    fvcore==0.1.5.post20221221 \
    psutil==5.9.5 \
    tqdm==4.66.1 \
    wandb==0.16.1 \
    peft \
    huggingface_hub

echo "==> Installing flash-attn 2.7.4.post1 (prebuilt wheel for torch 2.5 + cu12 + cp310)"
FA_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
pip install --no-input "${FA_WHEEL}" || \
    echo "[warn] flash-attn install failed — IV2 code imports it unconditionally so this WILL break runtime"

# ───────────── step 5: HF checkpoints ─────────────
mkdir -p "${INTERNVIDEO2_MODEL_PATH}"
cd "${INTERNVIDEO2_MODEL_PATH}"

S2_PT="${INTERNVIDEO2_MODEL_PATH}/internvideo2-s2_6b-224p-f4_with_audio_encoder.pt"
BEATS_PT="${INTERNVIDEO2_MODEL_PATH}/BEATs_iter3_plus_AS2M.pt"

if [[ ! -f "${S2_PT}" ]]; then
    echo "==> Downloading IV2-6B Stage2 (+audio) ckpt (~12 GB)"
    huggingface-cli download OpenGVLab/InternVideo2-Stage2_6B-224p-f4 \
        internvideo2-s2_6b-224p-f4_with_audio_encoder.pt \
        --local-dir "${INTERNVIDEO2_MODEL_PATH}"
else
    echo "==> Stage2 ckpt already present"
fi

if [[ ! -f "${BEATS_PT}" ]]; then
    # The transfer-kit README points at OpenGVLab/InternVideo2-Stage2_1B-224p-f4
    # but that file isn't in the repo (404). Use the public Microsoft mirror.
    echo "==> Downloading BEATs ckpt (~360 MB) from lpepino/beats_ckpts mirror"
    huggingface-cli download lpepino/beats_ckpts \
        BEATs_iter3_plus_AS2M.pt \
        --local-dir "${INTERNVIDEO2_MODEL_PATH}"
else
    echo "==> BEATs ckpt already present"
fi

echo ""
echo "===================================================="
echo "  Setup complete."
echo "  Next: bash ${KIT_DIR}/train_iv2_6b_didemo.sh"
echo "===================================================="
