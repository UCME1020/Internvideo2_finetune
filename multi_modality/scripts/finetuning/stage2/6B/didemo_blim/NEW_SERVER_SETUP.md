# InternVideo2-Stage2-6B DiDeMo — new-server setup (standalone transfer kit)

This guide is written for a **fresh server** (e.g. "130") cloning this
transfer repo and reproducing the BLiM-modified IV2-6B DiDeMo
fine-tune and/or zero-shot eval.

## Briefing for Claude on the new server

You are running on a server that does **not** have BLiM's prior layout.
Hardcoded `/data5/jyhong/BLiM/...` paths in the configs are *defaults*
only — every path is overridable via env vars below. Pick paths that
exist on this server (e.g. `/workspace/`, `/home/jyhong/`) and export
the env vars before launching `run.sh` / `smoke.sh` / `eval.sh`.

Required env vars (export once per shell session):

```bash
export INTERNVIDEO2_MODEL_PATH=<dir that will hold downloaded *.pth / *.pt>
export BLIM_ANNO_DIR=<dir that will hold didemo_ret_*_filtered.json>
export BLIM_VIDEO_DIR=<dir holding DiDeMo *.mp4 videos>
```

If you leave them unset, the configs fall back to the original
`/data5/jyhong/BLiM/...` layout (which will fail on a fresh server).

## 0. Clone this transfer repo

```bash
git clone https://github.com/UCME1020/Internvideo2_finetune.git iv2_blim
cd iv2_blim
```

Files you start with:

```
iv2_blim/
├── README.md
├── multi_modality/scripts/         # BLiM-modified configs + launchers
└── data/DiDeMo/                    # filtered annotations (1002 entries)
```

## 1. Clone upstream InternVideo2 codebase

The upstream code (models, dataloader, tasks/pretrain.py, configs/model.py
etc.) is NOT in this repo. Pull it from OpenGVLab and merge our scripts
on top:

```bash
git clone --depth=1 https://github.com/OpenGVLab/InternVideo.git _tmp_iv2
mv _tmp_iv2/InternVideo2 ./InternVideo2_upstream
rm -rf _tmp_iv2

# Drop our scripts into upstream tree (paths line up exactly)
cp -r multi_modality/scripts/. InternVideo2_upstream/multi_modality/scripts/

# Place the filtered DiDeMo annotations where BLIM_ANNO_DIR will point
mkdir -p "${BLIM_ANNO_DIR}"
cp data/DiDeMo/*.json "${BLIM_ANNO_DIR}/"
```

## 2. Conda env

```bash
cd InternVideo2_upstream/multi_modality
conda env create -f vl.yml      # creates env "vl" with most deps
conda activate vl
pip install deepspeed
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

**Do NOT** install `dropout_layer_norm` or `fused_dense` C++ extensions —
our configs set `flag = False` to avoid them (pure-pytorch fallback).
Building them on a fresh machine is fragile and unnecessary.

## 3. Download checkpoints

The 6B finetune and 6B zero-shot need **different** sets of files. Pick
the one matching the task you'll run.

```bash
huggingface-cli login    # only if you haven't
mkdir -p "${INTERNVIDEO2_MODEL_PATH}"
cd "${INTERNVIDEO2_MODEL_PATH}"
```

### 3a. For finetune (`run.sh` / `smoke.sh`)

```bash
# Combined Stage2 ckpt (vision + audio_encoder bundled, ~12 GB)
huggingface-cli download OpenGVLab/InternVideo2-Stage2_6B-224p-f4 \
    internvideo2-s2_6b-224p-f4_with_audio_encoder.pt \
    --local-dir .

# Standalone BEATs ckpt (~600 MB), only needed to satisfy a constructor
# assert; weights are overwritten by the combined ckpt above.
huggingface-cli download OpenGVLab/InternVideo2-Stage2_1B-224p-f4 \
    BEATs_iter3_plus_AS2M.pt \
    --local-dir .
```

### 3b. For zero-shot eval (`evaluation/clip/zero_shot/6B/didemo_blim*/eval.sh`)

```bash
# InternVideo2 vision-only ckpt
huggingface-cli download OpenGVLab/InternVideo2-Stage2_6B-224p-f4 \
    InternVideo2_Stage2_6B.pth --local-dir .

# CLIP-style text projector ckpt
huggingface-cli download OpenGVLab/InternVideo2-CLIP-6B-224p-f8 \
    InternVideo2_CLIP_6B.pth --local-dir .

# InternVL-C 13B text encoder (~13 GB)
mkdir -p internvl
huggingface-cli download OpenGVLab/InternVL \
    internvl_c_13b_224px.pth --local-dir internvl

# Chinese-Alpaca-LoRA-7B text encoder (~14 GB, a directory)
huggingface-cli download ziqingyang/chinese-alpaca-lora-7b \
    --local-dir chinese_alpaca_lora_7b
```

(If a repo ID is gated, check the OpenGVLab HF collection for the
current canonical IDs — they occasionally rename.)

## 4. DiDeMo videos (~61 GB, not in this repo)

You need ~10k `.mp4` files under `${BLIM_VIDEO_DIR}`. Either:

- **rsync from the working server** (fastest if you have ssh access):
  ```bash
  rsync -av --progress \
      <user>@<working-server>:/data5/jyhong/BLiM/dataset/DiDeMo/DiDeMo/ \
      "${BLIM_VIDEO_DIR}/"
  ```
- **Re-download from DiDeMo official** (slower, see CLIP4Clip / VINDLU
  release notes for the canonical YouTube URL list).

Optional sanity check: `ls "${BLIM_VIDEO_DIR}" | wc -l` should be ~10k.

The filtered annotation JSONs ship with this repo at `data/DiDeMo/*.json`
(step 1) — no regeneration step needed.

## 5. Smoke test (1 GPU, eval-only, ~10 min)

```bash
cd InternVideo2_upstream/multi_modality
NUM_GPUS=1 CUDA_VISIBLE_DEVICES=0 \
    bash scripts/finetuning/stage2/6B/didemo_blim/smoke.sh
```

Success criteria:
- `smoke_TS/train.log` ends with eval metrics, no traceback
- `smoke_TS/eval_res_latest.json` produced
- expect `t2v_r1` ≈ 0.59 (paper zero-shot baseline) — the smoke uses
  the same ckpt as finetune step 0, so result equals zero-shot.

## 6. Full finetune (4 GPUs, ~6h on 8×A100-40GB)

```bash
cd InternVideo2_upstream/multi_modality
NUM_GPUS=4 bash scripts/finetuning/stage2/6B/didemo_blim/run.sh
```

Expected: T2V R@1 (match score) ~68-69 after 5 epochs (paper: ~67-68).

If you have fewer GPUs:
- 4×A100: change `deepspeed.stage=2` → `stage=3` in config.py
- Less than 4: would need much smaller batch, results may degrade

## 7. Zero-shot only (4 GPUs, ~30 min)

```bash
cd InternVideo2_upstream/multi_modality
NUM_GPUS=4 bash scripts/evaluation/clip/zero_shot/6B/didemo_blim/eval.sh
# or for the "iso" variant (CLIP-style isolated vision, flash_attn ON):
NUM_GPUS=4 bash scripts/evaluation/clip/zero_shot/6B/didemo_blim_iso/eval.sh
```

Expected zero-shot t2v_r1 ≈ 59.

## Common gotchas

1. **`Module not found: dropout_layer_norm`**: ignore — config has
   `flag = False` to avoid it. If you see an assert/NameError about
   `DropoutAddRMSNorm` actually being called, double-check that
   `use_fused_rmsnorm = False` in your config (it should be).
2. **`Couldn't load ckpt at vision_ckpt_path`**: zero-shot needs the
   *separate* `InternVideo2_Stage2_6B.pth` from step 3b — NOT the
   combined `..._with_audio_encoder.pt` from step 3a.
3. **decord RuntimeError on video N/N**: a broken DiDeMo clip slipped
   past the shipped filter. Re-download the clip from Flickr (the
   filename encodes `{user_id}_{photo_id}_{secret}`) and re-verify with
   decord, or drop the entry from the filtered JSON.
4. **OOM during eval**: lower `batch_size_test` in config.py
   (currently 2 for 6B + 7B-text fits 40 GB tight). For 80 GB cards,
   try 8.

## Verifying repro

After full finetune, the run output dir should contain:

```
run.sh_TS/
├── train.log
├── config.json
├── ckpt_best.pth
└── eval_res_best.json   ← contains didemo_ret_test_match.t2v_r1 ≈ 68.x
```

Quote that number back when reporting.
