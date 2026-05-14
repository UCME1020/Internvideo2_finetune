# InternVideo2-Stage2-6B DiDeMo finetune — new server setup

Reproduce the BLiM-mirrored DiDeMo finetune on a fresh machine.

## 0. Clone BLiM

```bash
git clone https://github.com/mlvlab/BLiM.git
cd BLiM
```

## 1. Clone InternVideo2 codebase

`InternVideo2/` is gitignored — clone fresh from OpenGVLab:

```bash
git clone https://github.com/OpenGVLab/InternVideo.git _tmp_iv2
mkdir -p InternVideo2
cp -r _tmp_iv2/InternVideo2/multi_modality InternVideo2/
rm -rf _tmp_iv2
```

(Or just `mv _tmp_iv2/InternVideo2 .` if you want the whole repo.)

## 2. Download checkpoints (HuggingFace)

Need 2 files:
- `internvideo2-s2_6b-224p-f4_with_audio_encoder.pt` (~12 GB)
- `BEATs_iter3_plus_AS2M.pt` (~600 MB)

```bash
huggingface-cli login   # if not already
mkdir -p InternVideo2/multi_modality/pretrained
cd InternVideo2/multi_modality/pretrained

# IV2-Stage2-6B (gated — agree at https://huggingface.co/OpenGVLab/InternVideo2-Stage2_6B-224p-f4)
huggingface-cli download OpenGVLab/InternVideo2-Stage2_6B-224p-f4 \
    internvideo2-s2_6b-224p-f4_with_audio_encoder.pt \
    --local-dir .

# BEATs audio encoder
huggingface-cli download OpenGVLab/InternVideo2-Stage2_1B-224p-f4 \
    BEATs_iter3_plus_AS2M.pt \
    --local-dir .

cd ../../..
```

(If HF gated, check OpenGVLab's collection for the correct repo IDs.)

## 3. DiDeMo data

Need:
- Annotation: `data/DiDeMo/didemo_ret_train_filtered.json` + `didemo_ret_test_filtered.json` (in this commit)
- Videos: `dataset/DiDeMo/DiDeMo/*.mp4` (~61 GB) — same source as BLiM training

Annotation files are tracked; videos must be transferred separately
(rsync from working server, or re-download from DiDeMo official).

## 4. Environment

DeepSpeed + flash_attn required. The IV2 codebase ships `vl.yml` (conda env).

```bash
cd InternVideo2/multi_modality
conda env create -f vl.yml   # creates env "vl"
conda activate vl
# Add: deepspeed, flash-attn matching CUDA version
pip install deepspeed
pip install flash-attn --no-build-isolation
cd ../..
```

## 5. Point config to your paths

Either:

(a) **Match the original layout** — put data under `/data5/jyhong/BLiM/...` (works as-is)

(b) **Use env vars** — set before running:

```bash
export BLIM_ANNO_DIR=/your/path/data/DiDeMo
export BLIM_VIDEO_DIR=/your/path/dataset/DiDeMo/DiDeMo
export INTERNVIDEO2_MODEL_PATH=/your/path/InternVideo2/multi_modality/pretrained
```

## 6. Smoke test (5 steps, ~10 min)

```bash
cd InternVideo2/multi_modality
NUM_GPUS=8 bash scripts/finetuning/stage2/6B/didemo_blim/smoke.sh
```

Should finish without error and produce `smoke_TS/eval_res_latest.json`.

## 7. Full finetune (5 epochs, ~6 h on 8×A100)

```bash
cd InternVideo2/multi_modality
NUM_GPUS=8 bash scripts/finetuning/stage2/6B/didemo_blim/run.sh
```

Expected: T2V R@1 (match) ~68-69 after 5 epochs (paper finetune number is ~67-68 range).
Zero-shot baseline of same model is 59.08 (verified).

## Notes

- Annotation has 1002 entries (1 broken video filtered from BLiM's 1003).
  For exact-paper-match would need canonical 1004 from CLIP4Clip/VINDLU.
- Audio encoder is loaded but actually frozen + unused for DiDeMo retrieval
  (vtm/vtc only). The `_with_audio_encoder.pt` ckpt just bundles both.
- DeepSpeed Stage 2 is required for 6B + 8 GPU; lower GPU count needs Stage 3.
