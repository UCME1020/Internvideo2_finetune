# InternVideo2-Stage2-6B DiDeMo finetune — transfer kit

Standalone bundle of the BLiM-modified scripts needed to fine-tune (and
zero-shot-evaluate) **InternVideo2-Stage2-6B** on **DiDeMo** retrieval.

Contents:

```text
multi_modality/scripts/
├── finetuning/stage2/6B/didemo_blim/
│   ├── config.py             # BLiM fork of upstream didemo/config.py
│   ├── run.sh                # 8-GPU torchrun launcher
│   ├── smoke.sh              # 1-GPU eval-only end-to-end smoke
│   ├── prefilter_didemo.py   # drops unreadable clips, writes *_filtered.json
│   └── NEW_SERVER_SETUP.md   # full reproduce guide (read this!)
└── evaluation/clip/zero_shot/6B/
    ├── didemo_blim/          # zero-shot eval (BLIP-style fusion head)
    └── didemo_blim_iso/      # zero-shot eval (CLIP-iso, vision-only)

data/DiDeMo/
├── didemo_ret_train_filtered.json   # 1002 entries (1 broken video dropped)
└── didemo_ret_test_filtered.json
```

## Quick start

**Full step-by-step guide** lives in
[`multi_modality/scripts/finetuning/stage2/6B/didemo_blim/NEW_SERVER_SETUP.md`](multi_modality/scripts/finetuning/stage2/6B/didemo_blim/NEW_SERVER_SETUP.md)
— read that first. The README here only shows the high-level shape.

```bash
# 0) Choose paths on this server, then export env vars
export INTERNVIDEO2_MODEL_PATH=/path/for/pretrained
export BLIM_ANNO_DIR=/path/for/annotations
export BLIM_VIDEO_DIR=/path/for/didemo_videos

# 1) Clone this repo + upstream IV2 + drop scripts (see SETUP §0-1)
# 2) conda env (see SETUP §2)
# 3) Download HF checkpoints (see SETUP §3 — different sets for FT vs ZS!)
# 4) DiDeMo videos rsync/download (see SETUP §4, ~61 GB)
# 5) Smoke (see SETUP §5)         — bash .../didemo_blim/smoke.sh
# 6) Full finetune (see SETUP §6) — bash .../didemo_blim/run.sh
# 7) Zero-shot (see SETUP §7)     — bash .../zero_shot/6B/didemo_blim/eval.sh
```

## Notes for Claude / agent on the new server

- All config paths default to `/data5/jyhong/BLiM/...` (origin server)
  but are **env-var overridable**: `INTERNVIDEO2_MODEL_PATH`,
  `BLIM_ANNO_DIR`, `BLIM_VIDEO_DIR`. Set them before launching.
- Don't try to build `dropout_layer_norm` / `fused_dense` C++ extensions
  — configs intentionally bypass them via `flag = False`.
- Finetune and zero-shot need **different** HF checkpoint sets — see
  SETUP §3a vs §3b.
- Filtered annotations (1002 entries, 1 broken video dropped) are
  shipped in `data/DiDeMo/`; you do not need to run prefilter unless
  you want to re-derive from raw `didemo_ret_*.json`.
