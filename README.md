# InternVideo2-Stage2-6B DiDeMo finetune — transfer kit

Standalone bundle of the BLiM-modified scripts needed to fine-tune (and
zero-shot-evaluate) **InternVideo2-Stage2-6B** on **DiDeMo** retrieval.

Contents:

```
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

## Quick start (new server)

```bash
# 1) Clone this transfer repo
git clone https://github.com/UCME1020/Internvideo2_finetune.git iv2_blim
cd iv2_blim

# 2) Clone upstream InternVideo2 codebase
git clone --depth=1 https://github.com/OpenGVLab/InternVideo.git _tmp_iv2
mv _tmp_iv2/InternVideo2 ./InternVideo2_upstream
rm -rf _tmp_iv2

# 3) Drop our scripts into the upstream tree (paths match exactly)
cp -r multi_modality/scripts/. InternVideo2_upstream/multi_modality/scripts/

# 4) Place the filtered annotations
#    config.py defaults to /data5/jyhong/BLiM/data/DiDeMo — either replicate that
#    layout, or override with env vars (BLIM_ANNO_DIR / BLIM_VIDEO_DIR).
mkdir -p /data5/jyhong/BLiM/data/DiDeMo
cp data/DiDeMo/*.json /data5/jyhong/BLiM/data/DiDeMo/

# 5) Transfer DiDeMo videos separately (~61 GB, not in this repo).
#    See: multi_modality/scripts/finetuning/stage2/6B/didemo_blim/NEW_SERVER_SETUP.md

# 6) Download HF checkpoints + set up conda env + smoke test + full run
#    Detailed steps: see NEW_SERVER_SETUP.md inside didemo_blim/.
```

Detailed setup (env, checkpoints, video data, smoke, full run) lives in
[`multi_modality/scripts/finetuning/stage2/6B/didemo_blim/NEW_SERVER_SETUP.md`](multi_modality/scripts/finetuning/stage2/6B/didemo_blim/NEW_SERVER_SETUP.md).
