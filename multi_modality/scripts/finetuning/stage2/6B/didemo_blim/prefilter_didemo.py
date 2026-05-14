"""Drop DiDeMo entries whose video file is unreadable by decord.

Some DiDeMo clips were saved truncated (<200KB). decord raises a hard
RuntimeError on these and the official base_dataset.py re-raises, killing
the whole job. We scan upfront, write filtered JSONs, and point config.py
at them.

Run from the multi_modality dir:
    python scripts/finetuning/stage2/6B/didemo_blim/prefilter_didemo.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from decord import VideoReader
from tqdm import tqdm

ANNO_DIR = Path(os.environ.get(
    "BLIM_ANNO_DIR", "/data5/jyhong/BLiM/data/DiDeMo"
))
VIDEO_DIR = Path(os.environ.get(
    "BLIM_VIDEO_DIR", "/data5/jyhong/BLiM/dataset/DiDeMo/DiDeMo"
))
SPLITS = ("train", "test")


def main():
    bad_videos: set[str] = set()
    for split in SPLITS:
        src = ANNO_DIR / f"didemo_ret_{split}.json"
        dst = ANNO_DIR / f"didemo_ret_{split}_filtered.json"
        with open(src) as f:
            data = json.load(f)
        kept = []
        skipped = 0
        for ann in tqdm(data, desc=f"verifying {split}"):
            vp = str(VIDEO_DIR / ann["video"])
            if vp in bad_videos:
                skipped += 1
                continue
            try:
                vr = VideoReader(vp, num_threads=1)
                if len(vr) == 0:
                    raise RuntimeError("0 frames")
            except Exception as e:  # noqa: BLE001
                bad_videos.add(vp)
                skipped += 1
                continue
            kept.append(ann)
        print(f"[{split}] kept {len(kept)} / {len(data)}  skipped {skipped}")
        with open(dst, "w") as f:
            json.dump(kept, f)
        print(f"[{split}] wrote {dst}")
    print(f"\nUnique broken videos: {len(bad_videos)}")
    if bad_videos:
        bad_list = ANNO_DIR / "didemo_broken_videos.txt"
        with open(bad_list, "w") as f:
            f.write("\n".join(sorted(bad_videos)))
        print(f"Listed at {bad_list}")


if __name__ == "__main__":
    main()
