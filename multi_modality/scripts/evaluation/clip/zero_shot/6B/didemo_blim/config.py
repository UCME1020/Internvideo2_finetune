# BLiM zero-shot fork of scripts/evaluation/clip/zero_shot/6B/config_didemo.py
# Differences:
#   * Hard-coded BLiM annotation + video paths (skips configs/data.py registry)
#   * flash_attn / fused_rmsnorm / fused_mlp = False (our env lacks the CUDA exts)
#   * use_checkpoint=True for vision_encoder (memory)
#   * num_workers reduced; batch_size_test set conservatively
#   * 8 frames (paper default for CLIP-6B-224p-f8)
#
# Launch: bash scripts/evaluation/clip/zero_shot/6B/didemo_blim/eval.sh
import os as __os
_MODEL_PATH = __os.environ.get(
    "INTERNVIDEO2_MODEL_PATH",
    "/data5/jyhong/BLiM/InternVideo2/multi_modality/pretrained",
)

# Pure-pytorch path (we lack fused CUDA extensions).
flag = False

# ========================= data ==========================
_BLIM_ANNO_DIR = __os.environ.get(
    "BLIM_ANNO_DIR", "/data5/jyhong/BLiM/data/DiDeMo"
)
_BLIM_VIDEO_DIR = __os.environ.get(
    "BLIM_VIDEO_DIR", "/data5/jyhong/BLiM/dataset/DiDeMo/DiDeMo"
)

# Use filtered annotations (1 broken video removed) — same as Stage2 BLiM run.
_DIDEMO_TEST = dict(
    anno_path=f"{_BLIM_ANNO_DIR}/didemo_ret_test_filtered.json",
    data_root=_BLIM_VIDEO_DIR,
    media_type="video",
    is_paragraph_retrieval=True,
    trimmed30=True,
    max_txt_l=64,
)

# CLIP evaluation: official configs set train_file = test for lazy load,
# only test_file is actually used in `evaluate=True` path.
train_file = _DIDEMO_TEST
test_file = dict(ret_test=_DIDEMO_TEST)
test_types = ["ret_test"]
num_workers = 6

stop_key = None
is_paragraph_retrieval = True
trimmed30 = True

# ========================= input ==========================
num_frames = 8
num_frames_test = 8
batch_size = 16          # train (not used in eval mode)
batch_size_test = 2      # eval batch — 6B vision + 7B LLaMA in bf16 ≈ 31 GB, leaving ~8 GB for activations on 40 GB
max_txt_l = 32

inputs = dict(
    image_res=224,
    video_input=dict(
        num_frames="${num_frames}",
        sample_type="rand",
        num_frames_test="${num_frames_test}",
        sample_type_test="middle",
        random_aug=False,
    ),
    max_txt_l=dict(image="${max_txt_l}", video="${max_txt_l}"),
    batch_size=dict(image="${batch_size}", video="${batch_size}"),
    batch_size_test=dict(image="${batch_size_test}", video="${batch_size_test}"),
)

# ========================= model ==========================
model = dict(
    model_cls="InternVideo2_CLIP",
    vision_encoder=dict(
        name="internvideo2_6B",
        in_chans=3,
        patch_size=14,
        img_size=224,
        qkv_bias=False,
        drop_path_rate=0.35,
        head_drop_path_rate=0.,
        embed_dim=3200,
        num_heads=25,
        mlp_ratio=4,
        init_values=0.1,
        qk_normalization=True,
        depth=48,
        use_flash_attn=flag,
        use_fused_rmsnorm=flag,
        use_fused_mlp=flag,
        fused_mlp_heuristic=1,
        drop_cls_token=False,
        attn_pool_num_heads=16,
        clip_embed_dim=768,
        layerscale_no_force_fp32=True,
        num_frames="${num_frames}",
        tubelet_size=1,
        sep_pos_embed=False,
        use_checkpoint=True,         # save memory for 48-layer 6B vision
        checkpoint_num=48,
    ),
    text_encoder=dict(
        use_flash_attn=flag,
        transformer_width=4096,
        llama_path=f"{_MODEL_PATH}/chinese_alpaca_lora_7b",
        use_lora=True,
    ),
    temp=1 / 100.0,
    temp_min=1 / 100.0,
    freeze_vision=True,
    open_vision_clip_projector=True,
    freeze_text=True,
    open_text_projection=False,
    open_text_lora=False,
    tokenizer_path=f"{_MODEL_PATH}/chinese_alpaca_lora_7b",
    vision_ckpt_path=f"{_MODEL_PATH}/InternVideo2_Stage2_6B.pth",
    load_vision_ckpt_from_internvideo2_stage2=True,
    text_ckpt_path=f"{_MODEL_PATH}/internvl/internvl_c_13b_224px.pth",
    extra_ckpt_path=f"{_MODEL_PATH}/InternVideo2_CLIP_6B.pth",
)

criterion = dict(
    loss_weight=dict(vtc=1.0),  # eval-only; not used
)

optimizer = dict(
    opt="adamW",
    lr=4e-4,
    opt_betas=[0.9, 0.98],
    weight_decay=0.2,
    max_grad_norm=-1,
    different_lr=dict(enable=False, module_names=[], lr=1e-3),
)
scheduler = dict(sched="cosine", epochs=3, min_lr_multi=0.01, warmup_epochs=0.6)

# ── Zero-shot specific ─────────────────────────────────────────────────
evaluate = True
deep_fusion = False
evaluation = dict(
    eval_frame_ensemble="concat",  # [concat, max, mean, lse]
    eval_x_only=False,
    k_test=128,
    eval_offload=True,
)

use_half_precision = True
use_bf16 = True
gradient_checkpointing = True

# ========================= others ==========================
wandb = dict(enable=False, entity="opengvlab", project="InternVideo2_CLIP")
dist_url = "env://"
device = "cuda"
mode = "pt"

output_dir = None
resume = False
debug = False
log_freq = 1
seed = 42

save_latest = False
save_iter = 500
auto_resume = False
pretrained_path = ""

deepspeed = dict(enable=False, stage=0)
