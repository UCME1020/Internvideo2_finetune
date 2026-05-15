# IV2-1B + DiDeMo finetune — reproduces the BLiM paper (arxiv 2507.23284)
# Table 1 "InternVideo2 1B*" recipe (T2V R@1=75.3 on DiDeMo).
# Source: BLiM/iv2_finetune/scripts/finetuning/stage2/1B/didemo/config.py
#
# Key recipe choices (vs our earlier 1B attempt that scored 69.26):
#   - num_frames = 8 (was 4) — paragraph retrieval benefits from more frames
#   - lr = 5e-6 (was 1e-5) — 1B is stable at the same lr as 6B
#   - criterion.mlm = 0.0 (was 1.0) — BLiM disables MLM for retrieval
#   - evaluation.use_dsl_for_match = True (was unset) — DSL-normalized rerank
#   - best_key uses "_match (use_dsl)" suffix (matches the DSL flag)
#   - max_txt_l = 64 (was 40) — paragraph caps are longer
#   - origin_num_frames = 4 → real pos_embed interpolation 4→8 (was no-op)
#
# Other 1B-specific bits (vs our 6B BLiM config):
#   - model_cls = InternVideo2_Stage2_visual (no audio encoder)
#   - vision_encoder = pretrain_internvideo2_1b_patch14_224 (d_model=1408)
#   - use_checkpoint = False, sep_image_video_pos_embed = True
#   - deepspeed.stage = 1 (1B + AdamW fits easily)
#   - embed_dim = 512
#
# Pretrained ckpt: InternVideo2-stage2_1b-224p-f4.pt (~2.8 GB, HF
# OpenGVLab/InternVideo2-Stage2_1B-224p-f4). Loaded via top-level
# `pretrained_path` (vision_encoder.pretrained=None, same trick as 6B).
#
# Launch: bash scripts/finetuning/stage2/1B/didemo_blim/run.sh
import os as __os
_MODEL_PATH = __os.environ.get(
    "INTERNVIDEO2_MODEL_PATH",
    "/data5/ucjung/PoLaRT/pretrained/iv2_1b",
)

from configs.model import *  # noqa: F401,F403 — provides TextEncoders, etc.

# ========================= data ==========================
_BLIM_ANNO_DIR = __os.environ.get(
    "BLIM_ANNO_DIR",
    "/data5/ucjung/PoLaRT/Internvideo2_finetune/data/DiDeMo",
)
_BLIM_VIDEO_DIR = __os.environ.get(
    "BLIM_VIDEO_DIR",
    "/data1/jyhong/workspace/ECCV2026/dataset/DiDeMo/DiDeMo",
)

train_file = dict(
    anno_path=f"{_BLIM_ANNO_DIR}/didemo_ret_train_filtered.json",
    data_root=_BLIM_VIDEO_DIR,
    media_type="video",
    is_paragraph_retrieval=True,
    trimmed30=True,
    max_txt_l=64,
)

test_file = dict(didemo_ret_test=dict(
    anno_path=f"{_BLIM_ANNO_DIR}/didemo_ret_test_filtered.json",
    data_root=_BLIM_VIDEO_DIR,
    media_type="video",
    is_paragraph_retrieval=True,
    trimmed30=True,
    max_txt_l=64,
))

test_types = ["didemo_ret_test"]
num_workers = 6

best_key = ["didemo_ret_test_match (use_dsl)", "t2v_r1"]

# ========================= input ==========================
# Match BLiM recipe: 8 frames train+test, max_txt_l=64 for paragraph retrieval.
num_frames = 8
num_frames_test = 8
batch_size = 16
batch_size_test = 4       # smaller eval batch — 8 frames = 2x tokens vs 4-frame
max_txt_l = 64

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

# Fused ops (flash-attn + fused_mlp + fused_rmsnorm) are now buildable in env.
flag = True

# ========================= model ==========================
text_enc = "bert_large"
model = dict(
    model_cls="InternVideo2_Stage2_visual",
    vision_encoder=dict(
        name="pretrain_internvideo2_1b_patch14_224",
        img_size=224,
        num_frames="${num_frames}",
        tubelet_size=1,
        patch_size=14,
        d_model=1408,
        clip_embed_dim=768,
        clip_teacher_embed_dim=3200,
        clip_teacher_final_dim=768,
        clip_norm_type='l2',
        clip_return_layer=6,
        clip_student_return_interval=1,
        # vision-only load skipped — top-level pretrained_path handles it
        pretrained=None,
        use_checkpoint=False,
        checkpoint_num=40,
        use_flash_attn=flag,
        use_fused_rmsnorm=flag,
        use_fused_mlp=flag,
        # clip teacher
        clip_teacher=None,
        clip_input_resolution=224,
        clip_teacher_return_interval=1,
        # mask
        video_mask_type="random",
        video_mask_ratio=0.8,
        image_mask_type="random",
        image_mask_ratio=0.5,
        sep_image_video_pos_embed=True,
        keep_temporal=False,
        only_mask=True,
    ),
    text_encoder="${TextEncoders[${text_enc}]}",
    multimodal=dict(enable=True),
    embed_dim=512,
    temp=0.07,
    find_unused_parameters=False,
)

criterion = dict(
    loss_weight=dict(
        vtc=1.0,
        mlm=0.0,  # BLiM 1B retrieval recipe disables MLM
        vtm=1.0,
        mvm=0.0,
        uta=0.0,
    ),
    vtm_hard_neg=True,
    mlm_masking_prob=0.5,
    distill_final_features=True,
    clip_loss_ratio=[1., 1.],
)

optimizer = dict(
    opt="adamW",
    lr=5e-6,                  # BLiM 1B finetune recipe
    opt_betas=[0.9, 0.98],
    weight_decay=0.05,
    max_grad_norm=3.,
    different_lr=dict(enable=False, module_names=[], lr=1e-3),
)

scheduler = dict(sched="cosine", epochs=5, min_lr_multi=0.01, warmup_epochs=1)

evaluate = False
zero_shot = False  # smoke.sh sets True to trigger text_encoder.bert.* → text_encoder.* remap
deep_fusion = False
evaluation = dict(
    eval_frame_ensemble="concat",
    eval_x_only=False,
    k_test=128,
    eval_offload=True,
    use_dsl_for_match=True,   # BLiM uses DSL-normalized cross-encoder rerank
)

use_half_precision = True
use_bf16 = True

gradient_checkpointing = True  # text encoder activation checkpointing
use_flash_sdp = False
use_mem_efficient_sdp = False and not use_flash_sdp
compile_model = False

# Pretrained ckpt has 4-frame pos_embed; we use num_frames=8 → real interp 4→8.
origin_num_frames = 4

# ========================= wandb ==========================
wandb = dict(
    enable=False,
    entity="opengvlab",
    project="InternVideo2-Stage2",
)
dist_url = "env://"
device = "cuda"
mode = "ret"

# ========================= others ==========================
output_dir = None
resume = False
debug = False
log_freq = 100
seed = 42

# Per-epoch heavy ckpt is disabled in tasks/pretrain.py — only best_model.pth
# (model-only, rank0) is saved after eval. Set save_deepspeed_resume=True
# to opt back in to occasional full DeepSpeed ckpts for crash recovery.
save_latest = False
save_ckpt_iter = None
delete_ds_optim_states = False  # no-op: per-epoch DS save is skipped
save_deepspeed_resume = False
save_deepspeed_every_n_epochs = 0  # only used if save_deepspeed_resume=True
delete_deepspeed_resume_on_finish = False

auto_resume = False
jump_evaluate = False
pretrained_path = ""  # set by run.sh

deepspeed = dict(
    enable=True,
    stage=1,  # 1B + AdamW: optim state ~12 GB, fits per-GPU; no need for stage 2
)
