# BLiM fork of scripts/finetuning/stage2/6B/didemo/config.py
# Changes from upstream:
#   1) Inline DiDeMo data paths (no env-var indirection through configs/data.py)
#      → uses our existing JSON annotations and video dir
#   2) vision_encoder.pretrained = None — full multimodal weights come from
#      top-level pretrained_path (the combined .pt). Avoids the prefix
#      mismatch problem (combined .pt keys are vision_encoder.* but the
#      vision-only loader expects bare keys).
#   3) audio_model_path points to the standalone BEATs ckpt so
#      build_audio_encoder() passes its assert. The combined .pt then
#      overwrites these weights with the real Stage2 audio_encoder.*.
#   4) flag = False → disables FusedMLP / DropoutAddRMSNorm / FlashAttn
#      paths. Our env has flash-attn 2.7 but not the older fused_dense /
#      rms_norm ops; pure-pytorch fallback is slower but bit-stable.
#   5) origin_num_frames = 4 → ckpt was trained with 4 frames, matches our
#      num_frames=4 → interpolate_pos_embed_internvideo2_new is a no-op.
#
# Launch: bash scripts/finetuning/stage2/6B/didemo_blim/run.sh
# NOTE: __prefixed names are filtered out by utils/config.py's loader
# (`not name.startswith("__")`), so we double-underscore module imports and
# private constants we don't want to appear in the JSON dump.
import os as __os
_MODEL_PATH = __os.environ.get(
    "INTERNVIDEO2_MODEL_PATH",
    "/data5/jyhong/BLiM/InternVideo2/multi_modality/pretrained",
)

from configs.model import *  # noqa: F401,F403 — provides TextEncoders, etc.

# ========================= data ==========================
# Override on a new server via env vars:
#   export BLIM_ANNO_DIR=/path/to/data/DiDeMo
#   export BLIM_VIDEO_DIR=/path/to/dataset/DiDeMo/DiDeMo
_BLIM_ANNO_DIR = __os.environ.get("BLIM_ANNO_DIR", "/data5/jyhong/BLiM/data/DiDeMo")
_BLIM_VIDEO_DIR = __os.environ.get("BLIM_VIDEO_DIR", "/data5/jyhong/BLiM/dataset/DiDeMo/DiDeMo")

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

best_key = ["didemo_ret_test_match", "t2v_r1"]

# ========================= input ==========================
num_frames = 4
num_frames_test = 4
batch_size = 8
batch_size_test = 8
max_txt_l = 40

inputs = dict(
    image_res=224,
    audio_input=dict(
        audio_sample_rate=16000,
        has_multi_audio_gt=False,
        audio_reader_type='torchaudio',
        max_audio_length=10
    ),
    video_input=dict(
        num_frames="${num_frames}",
        sample_type="rand",
        num_frames_test="${num_frames_test}",
        sample_type_test="middle",
        random_aug=False,
    ),
    max_txt_l=dict(image="${max_txt_l}", audio="${max_txt_l}", video="${max_txt_l}", audio_video="${max_txt_l}"),
    batch_size=dict(image="${batch_size}", audio="${batch_size}", video="${batch_size}", audio_video="${batch_size}"),
    batch_size_test=dict(image="${batch_size_test}", audio="${batch_size_test}", video="${batch_size_test}", audio_video="${batch_size_test}"),
)

# Disable fused ops; pure-pytorch path. Flash-attn 2.7 lacks the legacy
# fused_dense / rms_norm CUDA extensions this code expects.
flag = False

# ========================= model ==========================
text_enc = "bert_large"
model = dict(
    model_cls="InternVideo2_Stage2_audiovisual",
    audio_encoder=dict(
        name='beats',
        d_model=768,
        audio_model_path=f"{_MODEL_PATH}/BEATs_iter3_plus_AS2M.pt",
    ),
    vision_encoder=dict(
        # backbone
        name="pretrain_internvideo2_6b_patch14_224",
        img_size=224,
        num_frames="${num_frames}",
        tubelet_size=1,
        patch_size=14,
        d_model=3200,
        clip_embed_dim=768,
        clip_teacher_embed_dim=3200,
        clip_teacher_final_dim=768,
        clip_norm_type='l2',
        clip_return_layer=6,
        clip_student_return_interval=1,
        # Skip the vision-only load — top-level pretrained_path handles it.
        pretrained=None,
        use_checkpoint=True,
        checkpoint_num=48,
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
        sep_image_video_pos_embed=False,
        keep_temporal=False,
        only_mask=True
    ),
    text_encoder="${TextEncoders[${text_enc}]}",
    multimodal=dict(enable=True),
    contra_dim=768,
    av_concat_dim=768,
    temp=0.07,
    find_unused_parameters=False,
    freeze_vision=False,
    freeze_audio=True
)

criterion = dict(
    loss_weight=dict(
        vtc=1.0,
        mlm=0.0,
        vtm=1.0,
        uta=0.0,
        atc=0.0, avc=0.0, avtc=0.0,
        atm=0.0, avtm=0.0,
        amlm=0.0, avmlm=0.0
    ),
    loss_caption=dict(
        vtc='avs_captions', vtm='avs_captions', mlm='avs_captions',
        avtc='avs_captions', avtm='avs_captions', avmlm='avs_captions',
    ),
    vtm_hard_neg=True,
    mlm_masking_prob=0.5,
    distill_final_features=True,
    clip_loss_ratio=[1., 1.],
    uta_image_only=True
)

optimizer = dict(
    opt="adamW",
    lr=5e-6,
    opt_betas=[0.9, 0.98],
    weight_decay=0.05,
    max_grad_norm=3.,
    different_lr=dict(enable=False, module_names=[], lr=1e-3),
)

scheduler = dict(sched="cosine", epochs=5, min_lr_multi=0.01, warmup_epochs=1)

evaluate = False
deep_fusion = False
evaluation = dict(
    eval_frame_ensemble="concat",
    eval_x_only=False,
    k_test=128,
    eval_offload=True,
)

use_half_precision = True
use_bf16 = True

gradient_checkpointing = True  # for text encoder
use_flash_sdp = False
use_mem_efficient_sdp = False and not use_flash_sdp
compile_model = False

# Pretrained ckpt was trained with 4 frames; we also use 4 → no-op interp,
# but the path needs to run to register the source frame count.
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

save_latest = True
auto_resume = False
jump_evaluate = False
pretrained_path = ""  # set by run.sh
save_ckpt_iter = None
delete_ds_optim_states = True

deepspeed = dict(
    enable=True,
    stage=2,
)
