# IV2-6B MSRVTT finetune (BLiM-style).
import os as __os
_MODEL_PATH = __os.environ.get(
    "INTERNVIDEO2_MODEL_PATH",
    "/data5/ucjung/PoLaRT/pretrained/iv2_6b",
)

from configs.model import *  # noqa: F401,F403

# ========================= data ==========================
_ANNO_DIR = __os.environ.get(
    "BLIM_MSRVTT_ANNO_DIR",
    "/data5/ucjung/PoLaRT/Internvideo2_finetune/data/MSRVTT",
)
_VIDEO_DIR = __os.environ.get(
    "BLIM_MSRVTT_VIDEO_DIR",
    "/data1/dataset/MSRVTT/videos/all",
)

train_file = dict(
    anno_path=f"{_ANNO_DIR}/msrvtt_ret_train.json",
    data_root=_VIDEO_DIR,
    media_type="video",
)

test_file = dict(msrvtt_1k_test=dict(
    anno_path=f"{_ANNO_DIR}/msrvtt_ret_test.json",
    data_root=_VIDEO_DIR,
    media_type="video",
))

test_types = ["msrvtt_1k_test"]
num_workers = 6

best_key = ["msrvtt_1k_test_match", "t2v_r1"]

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
        pretrained=None,
        use_checkpoint=True,
        checkpoint_num=48,
        use_flash_attn=flag,
        use_fused_rmsnorm=flag,
        use_fused_mlp=flag,
        clip_teacher=None,
        clip_input_resolution=224,
        clip_teacher_return_interval=1,
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
zero_shot = False
deep_fusion = False
evaluation = dict(
    eval_frame_ensemble="concat",
    eval_x_only=False,
    k_test=128,
    eval_offload=True,
)

use_half_precision = True
use_bf16 = True

gradient_checkpointing = True
use_flash_sdp = False
use_mem_efficient_sdp = False and not use_flash_sdp
compile_model = False

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
    stage=2,
)
