#!/bin/sh
OUTPUT_DIR='outs/beit/autoreg_dino_beit_large_patch16_lr001_aug'
export HUGGINGFACE_HUB_CACHE=./pretrained_ckpts

DATA_PATH='data_lists/co3d_train_gcp.txt'
DATA_VAL_PATH='data_lists/co3d_val_gcp.txt'
DATA_ROOT='/mnt/disks/cvm-data/'

NCCL_DEBUG=INFO OMP_NUM_THREADS=1 /opt/conda/bin/python -m torch.distributed.run \
     --standalone \
     --nnodes=1 \
     --nproc_per_node=2 \
      main_pretrain_dino.py \
        --num_workers 20 \
        --data_root ${DATA_ROOT} \
        --data_path ${DATA_PATH} \
        --data_val_path ${DATA_VAL_PATH} \
        --mask_type autoregressive \
        --mask_ratio 0.875 \
        --model autoreg_dino_beit_large_patch16 \
        --batch_size 8 \
        --lr 0.001 \
        --no-binocular \
        --num_frames 8 \
        --weight_decay 0.0001 \
        --sampling_rate 4 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 20 \
        --seed 84 \
        --save_ckpt_freq 10 \
        --epochs 300 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --opt_eps 1e-7 \
        --schedule_free \
        --no-normalize_target \
        --eval_co3d \
        --eval_co3d_every 1 \
        --eval_co3d_batch_size 256 \
        --eval_co3d_epochs 50\
        --use_cce \
        --num_classes 1000 \
        --attn_drop_rate 0 \
        --categorical_camera \
        --camera_params \
        --drop_path 0.1 \
        --out_dim 65536 \
        --pos_embed rel_3d \
        --timm_pool \
        --photo_transform \
        --shared_transform \
        --save_to_bucket \
        --single_video