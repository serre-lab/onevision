#!/bin/sh
OUTPUT_DIR='outs/beit/videomae_beit_large_patch16_lr001'
export HUGGINGFACE_HUB_CACHE=./pretrained_ckpts

DATA_PATH='data_lists/co3d_train_gcp.txt'
DATA_VAL_PATH='data_lists/co3d_val_gcp.txt'
DATA_ROOT='/mnt/disks/cvm-data/'


NCCL_DEBUG=INFO OMP_NUM_THREADS=1 /opt/conda/bin/python -m torch.distributed.run \
     --standalone \
     --nnodes=1 \
     --nproc_per_node=2 \
      main_pretrain_videomae.py \
        --num_workers 20 \
        --data_root ${DATA_ROOT} \
        --data_path ${DATA_PATH} \
        --data_val_path ${DATA_VAL_PATH} \
        --mask_type tube \
        --mask_ratio 0.875 \
        --model videomae_beit_large_patch16 \
        --batch_size 12 \
        --lr 0.001 \
        --no-binocular \
        --num_frames 8 \
        --weight_decay 0.01 \
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
        --drop_path 0.1 \
        --categorical_camera \
        --feature_loss \
        --decoder_pos_embed 1d_spatial \
        --save_to_bucket \
        --single_video
