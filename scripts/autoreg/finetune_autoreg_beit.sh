module load libjpeg-turbo
OUTPUT_DIR='outs/autoreg_beit_large_patch16_no_cam_aug_0227'
export HUGGINGFACE_HUB_CACHE=./pretrained_ckpts

DATA_PATH='data_lists/co3d_train.txt'
DATA_VAL_PATH='data_lists/co3d_val.txt'
DATA_ROOT='/oscar/data/tserre/Shared/'


NCCL_DEBUG=INFO OMP_NUM_THREADS=1  python3 -m torch.distributed.run \
     --standalone \
     --nnodes=1 \
     --nproc_per_node=2 \
      main_pretrain_autoreg.py \
        --num_workers 16 \
        --data_root ${DATA_ROOT} \
        --data_path ${DATA_PATH} \
        --data_val_path ${DATA_VAL_PATH} \
        --mask_type autoregressive \
        --mask_ratio 0.875 \
        --model autoreg_beit_large_patch16 \
        --batch_size 16 \
        --lr 0.0005 \
        --no-binocular \
        --num_frames 8 \
        --weight_decay 0.01 \
        --sampling_rate 4 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 10 \
        --seed 84 \
        --save_ckpt_freq 2 \
        --epochs 20 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --opt_eps 1e-7 \
        --schedule_free \
        --no-normalize_target \
        --eval_co3d \
        --eval_co3d_every 1 \
        --eval_co3d_batch_size 128 \
        --eval_co3d_epochs 50\
        --use_cce \
        --num_classes 1000 \
        --attn_drop_rate 0 \
        --drop_path 0.1 \
        --categorical_camera \
        --feature_loss \
        --decoder_pos_embed 3d \
        --photo_trasnform
        #--not_pretrained \
        #--clip_grad 1 \
        # --decoder_cls \
        # --use_cls \
        #--camera_params \
        #--decoder_pos_embed 3d \
        # --timm_pool \
