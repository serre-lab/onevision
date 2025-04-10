module load libjpeg-turbo
OUTPUT_DIR='outs/autoreg_vit_small_patch16_cam_combined_data_6_frames_0204'
export HUGGINGFACE_HUB_CACHE=./pretrained_ckpts

DATA_PATH='data_lists/combined_train.txt'
DATA_VAL_PATH='data_lists/combined_val.txt'
DATA_ROOT='/cifs/data/tserre_lrs/projects/projects/prj_video_imagenet/PeRFception/data/'


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
        --model autoreg_vit_small_patch16 \
        --batch_size 64 \
        --lr 0.015 \
        --no-binocular \
        --num_frames 6 \
        --weight_decay 0.01 \
        --sampling_rate 4 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 10 \
        --seed 84 \
        --save_ckpt_freq 2 \
        --epochs 50 \
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
        --drop_path 0 \
        --categorical_camera \
        --feature_loss \
        --decoder_pos_embed 3d \
        --dataset co3d_mvimgnet
        #--clip_grad 1 \
        # --decoder_cls \
        # --use_cls \
        # --kl_div
        # --camera_params \
