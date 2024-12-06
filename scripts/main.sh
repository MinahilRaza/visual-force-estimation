#!/usr/bin/bash

TRAIN_AUTOENCODER=false
TRAIN_FORCE_ESTIMATION=true
USE_TRANSFORMER=false

vae_model_path="weights/auto_encoder/resnet50/"
fe_model_path="weights/force_estimation_network"
fe_model_state=robot

if $TRAIN_AUTOENCODER; then
    python train_auto_encoder.py --batch_size 4 \
        --num_epochs 10 \
        --lr 0.000001 \
        --base_model resnet50 \
        --out_dir $vae_model_path
fi

if $TRAIN_FORCE_ESTIMATION; then
    if $USE_TRANSFORMER; then
        transformer_model_path="${fe_model_path}/transformer"

        python train_transformer.py --batch_size 32 \
            --lr 0.00002 \
            --num_epochs 1 \
            --force_runs 1 \
            --out_dir $transformer_model_path \
            --normalize_targets \
            --use_acceleration \
            --seq_length 10 \
            --overfit

        python evaluate.py --run 9 --weights $transformer_model_path --model res_net --model_type transformer --use_acceleration --overfit --state $fe_model_state
        python evaluate.py --run 10 --weights $transformer_model_path --model res_net --model_type transformer --use_acceleration --overfit --state $fe_model_state
        python evaluate.py --run 11 --weights $transformer_model_path --model res_net --model_type transformer --use_acceleration --overfit --state $fe_model_state
    else
        normal_model_path="${fe_model_path}/feed_forward"

        python train.py --batch_size 32 \
            --lr 0.00002 \
            --model res_net \
            --num_epochs 1 \
            --force_runs 1 \
            --out_dir $normal_model_path \
            --normalize_targets \
            --use_acceleration \
            --state $fe_model_state

        python evaluate.py --run 9 --weights $normal_model_path --model res_net --model_type vision_robot --use_acceleration --overfit --state $fe_model_state
        python evaluate.py --run 10 --weights $normal_model_path --model res_net --model_type vision_robot --use_acceleration --overfit --state $fe_model_state
        python evaluate.py --run 11 --weights $normal_model_path --model res_net --model_type vision_robot --use_acceleration --overfit --state $fe_model_state
    fi
fi
