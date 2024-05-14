#!/usr/bin/bash

TRAIN_AUTOENCODER=false
TRAIN_FORCE_ESTIMATION=true

vae_model_path="weights/auto_encoder/resnet50/"
# fe_model_path="weights/force_estimation_network/overfitting_2/resnet50"
fe_model_path="weights/force_estimation_network/resnet50"
fe_model_state=robot

if $TRAIN_AUTOENCODER; then
    python train_auto_encoder.py --batch_size 4 \
        --num_epochs 10 \
        --lr 0.000001 \
        --base_model resnet50 \
        --out_dir $vae_model_path
fi

if $TRAIN_FORCE_ESTIMATION; then
    python train.py --batch_size 32 \
        --lr 0.00001 \
        --model res_net \
        --num_epochs 200 \
        --force_runs 1 \
        --out_dir $fe_model_path \
        --normalize_targets \
        --use_acceleration \
        --state $fe_model_state \
        --overfit

    python evaluate.py --run 1 --weights $fe_model_path --model res_net --use_acceleration --overfit --state $fe_model_state
fi
