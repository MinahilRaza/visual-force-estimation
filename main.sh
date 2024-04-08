#!/usr/bin/bash

TRAIN_AUTOENCODER=false
TRAIN_FORCE_ESTIMATION=true

model_path="weights/auto_encoder/resnet50/"

if $TRAIN_AUTOENCODER; then
    python train_auto_encoder.py --batch_size 4 \
        --num_epochs 10 \
        --lr 0.00001 \
        --base_model resnet50 \
        --out_dir $model_path
fi

if $TRAIN_FORCE_ESTIMATION; then
    python train.py --batch_size 8 \
        --lr 0.00001 \
        --model res_net \
        --num_epochs 30 \
        --force_runs 1 2 3 4 6 8 \
        --no_force_runs 1 4 \
        --use_acceleration
fi

# python train.py --batch_size 8 --lr 0.0005 --model efficientnet_v2_m --num_epochs 30 --force_runs 1 2 3 4 6 8 --no_force_runs 1 4
