#!/usr/bin/bash

TRAIN_AUTOENCODER=true
TRAIN_FORCE_ESTIMATION=true

model_path="weights/auto_encoder/resnet50/"

if $TRAIN_AUTOENCODER; then
    python train_auto_encoder.py --batch_size 4 \
        --num_epochs 5 \
        --lr 0.0001 \
        --base_model resnet50 \
        --out_dir $model_path
fi

if $TRAIN_FORCE_ESTIMATION; then
    python train.py --batch_size 8 \
        --lr 0.00001 \
        --model $model_path \
        --num_epochs 30 \
        --force_runs 1 2 3 4 6 8 \
        --no_force_runs 1 4
fi

# python train.py --batch_size 8 --lr 0.0005 --model efficientnet_v2_m --num_epochs 30 --force_runs 1 2 3 4 6 8 --no_force_runs 1 4
