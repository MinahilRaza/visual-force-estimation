#!/usr/bin/bash

TRAIN_AUTOENCODER=true
TRAIN_FORCE_ESTIMATION=false

if $TRAIN_FORCE_ESTIMATION; then
    python train.py --batch_size 8 \
        --lr 0.00001 \
        --model weights/auto_encoder/run_2_res_net_epochs_30/best_params.pth \
        --num_epochs 50 \
        --force_runs 1 2 \
        --no_force_runs 1 4
fi

if $TRAIN_AUTOENCODER; then
    python train_auto_encoder.py --batch_size 4 --num_epochs 30 --lr 0.0001 --base_model "resnet18"
fi

# python train.py --batch_size 8 --lr 0.0005 --model efficientnet_v2_m --num_epochs 30 --force_runs 1 2 3 4 6 8 --no_force_runs 1 4
