#!/usr/bin/bash

# python train.py --batch_size 8 --lr 0.00001 --model res_net --num_epochs 50 --force_runs 1 2 3 4 6 8 --no_force_runs 1 4

python train.py --batch_size 8 --lr 0.0005 --model efficientnet_v2_m --num_epochs 30 --force_runs 1 2 3 4 6 8 --no_force_runs 1 4