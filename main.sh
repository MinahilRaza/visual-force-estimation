#!/usr/bin/bash

python train.py --batch_size 16 --lr 0.0001 --num_epochs 50 --force_runs 1 2 3 4 6 8 --no_force_runs 1 4 --lr_scheduler