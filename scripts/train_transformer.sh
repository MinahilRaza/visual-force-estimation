seq_length=10
loss_criterion="l1"
fe_model_path="force_estimation_network/state-transformer_seq_${seq_length}_${loss_criterion}"

python src/train_transformer.py --batch_size 32 \
     --lr 0.00001 \
     --num_epochs 150 \
     --force_runs 1 2 3 4 6 7 8 9 10 11 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 30 31 32 34 35 37 38 40 41 42 43 44 46 47 48 49 50\
     --out_dir $fe_model_path \
     --normalize_targets \
     --use_acceleration \
     --seq_length $seq_length \
     --state linear \
     --use_kfold \
     --k_folds 6 \
     --loss_criterion $loss_criterion