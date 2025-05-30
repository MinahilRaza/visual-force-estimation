seq_length=10
loss_criterion="mse"
fe_model_path="force_estimation_network/transformer-15runs_seq_${seq_length}_${loss_criterion}"

python src/train_transformer.py --batch_size 32 \
     --lr 0.00001 \
     --num_epochs 150 \
     --force_runs 1 2 3 4 5 6 7 8 9 10 11 12\
     --out_dir $fe_model_path \
     --normalize_targets \
     --use_acceleration \
     --seq_length $seq_length \
     --state linear \
     --use_kfold \
     --k_folds 4 \
     --loss_criterion $loss_criterion