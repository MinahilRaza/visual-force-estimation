fe_model_path="weights/force_estimation_network/transformer"
fe_model_state=vision

python train_transformer.py --batch_size 32 \
    --lr 0.00001 \
    --num_epochs 100 \
    --force_runs 1 2 3 4 6 8 \
    --no_force_runs 1 4 \
    --out_dir $fe_model_path \
    --normalize_targets \
    --use_acceleration \
    --seq_length 10 \
    --overfit

python evaluate.py --run 1 --weights $fe_model_path --model res_net --model_type transformer --use_acceleration --state $fe_model_state
python evaluate.py --run 2 --weights $fe_model_path --model res_net --model_type transformer --use_acceleration --state $fe_model_state
python evaluate.py --run 3 --weights $fe_model_path --model res_net --model_type transformer --use_acceleration --state $fe_model_state
