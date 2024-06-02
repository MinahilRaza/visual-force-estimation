fe_model_path="weights/force_estimation_network/transformer"

python train_transformer.py --batch_size 32 \
     --lr 0.00001 \
     --num_epochs 50 \
     --force_runs 1 2 3 4 6 8 9 10 11 12 13 14 15 17 18 19 20 \
     --no_force_runs 1 2 3 4 5 6 8 9 10 11 12 14 15 16 17 18 19 20 \
     --out_dir $fe_model_path \
     --normalize_targets \
     --use_acceleration \
     --seq_length 10 \
     --state linear

python evaluate.py --run 1 --weights $fe_model_path --model res_net --model_type transformer --use_acceleration --state linear
python evaluate.py --run 2 --weights $fe_model_path --model res_net --model_type transformer --use_acceleration --state linear
python evaluate.py --run 3 --weights $fe_model_path --model res_net --model_type transformer --use_acceleration --state linear
python evaluate.py --run 4 --weights $fe_model_path --model res_net --model_type transformer --use_acceleration --state linear
python evaluate.py --run 17 --weights $fe_model_path --model res_net --model_type transformer --use_acceleration --state linear
