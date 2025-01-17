fe_model_path="weights/force_estimation_network/state-transformer-weightedmse"

python src/train_transformer.py --batch_size 32 \
     --lr 0.00001 \
     --num_epochs 100 \
     --force_runs 1 2 3 4 6 8 9 10 11 12 17 19 20 21 22 23 24 26 27 28 30 34 35 36 37 38 39 40 41 42 43 44 47 48 49\
     --out_dir $fe_model_path \
     --normalize_targets \
     --use_acceleration \
     --seq_length 10 \
     --state linear

# evaluate on test set (10%)
python src/evaluate.py --run 25 --weights $fe_model_path --model res_net --model_type transformer --use_acceleration --state linear
python src/evaluate.py --run 29 --weights $fe_model_path --model res_net --model_type transformer --use_acceleration --state linear
python src/evaluate.py --run 31 --weights $fe_model_path --model res_net --model_type transformer --use_acceleration --state linear
python src/evaluate.py --run 50 --weights $fe_model_path --model res_net --model_type transformer --use_acceleration --state linear