fe_model_path="weights/force_estimation_network/state-transformer"

python src/train_transformer.py --batch_size 32 \
     --lr 0.00001 \
     --num_epochs 100 \
     --force_runs 2 5 6 7 8 9 10 11 12 14 16 22 24 25 26 27 28 29 31 32 33 34 35 36 37 38 39 41 42 43 44 45 46 47 48 49 50\
     --out_dir $fe_model_path \
     --normalize_targets \
     --use_acceleration \
     --seq_length 10 \
     --state linear

# evaluate on validation set
python src/evaluate.py --run 1 --weights $fe_model_path --model res_net --model_type transformer --use_acceleration --state linear
python src/evaluate.py --run 3 --weights $fe_model_path --model res_net --model_type transformer --use_acceleration --state linear
python src/evaluate.py --run 4 --weights $fe_model_path --model res_net --model_type transformer --use_acceleration --state linear
python src/evaluate.py --run 15 --weights $fe_model_path --model res_net --model_type transformer --use_acceleration --state linear
python src/evaluate.py --run 18 --weights $fe_model_path --model res_net --model_type transformer --use_acceleration --state linear
python src/evaluate.py --run 20 --weights $fe_model_path --model res_net --model_type transformer --use_acceleration --state linear
python src/evaluate.py --run 21 --weights $fe_model_path --model res_net --model_type transformer --use_acceleration --state linear
python src/evaluate.py --run 30 --weights $fe_model_path --model res_net --model_type transformer --use_acceleration --state linear

# visualize results on test set
python src/evaluate.py --run 13 --weights $fe_model_path --model res_net --model_type transformer --use_acceleration --state linear
python src/evaluate.py --run 17 --weights $fe_model_path --model res_net --model_type transformer --use_acceleration --state linear
python src/evaluate.py --run 19 --weights $fe_model_path --model res_net --model_type transformer --use_acceleration --state linear
python src/evaluate.py --run 23 --weights $fe_model_path --model res_net --model_type transformer --use_acceleration --state linear
python src/evaluate.py --run 40 --weights $fe_model_path --model res_net --model_type transformer --use_acceleration --state linear