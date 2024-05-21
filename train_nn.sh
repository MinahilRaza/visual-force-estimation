normal_model_path="weights/force_estimation_network/feed_forward"
fe_model_state=vision

python train.py --batch_size 32 \
    --lr 0.00002 \
    --model res_net \
    --num_epochs 200 \
    --force_runs 1 2 3 4 --out_dir $normal_model_path \
    --normalize_targets \
    --use_acceleration \
    --state $fe_model_state \
    --overfit

python evaluate.py --run 1 --weights $normal_model_path --model res_net --model_type vision_robot --use_acceleration --overfit --state $fe_model_state
python evaluate.py --run 2 --weights $normal_model_path --model res_net --model_type vision_robot --use_acceleration --overfit --state $fe_model_state
python evaluate.py --run 11 --weights $normal_model_path --model res_net --model_type vision_robot --use_acceleration --overfit --state $fe_model_state
