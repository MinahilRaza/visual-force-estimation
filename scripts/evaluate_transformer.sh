#!/bin/bash

# Define cleanup or interrupt handling
handle_sigint() {
    echo "Caught SIGINT (Ctrl+C)! Cleaning up..."
    # Put any cleanup commands here
    exit 1
}

# Register the handler for SIGINT
trap handle_sigint SIGINT

if [[ "$OSTYPE" == "darwin"* ]]; then
  export OMP_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export MKL_NUM_THREADS=1
fi

seq_length=10
epochs=150
dataset="full" 

# Define the validation runs for each fold
declare -A fold_runs

if [ "$dataset" == "full" ]; then
    # For the full dataset, use the following runs
    fold_runs[0]="6 10 17 23 30 31 48"
    fold_runs[1]="8 11 16 21 32 35 37"
    fold_runs[2]="1 19 20 28 41 43 44"
    fold_runs[3]="2 3 7 15 25 38 49"
    fold_runs[4]="4 14 27 40 42 46 50"
    fold_runs[5]="9 18 22 24 26 34 47"

    loss_criterion=("l1" "huber" "mse")
else
    fold_runs[0]="1 10 11"
    fold_runs[1]="3 6 9"
    fold_runs[2]="2 5 12"
    fold_runs[3]="4 7 8"

    loss_criterion=("mse")
fi


for loss_criterion_name in "${loss_criterion[@]}"; do
    
    echo "Evaluating for Sequence Length: $seq_length, Epochs: $epochs, Loss Criterion: $loss_criterion_name"
    # Loop through each fold and its validation runs
    for fold in "${!fold_runs[@]}"; do
        if [ "$dataset" == "full" ]; then
            fe_model_path="weights/force_estimation_network/state-transformer_seq_${seq_length}_${loss_criterion_name}_epochs_${epochs}/fold_${fold}_of_6"
        else
            fe_model_path="weights/force_estimation_network/transformer-15runs_seq_${seq_length}_${loss_criterion}_epochs_${epochs}/fold_${fold}_of_4"
        fi

        echo "  Fold $fold, Model Path: $fe_model_path"

        for run in ${fold_runs[$fold]}; do
            echo "    Evaluating run $run"
            python src/evaluate.py --run $run --weights "$fe_model_path" --model res_net --model_type transformer --use_acceleration --state linear --seq_length $seq_length --loss_criterion $loss_criterion_name
        done
    done
done