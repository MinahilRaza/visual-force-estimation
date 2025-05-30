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
dataset="reduced" # Change to "full" for the full dataset

# Define the validation runs for each fold
declare -A fold_runs

if dataset == "full"; then
    # For the full dataset, use the following runs
    fold_runs[0]="1 2 3 4 5 6 7 8 9 10"
    fold_runs[1]="11 12 13 14 15 16 17 18"
    fold_runs[2]="19 20 21 22 23 24 25 26"
    fold_runs[3]="27 28 29 30 31 32 33 34"
    fold_runs[4]="35 36 37 38 39 40"
    fold_runs[5]="41 42 43 44 45"

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