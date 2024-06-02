import json
from typing import Dict, List
import torch
import argparse
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.robot_state_transformer import RobotStateTransformer, TransformerConfig, EncoderState
from trainer.trainer import TransformerTrainer, LRSchedulerConfig
from dataset import SequentialDataset

import util
import constants

NUM_EPOCHS = 50
FORCE_RUNS = [1, 2, 3, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15]
NO_FORCE_RUNS = [1, 2, 3, 4, 5, 6]
USE_ACCELERATION = True
NORMALIZE_TARGETS = True
COUNT = 0


def get_transformer_config(num_robot_features: int,
                           hidden_layers: list,
                           dropout_rate: float,
                           num_heads: int,
                           num_encoder_layers: int,
                           num_decoder_layers: int,
                           dim_feedforward: int) -> TransformerConfig:
    return TransformerConfig(
        num_robot_features=num_robot_features,
        hidden_layers=hidden_layers,
        dropout_rate=dropout_rate,
        use_batch_norm=True,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        encoder_state=EncoderState.LINEAR
    )


def load_data(seq_length: int, batch_size) -> Dict[str, DataLoader]:
    data_dir = "data"
    run_nums = {"train": (FORCE_RUNS, NO_FORCE_RUNS),
                "test": constants.DEFAULT_TEST_RUNS}
    sets = ["train", "test"]
    data_loaders = {}

    for s in sets:
        features, targets, _, _ = util.load_dataset(path=data_dir,
                                                    force_policy_runs=run_nums[s][0],
                                                    no_force_policy_runs=run_nums[s][1],
                                                    sequential=True,
                                                    use_acceleration=USE_ACCELERATION,
                                                    crop_runs=False)
        assert isinstance(features, list)
        assert isinstance(targets, list)
        dataset_kwargs = {"feature_scaler_path": constants.FEATURE_SCALER_FN,
                          "target_scaler_path": constants.TARGET_SCALER_FN} if s == "test" else {}
        dataset = SequentialDataset(robot_features_list=features,
                                    force_targets_list=targets,
                                    normalize_targets=NORMALIZE_TARGETS,
                                    seq_length=seq_length,
                                    **dataset_kwargs)
        print(
            f"[INFO] Loaded Sequential Dataset {s} with {len(dataset)} samples!")
        data_loaders[s] = DataLoader(
            dataset, batch_size=batch_size, shuffle=(s == "train"), drop_last=True)

    return data_loaders


def train_and_evaluate(batch_size: int,
                       lr: float,
                       hidden_layers: List[int],
                       num_heads: int,
                       num_encoder_layers: int,
                       num_decoder_layers: int,
                       dim_feedforward: int,
                       dropout_rate: float,
                       seq_length: int) -> float:
    global COUNT
    print(f"Training with batch_size={batch_size}, lr={lr}, hidden_layers={hidden_layers}, num_heads={num_heads}, num_encoder_layers={num_encoder_layers}, num_decoder_layers={num_decoder_layers}, dim_feedforward={dim_feedforward}, dropout_rate={dropout_rate}")

    log_dir = f"runs/search/{COUNT}"
    COUNT += 1
    writer = SummaryWriter(log_dir=log_dir)

    data_loaders = load_data(seq_length, batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_robot_features = constants.NUM_ROBOT_FEATURES_INCL_ACCEL
    config = get_transformer_config(num_robot_features, hidden_layers, dropout_rate,
                                    num_heads, num_encoder_layers, num_decoder_layers, dim_feedforward)
    model = RobotStateTransformer(config)
    model.to(device)

    print(f"[INFO] Using Device: {device}")
    print(f"[INFO] Training Model with Seq Length: {seq_length}")
    print(f"[INFO] Batch Size: {batch_size}")
    print(f"[INFO] Learning Rate: {lr}")

    weights_dir = "weights/force_estimation_network/transformer_search"
    lr_scheduler_config = None
    trainer = TransformerTrainer(model,
                                 data_loaders,
                                 device,
                                 criterion="mse",
                                 lr=lr,
                                 regularized=True,
                                 weights_dir=weights_dir,
                                 writer=writer,
                                 lr_scheduler_config=lr_scheduler_config,
                                 use_acceleration=USE_ACCELERATION)
    trainer.train(num_epochs=NUM_EPOCHS)
    return trainer.best_test_acc


def hyperparameter_search():
    global COUNT
    batch_sizes = [16, 32, 64]
    learning_rates = [1e-4, 1e-5, 1e-6]
    hidden_layers = [[128, 256], [256, 512]]
    num_heads_options = [4]
    num_encoder_layers_options = [4]
    num_decoder_layers_options = [2]
    dim_feedforward_options = [256]
    dropout_rates = [0.3]
    seq_lengths = [5, 10, 20]

    best_rmse = float('inf')
    best_params = None
    log_file = "hyperparameter_search_log.json"
    log_data = {}

    for seq_len in seq_lengths:
        for batch_size in batch_sizes:
            for lr in learning_rates:
                for hidden_layer in hidden_layers:
                    for num_heads in num_heads_options:
                        for num_encoder_layers in num_encoder_layers_options:
                            for num_decoder_layers in num_decoder_layers_options:
                                for dim_feedforward in dim_feedforward_options:
                                    for dropout_rate in dropout_rates:
                                        rmse = train_and_evaluate(batch_size,
                                                                  lr,
                                                                  hidden_layer,
                                                                  num_heads,
                                                                  num_encoder_layers,
                                                                  num_decoder_layers,
                                                                  dim_feedforward,
                                                                  dropout_rate,
                                                                  seq_len)
                                        log_data[COUNT] = {
                                            "batch_size": batch_size,
                                            "lr": lr,
                                            "hidden_layers": hidden_layer,
                                            "num_heads": num_heads,
                                            "num_encoder_layers": num_encoder_layers,
                                            "num_decoder_layers": num_decoder_layers,
                                            "dim_feedforward": dim_feedforward,
                                            "dropout_rate": dropout_rate,
                                            "seq_length": seq_len,
                                            "rmse": rmse
                                        }
                                        if rmse < best_rmse:
                                            print("="*40)
                                            print(
                                                f"Found new best RMSE: {rmse}")
                                            print("="*40)

                                            best_rmse = rmse
                                            best_params = (batch_size, lr, hidden_layer, num_heads, num_encoder_layers,
                                                           num_decoder_layers, dim_feedforward, dropout_rate, seq_len)
                                        with open(log_file, "w") as f:
                                            json.dump(log_data, f, indent=4)

    print(f"Best RMSE: {best_rmse}")
    print(f"Best Parameters: \nbatch_size={best_params[0]}\nlr={best_params[1]}\nhidden_layers={best_params[2]} \n num_heads={best_params[3]}\n num_encoder_layers={best_params[4]}\n num_decoder_layers={best_params[5]}\n dim_feedforward={best_params[6]}\n dropout_rate={best_params[7]}")


if __name__ == "__main__":
    hyperparameter_search()
