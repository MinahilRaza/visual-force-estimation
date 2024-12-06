import torch
import argparse
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.robot_state_transformer import RobotStateTransformer
from trainer.trainer import TransformerTrainer, LRSchedulerConfig
from dataset import SequentialDataset

import util
import constants
from datetime import datetime

def parse_cmd_line() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", required=True, type=int)
    parser.add_argument("--lr", required=True, type=float)
    parser.add_argument("--num_epochs", required=True, type=int)
    parser.add_argument('--force_runs', nargs='+', type=int, required=True)
    parser.add_argument('--no_force_runs', nargs='*', type=int, default=[])
    parser.add_argument('--lr_scheduler', action='store_true', default=False)
    parser.add_argument('--use_acceleration',
                        action='store_true', default=False)
    parser.add_argument('--normalize_targets',
                        action='store_true', default=False)
    parser.add_argument("--out_dir", default=None, type=str)
    parser.add_argument("--overfit", action='store_true', default=False)
    parser.add_argument("--seq_length", required=True,
                        type=int, help="Length of the input sequences")
    parser.add_argument('--state',
                        choices=['linear', 'conv'],
                        required=True,
                        help='Set the model state: linear for using a linear feature extractor, conv for a Conv1 Layer'
                        )

    return parser.parse_args()


def train():
    """
    Train a Robot State Transformer model.

    This function trains a Robot State Transformer model based on the command line arguments.

    The following hyperparameters can be set through command line arguments:

    - batch_size: The batch size to use.
    - lr: The learning rate to use.
    - num_epochs: The number of epochs to train for.
    - seq_length: The length of the input sequences.
    - state: The model state to use, either 'linear' for a linear feature extractor or 'conv' for a Conv1 Layer feature extractor.
    - force_runs: The runs to use for training.
    - no_force_runs: The runs not to use for training.
    - lr_scheduler: Whether or not to use a learning rate scheduler.
    - use_acceleration: Whether or not to use acceleration as an input feature.
    - normalize_targets: Whether or not to normalize the targets.
    - out_dir: The directory to save the trained model weights in. If not specified, will save in
        "weights/robot_state_transformer/<num_epochs>_epochs_lr_<lr>".
    - overfit: If set, will only use one batch for training and testing.

    The function will print out various information about the training process, including the batch size, learning rate,
    number of epochs, sequence length, model state, and whether or not acceleration is being used.

    The function will also save the trained model weights in the specified directory.

    :return: None
    """
    args = parse_cmd_line()
    args.model = "transformer"
    args.use_pretrained = False

    print("Training Robot State Transformer Network")

    run_nums = util.get_run_numbers(args)
    current_time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = util.get_log_dir(args) + current_time
    
    hparams = {
        'batch_size': args.batch_size,
        'lr': args.lr,
        'num_epochs': args.num_epochs,
        'seq_length': args.seq_length
    }
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_hparams(hparams, {})

    data_dir = "data"
    sets = ["train", "test"]
    data_loaders = {}

    for s in sets:
        features, targets, _, _ = util.load_dataset(path=data_dir,
                                                    force_policy_runs=run_nums[s][0],
                                                    no_force_policy_runs=run_nums[s][1],
                                                    sequential=True,
                                                    use_acceleration=args.use_acceleration,
                                                    crop_runs=False)
        assert isinstance(features, list)
        assert isinstance(targets, list)
        dataset_kwargs = {"feature_scaler_path": constants.FEATURE_SCALER_FN,
                          "target_scaler_path": constants.TARGET_SCALER_FN} if s == "test" else {}
        dataset = SequentialDataset(robot_features_list=features,
                                    force_targets_list=targets,
                                    normalize_targets=args.normalize_targets,
                                    seq_length=args.seq_length,
                                    **dataset_kwargs)
        print(
            f"[INFO] Loaded Sequential Dataset {s} with {len(dataset)} samples!")
        data_loaders[s] = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=(s == "train"), drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = util.get_transformer_config(args)
    model = RobotStateTransformer(config)
    model.to(device)

    print(f"[INFO] Using Device: {device}")
    print(f"[INFO] Training Model with Seq Length: {args.seq_length}")
    print(f"[INFO] Batch Size: {args.batch_size}")
    print(f"[INFO] Learning Rate: {args.lr}")
    print(f"[INFO] State: {args.state}")

    weights_dir = util.create_weights_path(
        "robot_state_transformer", args.num_epochs) if not args.out_dir else args.out_dir
    lr_scheduler_config = LRSchedulerConfig() if args.lr_scheduler else None
    trainer = TransformerTrainer(model,
                                 data_loaders,
                                 device,
                                 criterion="mse",
                                 lr=args.lr,
                                 regularized=True,
                                 weights_dir=weights_dir,
                                 writer=writer,
                                 lr_scheduler_config=lr_scheduler_config,
                                 use_acceleration=args.use_acceleration)
    trainer.train(num_epochs=args.num_epochs)


if __name__ == "__main__":
    train()