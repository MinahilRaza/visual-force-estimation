import os
from typing import List
import torch
import argparse

from torch.utils.data import DataLoader

from dataset import VisionRobotDataset
from models.vision_robot_net import VisionRobotNet
from trainer.trainer import ForceEstimationTrainer, LRSchedulerConfig
import constants
import util

from torch.utils.tensorboard import SummaryWriter


def parse_cmd_line() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", required=True, type=int)
    parser.add_argument("--lr", required=True, type=float)
    parser.add_argument("--num_epochs", required=True, type=int)
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument('--force_runs', nargs='+',
                        type=int, help='A list of the run numbers of the force policy rollouts that should be used for training', required=True)
    parser.add_argument('--no_force_runs', nargs='*',
                        type=int, default=[],
                        help='A list of the run numbers of the NO force policy rollouts that should be used for training')

    parser.add_argument('--lr_scheduler', action='store_true', default=False)
    parser.add_argument('--use_acceleration',
                        action='store_true', default=False)
    parser.add_argument('--normalize_targets',
                        action='store_true', default=False)
    parser.add_argument('--use_pretrained', action='store_true', default=False)
    parser.add_argument("--out_dir", default=None, type=str)
    parser.add_argument("--overfit", action='store_true', default=False)
    parser.add_argument('--state',
                        choices=['both', 'robot', 'vision'],
                        required=True,
                        help='Set the model state: both for VISION_AND_ROBOT, robot for ROBOT_ONLY, vision for VISION_ONLY'
                        )

    return parser.parse_args()


def train():
    """
    Train a VisionRobotNet model.

    This function trains a VisionRobotNet model based on the command line arguments.

    The following hyperparameters can be set through command line arguments:

    - batch_size: The batch size to use.
    - lr: The learning rate to use.
    - num_epochs: The number of epochs to train for.
    - model: The model to use, either 'res_net' or 'efficient_net'.
    - force_runs: The runs to use for training.
    - no_force_runs: The runs not to use for training.
    - lr_scheduler: Whether or not to use a learning rate scheduler.
    - use_acceleration: Whether or not to use acceleration as an input feature.
    - normalize_targets: Whether or not to normalize the targets.
    - out_dir: The directory to save the trained model weights in. If not specified, will save in
        "weights/force_estimation_network/<num_epochs>_epochs_lr_<lr>".
    - overfit: If set, will only use one batch for training and testing.

    The function will print out various information about the training process, including the batch size, learning rate,
    number of epochs, sequence length, model state, and whether or not acceleration is being used.

    The function will also save the trained model weights in the specified directory.
    """
    args = parse_cmd_line()
    print("Training Force Estimation Network")

    data_transforms = util.get_image_transforms(args)
    run_nums = util.get_run_numbers(args)
    log_dir = util.get_log_dir(args)

    hparams = {
        'batch_size': args.batch_size,
        'lr': args.lr,
        'num_epochs': args.num_epochs,
        'model': args.model
    }
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_custom_scalars(constants.LAYOUT)
    writer.add_hparams(hparams, {})

    data_dir = "data"
    sets = ["train", "test"]
    data_loaders: dict[str, DataLoader] = {}

    for s in sets:
        data = util.load_dataset(
            data_dir,
            force_policy_runs=run_nums[s][0],
            no_force_policy_runs=run_nums[s][1],
            sequential=False,
            use_acceleration=args.use_acceleration)
        dataset = VisionRobotDataset(
            *data, path=data_dir, img_transforms=data_transforms[s])
        print(f"[INFO] Loaded Dataset {s} with {len(dataset)} samples!")
        data_loaders[s] = DataLoader(
            dataset, batch_size=args.batch_size, drop_last=True)

    util.apply_scaling_to_datasets(
        data_loaders["train"].dataset,
        data_loaders["test"].dataset,
        normalize_targets=args.normalize_targets)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_config = util.get_vrn_config(args)
    model = VisionRobotNet(model_config)
    model.to(device)

    print(f"[INFO] Using Device: {device}")
    print(f"[INFO] Training Model: {args.model}")
    print(f"[INFO] Batch Size: {args.batch_size}")
    print(f"[INFO] Learning Rate: {args.lr}")
    print(f"[INFO] Pretrained Weights: {args.use_pretrained}")
    print(f"[INFO] Model State: {model_config.model_state}")

    weights_dir = util.create_weights_path(
        args.model, args.num_epochs) if not args.out_dir else args.out_dir
    lr_scheduler_config = LRSchedulerConfig() if args.lr_scheduler else None
    trainer = ForceEstimationTrainer(model,
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
