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
    parser.add_argument('--use_pretrained',
                        action='store_true', default=False)
    parser.add_argument("--out_dir", default=None, type=str)

    return parser.parse_args()


def train():
    args = parse_cmd_line()
    print("Training Force Estimation Network")

    data_transforms = {"train": constants.RES_NET_TEST_TRANSFORM,
                       "test": constants.RES_NET_TRAIN_TRANSFORM}
    run_nums = {"train": [args.force_runs, args.no_force_runs],
                "test": [[2], []]}

    hparams = {
        'batch_size': args.batch_size,
        'lr': args.lr,
        'num_epochs': args.num_epochs,
        'model': args.model
    }
    log_dir = util.get_log_dir(args)
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_custom_scalars(constants.LAYOUT)
    writer.add_hparams(hparams, {})

    data_dir = "data"
    sets = ["train", "test"]
    data_loaders: dict[DataLoader] = {}

    for s in sets:
        data = util.load_dataset(
            data_dir,
            force_policy_runs=run_nums[s][0],
            no_force_policy_runs=run_nums[s][1],
            use_acceleration=args.use_acceleration)
        dataset = VisionRobotDataset(
            *data, path=data_dir, transforms=data_transforms[s])
        print(f"[INFO] Loaded Dataset {s} with {len(dataset)} samples!")
        data_loaders[s] = DataLoader(
            dataset, batch_size=args.batch_size, drop_last=True)

    util.apply_scaling_to_datasets(
        data_loaders["train"].dataset,
        data_loaders["test"].dataset,
        normalize_targets=args.normalize_targets)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Using Device: {device}")
    print(f"[INFO] Training Model: {args.model}")
    print(f"[INFO] Batch Size: {args.batch_size}")
    print(f"[INFO] Learning Rate: {args.lr}")
    print(f"[INFO] Pretrained Weights: {args.use_pretrained}")

    model = VisionRobotNet(cnn_model_version=args.model,
                           num_image_features=constants.NUM_IMAGE_FEATURES,
                           num_robot_features=util.get_num_robot_features(
                               args),
                           use_pretrained=args.use_pretrained,
                           dropout_rate=0.2)
    model.to(device)

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
