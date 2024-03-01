import os
from typing import List
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from tqdm import tqdm
import argparse

from torch.utils.data import DataLoader

from util import load_dataset, apply_scaling_to_datasets, create_weights_path
from dataset import VisionRobotDataset
from transforms import CropBottom
from models import VisionRobotNet
from loss import RMSELoss
from trainer.trainer import Trainer
import constants

from torch.utils.tensorboard import SummaryWriter
WRITER = SummaryWriter()
layout = {
    "Training Plots": {
        "MSE": ["Multiline", ["MSE/train", "MSE/test"]],
        "RMSE": ["Multiline", ["RMSE/train", "RMSE/test"]],
    },
}
WRITER.add_custom_scalars(layout)


def parse_cmd_line() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", required=True)
    parser.add_argument("--lr", required=True)
    parser.add_argument("--num_epochs", required=True)
    parser.add_argument('--force_runs', nargs='+',
                        type=int, help='A list of the run numbers of the force policy rollouts that should be used for training', required=True)
    return parser.parse_args()


def train(args: argparse.Namespace):

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        CropBottom((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data_transforms = {"train": train_transform, "test": test_transform}
    run_nums = {"train": [args.force_runs, []],
                "test": [[9], []]}

    batch_size = int(args.batch_size)
    lr = float(args.lr)
    num_epochs = int(args.num_epochs)

    hparams = {
        'batch_size': batch_size,
        'lr': lr,
        'num_epochs': num_epochs
    }
    WRITER.add_hparams(hparams, {})

    data_dir = "data"
    sets = ["train", "test"]
    data_loaders: dict[DataLoader] = {}

    for s in sets:
        data = load_dataset(
            data_dir, force_policy_runs=run_nums[s][0], no_force_policy_runs=run_nums[s][1])
        dataset = VisionRobotDataset(
            *data, path=data_dir, transforms=data_transforms[s])
        print(f"Loaded Dataset {s} with {len(dataset)} samples!")
        data_loaders[s] = DataLoader(
            dataset, batch_size=batch_size, drop_last=True)

    apply_scaling_to_datasets(
        data_loaders["train"].dataset, data_loaders["test"].dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Using Device: {device}")

    model = VisionRobotNet(num_image_features=2 * constants.NUM_IMAGE_FEATURES,
                           num_robot_features=constants.NUM_ROBOT_FEATURES,
                           dropout_rate=0.2)
    model.to(device)

    weights_dir = create_weights_path(lr, num_epochs)
    trainer = Trainer(model,
                      data_loaders,
                      device,
                      criterion="mse",
                      lr=lr,
                      regularized=True,
                      weights_dir=weights_dir,
                      writer=WRITER)
    trainer.train(num_epochs=num_epochs)


if __name__ == "__main__":
    args = parse_cmd_line()
    train(args)
