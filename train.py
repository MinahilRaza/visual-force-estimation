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
    return parser.parse_args()


def train_model(model: VisionRobotNet,
                data_loaders: dict[DataLoader],
                device: torch.device,
                phases: List[str],
                num_epochs: int,
                lr: float,
                weights_dir: str) -> None:
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    rmse = RMSELoss()

    best_acc = float('inf')
    save_path_best = os.path.join(weights_dir, "best_params.pth")
    save_path_last = os.path.join(weights_dir, "last_params.pth")
    os.makedirs(weights_dir, exist_ok=True)

    for i in range(num_epochs):
        print(f"Epoch {i+1}/{num_epochs}")
        loss_phase = {}
        acc_phase = {}
        for phase in phases:
            if phase == "train":
                model.train()
            else:
                model.eval()

            total_loss_epoch = torch.zeros(
                1, requires_grad=False, device=device)
            total_acc_epoch = torch.zeros(
                1, requires_grad=False, device=device)

            with torch.set_grad_enabled(phase == "train"):
                for batch in tqdm(data_loaders[phase]):
                    img_left = batch["img_left"].to(device)
                    img_right = batch["img_right"].to(device)
                    features = batch["features"].to(device)
                    target = batch["target"].to(device)

                    out = model(img_left, img_right, features)

                    loss = criterion(out, target)
                    acc = rmse(out, target)
                    total_loss_epoch += loss
                    total_acc_epoch += acc

                    if phase == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

            avg_loss_epoch = total_loss_epoch / len(data_loaders[phase])
            avg_acc_epoch = total_acc_epoch / len(data_loaders[phase])
            loss_phase[phase] = avg_loss_epoch
            acc_phase[phase] = avg_acc_epoch
            WRITER.add_scalar(f"MSE/{phase}", avg_loss_epoch.item(), i)
            WRITER.add_scalar(f"RMSE/{phase}", avg_acc_epoch.item(), i)

            if phase == "test" and avg_acc_epoch.item() < best_acc:
                best_acc = avg_acc_epoch.item()
                torch.save(model.state_dict(), save_path_best)
                print(f"Saved new best model with \
                      RMSE:{round(avg_acc_epoch.item(), 4)}")
        print(f"Train Loss: {loss_phase['train'].item()}\t \
              Test Loss: {loss_phase['test'].item()} Test RMSE: {acc_phase['test'].item()}")
    torch.save(model.state_dict(), save_path_last)


def train(args: argparse.Namespace):

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
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
    run_nums = {"train": [1, 2, 3], "test": [7]}

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
        path = os.path.join(data_dir, s)
        data = load_dataset(path, run_nums=run_nums[s])
        dataset = VisionRobotDataset(
            *data, path=path, transforms=data_transforms[s])
        print(f"Loaded Dataset {s} with {len(dataset)} samples!")
        data_loaders[s] = DataLoader(dataset, batch_size=batch_size)

    apply_scaling_to_datasets(
        data_loaders["train"].dataset, data_loaders["test"].dataset)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"[INFO] Using Device: {device}")

    model = VisionRobotNet(num_image_features=30,
                           num_robot_features=41,
                           dropout_rate=0.2)
    model.to(device)

    weights_dir = create_weights_path(lr, num_epochs)
    train_model(model, data_loaders, device, sets,
                num_epochs=num_epochs, lr=lr, weights_dir=weights_dir)
    WRITER.flush()


if __name__ == "__main__":
    args = parse_cmd_line()
    train(args)
