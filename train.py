import os
from typing import List
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from tqdm import tqdm

from torch.utils.data import DataLoader

from util import load_dataset
from dataset import VisionRobotDataset
from transforms import CropBottom
from models import VisionRobotNet


def train_model(model: VisionRobotNet, data_loaders: dict[DataLoader], device: torch.device, phases: List[str], num_epochs: int, lr: float):
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for i in range(num_epochs):
        loss_phase = {}
        print(f"Epoch {i+1}/{num_epochs}")
        for phase in phases:
            if phase == "train":
                model.train()
            else:
                model.eval()

            total_loss_epoch = torch.zeros(
                1, requires_grad=False, device=device)

            with torch.set_grad_enabled(phase == "train"):
                for batch in tqdm(data_loaders[phase]):
                    img_left = batch["img_left"].to(device)
                    img_right = batch["img_right"].to(device)
                    features = batch["features"].to(device)
                    target = batch["target"].to(device)

                    out = model(img_left, img_right, features)
                    loss = criterion(out, target)
                    total_loss_epoch += loss

                    if phase == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

            avg_loss_epoch = total_loss_epoch / len(data_loaders[phase])
            loss_phase[phase] = avg_loss_epoch
        print(f"Train Loss: {loss_phase["train"].item()}\t \
              Test Loss: {loss_phase["test"].item()}")


def train():

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
    run_nums = {"train": [2, 1], "test": [7]}

    batch_size = 8

    data_dir = "data"
    sets = ["train", "test"]
    data_loaders = {}

    for s in sets:
        path = os.path.join(data_dir, s)
        data = load_dataset(path, run_nums=run_nums[s])
        dataset = VisionRobotDataset(
            *data, path=path, transforms=data_transforms[s])
        print(f"Loaded Dataset {s} with {len(dataset)} samples!")
        data_loaders[s] = DataLoader(dataset, batch_size=batch_size)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"[INFO] Using Device: {device}")

    model = VisionRobotNet(num_image_features=30,
                           num_robot_features=38,
                           dropout_rate=0.2)
    model.to(device)

    train_model(model, data_loaders, device, sets, num_epochs=10, lr=0.0001)


if __name__ == "__main__":
    train()
