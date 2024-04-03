import os
from typing import List, Optional, Dict
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from loss import RMSELoss

from abc import ABC, abstractmethod


class LRSchedulerConfig(object):
    step_size = 20
    gamma = 0.1


class TrainerBase(ABC):
    def __init__(self,
                 model: nn.Module,
                 data_loaders: dict[DataLoader],
                 device: torch.device,
                 criterion: str,
                 lr: float,
                 regularized: bool,
                 weights_dir: str,
                 writer: SummaryWriter,
                 lr_scheduler_config: Optional[LRSchedulerConfig] = None) -> None:
        self.model = model
        self.data_loaders = data_loaders
        assert "train" in data_loaders and "test" in data_loaders, \
            f"{data_loaders=}"

        self.device = device
        self.phases = ["train", "test"]

        assert isinstance(criterion, str), f"{criterion=}"
        self.criterion_name = criterion
        if self.criterion_name == "mse":
            self.criterion = nn.MSELoss()
            self.criterion.to(device)
        elif self.criterion_name == "rmse":
            self.criterion = RMSELoss()
            self.criterion.to(device)
        else:
            raise ValueError(f"Invalid Criterion specified: {criterion}")

        self.lr = lr
        weights_decay = 1e-5 if regularized else 0.0
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weights_decay)
        if lr_scheduler_config is not None:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                             step_size=lr_scheduler_config.step_size,
                                                             gamma=lr_scheduler_config.gamma)
        else:
            self.scheduler = None

        self.acc_module = RMSELoss()
        self.acc_module.to(device)

        self.weights_dir = weights_dir
        os.makedirs(weights_dir, exist_ok=True)
        self.save_path_best = os.path.join(weights_dir, "best_params.pth")
        self.save_path_last = os.path.join(weights_dir, "last_params.pth")

        self.writer = writer

    @abstractmethod
    def run_model(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        pass

    def train(self, num_epochs: int):
        scaler = torch.cuda.amp.GradScaler()
        best_acc = float('inf')
        epoch_logs = []
        for i in range(num_epochs):
            print(f"Epoch {i+1}/{num_epochs}")
            loss_phase = {}
            acc_phase = {}
            for phase in self.phases:
                if phase == "train":
                    self.model.train()
                else:
                    self.model.eval()

                total_loss_epoch = torch.zeros(
                    1, requires_grad=False, device=self.device)
                total_acc_epoch = torch.zeros(
                    1, requires_grad=False, device=self.device)

                with torch.set_grad_enabled(phase == "train"):
                    for batch in tqdm(self.data_loaders[phase]):
                        target = batch["target"].to(self.device)
                        with torch.cuda.amp.autocast():
                            out = self.run_model(batch)
                            assert out.dtype == torch.float16, f"{out.dtype}"

                            loss: torch.Tensor = self.criterion(
                                out, target)
                            acc: torch.Tensor = self.acc_module(
                                out, target)
                            total_loss_epoch += loss.item()
                            total_acc_epoch += acc.item()

                        if phase == "train":
                            scaler.scale(loss).backward()
                            scaler.step(self.optimizer)
                            scaler.update()
                            self.optimizer.zero_grad(set_to_none=True)
                            if self.scheduler:
                                self.scheduler.step()

                avg_loss_epoch = total_loss_epoch / \
                    len(self.data_loaders[phase])
                avg_acc_epoch = total_acc_epoch / len(self.data_loaders[phase])
                loss_phase[phase] = avg_loss_epoch
                acc_phase[phase] = avg_acc_epoch
                self.writer.add_scalar(
                    f"{self.criterion_name}/{phase}", avg_loss_epoch.item(), i)
                self.writer.add_scalar(
                    f"RMSE/{phase}", avg_acc_epoch.item(), i)

                if phase == "test" and avg_acc_epoch.item() < best_acc:
                    best_acc = avg_acc_epoch.item()
                    torch.save(self.model.state_dict(), self.save_path_best)
                    print(f"Saved new best model with \
                        RMSE:{round(avg_acc_epoch.item(), 4)}")
            epoch_logs.append({
                "Test Loss": loss_phase['test'].item(),
                "Test RMSE": acc_phase['test'].item()})
            print(f"Train Loss: {loss_phase['train'].item():.2f}\t \
                Test Loss: {loss_phase['test'].item():.2f} Test RMSE: {acc_phase['test'].item():.2f}")
        torch.save(self.model.state_dict(), self.save_path_last)
        self.save_logs(epoch_logs, best_acc)
        self.writer.flush()

    def save_logs(self, epoch_logs: List[str], best_acc: float) -> None:
        with open(os.path.join(self.weights_dir, "logs.txt"), "w", encoding="utf-8") as file:
            file.write(f"CNN Model: {self.model.cnn_version}\n")
            file.write(f"Best Acc: {best_acc}\n")
            for i, log in enumerate(epoch_logs):
                file.write(f"Epoch {i}: {log}\n")


class AutoEncoderTrainer(TrainerBase):
    def run_model(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        img = batch["img"].to(self.device)
        return self.model(img)


class ForceEstimationTrainer(TrainerBase):
    def run_model(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        img_left = batch["img_left"].to(self.device)
        img_right = batch["img_right"].to(self.device)
        features = batch["features"].to(self.device)
        return self.model(img_left, img_right, features)
