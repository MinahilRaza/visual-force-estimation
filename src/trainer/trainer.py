import os
import socket

from typing import List, Optional, Dict, Tuple
from abc import ABC, abstractmethod
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from loss import RMSELoss


class LRSchedulerConfig(object):
    milestones = [20, 80, 160]
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
                 use_acceleration: bool,
                 lr_scheduler_config: Optional[LRSchedulerConfig] = None) -> None:
        self.model = model
        assert hasattr(
            self.model, "version"), "Model needs to have version set!"
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
        elif self.criterion_name == "custom":
            self.criterion = None
        else:
            raise ValueError(f"Invalid Criterion specified: {criterion}")

        self.lr = lr
        weights_decay = 1e-5 if regularized else 0.0
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weights_decay)
        if lr_scheduler_config is not None:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                  milestones=lr_scheduler_config.milestones,
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
        self.hostname = socket.gethostname()
        self.use_acceleration = use_acceleration

    @property
    @abstractmethod
    def task(self):
        pass

    @abstractmethod
    def run_model(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        pass

    def train(self, num_epochs: int):
        scaler = torch.cuda.amp.GradScaler()
        best_acc = float('inf')
        epoch_logs = []
        for i in range(num_epochs):
            print(f"Epoch {i+1}/{num_epochs}")
            print(f"LR: {self.optimizer.param_groups[0]['lr']}")
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
                        with torch.cuda.amp.autocast():
                            out, loss, acc = self.run_model(batch)
                            assert out.dtype == torch.float16, f"{out.dtype}"

                            total_loss_epoch += loss.item()
                            total_acc_epoch += acc.item()

                        if phase == "train":
                            scaler.scale(loss).backward()
                            scaler.step(self.optimizer)
                            scaler.update()
                            self.optimizer.zero_grad(set_to_none=True)

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

            if self.scheduler:
                self.scheduler.step()

            epoch_logs.append({
                "Test Loss": loss_phase['test'].item(),
                "Test RMSE": acc_phase['test'].item()})
            print(f"Train Loss: {loss_phase['train'].item():.4f}\t \
                Test Loss: {loss_phase['test'].item():.4f} Test RMSE: {acc_phase['test'].item():.4f}")
        self.best_test_acc = best_acc
        torch.save(self.model.state_dict(), self.save_path_last)
        self.save_logs(epoch_logs, best_acc)
        self.writer.flush()

    def save_logs(self, epoch_logs: List[str], best_acc: float) -> None:
        with open(os.path.join(self.weights_dir, "logs.txt"), "w", encoding="utf-8") as file:
            file.write(f"Task: {self.task}\n")
            file.write(f"Host: {self.hostname}\n")
            file.write(f"Model Config: {self.model.config}\n")
            file.write(f"Best Acc: {best_acc}\n")
            file.write(f"Using Acceleration: {self.use_acceleration}\n")
            file.write(f"Learning Rate: {self.lr}\n")
            file.write(f"Using LR Scheduler: {self.scheduler is not None}\n")
            for i, log in enumerate(epoch_logs):
                file.write(f"Epoch {i+1}: {log}\n")


class VarAutoEncoderTrainer(TrainerBase):
    task = "var_auto_encoder"

    def custom_loss(self, out: torch.Tensor, target: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        recon_loss = F.mse_loss(out, target)
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_divergence

    def run_model(self, batch: dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img = batch["img"].to(self.device)
        target = batch["target"].to(self.device)
        out, z, mean, logvar = self.model(img)
        loss = self.custom_loss(out, target, mean, logvar)
        acc: torch.Tensor = self.acc_module(out, target)
        return out, loss, acc


class AutoEncoderTrainer(TrainerBase):
    task = "auto_encoder"

    def run_model(self, batch: dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img = batch["img"].to(self.device)
        target = batch["target"].to(self.device)
        out: torch.Tensor = self.model(img)
        loss: torch.Tensor = self.criterion(out, target)
        acc: torch.Tensor = self.acc_module(out, target)
        return out, loss, acc


class ForceEstimationTrainer(TrainerBase):
    task = "force_estimation"

    def run_model(self, batch: dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_left = batch["img_left"].to(self.device)
        img_right = batch["img_right"].to(self.device)
        features = batch["features"].to(self.device)
        target = batch["target"].to(self.device)
        out = self.model(img_left, img_right, features)
        loss: torch.Tensor = self.criterion(out, target)
        acc: torch.Tensor = self.acc_module(out, target)
        return out, loss, acc


class TransformerTrainer(TrainerBase):
    task = "robot_state_transformer"

    def run_model(self, batch: dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        robot_state = batch["features"].to(self.device)
        target = batch["target"].to(self.device)[:, -1, :]

        out = self.model(robot_state)[:, -1, :]

        loss: torch.Tensor = self.criterion(out, target)
        acc: torch.Tensor = self.acc_module(out, target)
        return out, loss, acc