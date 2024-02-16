from typing import List, Optional
import torch
import numpy as np
import os

from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image

from pathlib import Path


class VisionRobotDataset(Dataset):
    """
    Dataset class to store left and right images and robot data
    """

    def __init__(self,
                 robot_features: np.ndarray,
                 force_targets: np.ndarray,
                 img_left_paths: List[str],
                 img_right_paths: List[str],
                 path: str,
                 transforms: Optional[transforms.Compose] = None) -> None:
        self.num_samples, self.num_robot_features = robot_features.shape
        assert force_targets.shape[0] == self.num_samples, \
            f"force_labels size: {
                force_targets.shape} does not match samples nr: {self.num_samples}"
        assert len(img_left_paths) == self.num_samples
        assert len(img_right_paths) == self.num_samples

        self.robot_features = torch.from_numpy(robot_features).float()
        self.force_targets = torch.from_numpy(force_targets).float()
        self.img_left_paths = img_left_paths
        self.img_right_paths = img_right_paths
        self.transforms = transforms
        self.path = Path(path)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx):
        img_left_path = self.path / self.img_left_paths[idx]
        img_right_path = self.path / self.img_right_paths[idx]
        img_left = Image.open(img_left_path)
        img_right = Image.open(img_right_path)

        if self.transforms:
            img_left = self.transforms(img_left)
            img_right = self.transforms(img_right)

        return {"img_left": img_left, "img_right": img_right, "features": self.robot_features[idx], "target": self.force_targets[idx]}
