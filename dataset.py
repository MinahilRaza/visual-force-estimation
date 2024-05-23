from typing import List, Optional
import torch
import numpy as np
import joblib
import os

from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image

from pathlib import Path


class SequentialDataset(Dataset):
    """
    Dataset class to handle sequences of robot state features and force targets.
    """

    def __init__(self, robot_features: np.ndarray, force_targets: np.ndarray, seq_length: int, feature_scaler_path: Optional[str] = None) -> None:
        self.robot_features = torch.from_numpy(robot_features).float()
        self.force_targets = torch.from_numpy(force_targets).float()
        self.seq_length = seq_length
        self.num_samples = len(robot_features) - seq_length + 1

        if feature_scaler_path:
            assert os.path.isfile(
                feature_scaler_path), f"{feature_scaler_path=}"
            self.feature_scaler = joblib.load(feature_scaler_path)
            self.robot_features = torch.from_numpy(
                self.feature_scaler.transform(robot_features)
            ).float()
        else:
            self.feature_scaler = None

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx):
        start = idx
        end = start + self.seq_length
        return {
            "features": self.robot_features[start:end],
            "target": self.force_targets[start:end]
        }


class AutoEncoderDataset(Dataset):
    """
    Dataset class to store left and right images to train an auto encoder
    """

    def __init__(self,
                 img_left_paths: List[str],
                 img_right_paths: List[str],
                 path: str,
                 transforms: Optional[transforms.Compose] = None) -> None:
        assert len(img_left_paths) == len(img_right_paths)
        self.img_paths = img_left_paths + img_right_paths
        assert len(self.img_paths) == len(
            img_left_paths) + len(img_right_paths)
        self.transforms = transforms
        self.path = Path(path)

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.path / self.img_paths[idx]
        img = Image.open(img_path)

        assert img.size[0] == img.size[1] == 256, \
            f"{img.size=}, {img_path=}"

        if self.transforms:
            img = self.transforms(img)

        return {"img": img, "target": img}


class VisionRobotDataset(Dataset):
    """
    Dataset class to store left and right images and robot data.
    Optionally applies pre-fitted StandardScaler to robot features and MinMaxScaler to force targets.
    """

    def __init__(self,
                 robot_features: np.ndarray,
                 force_targets: np.ndarray,
                 img_left_paths: List[str],
                 img_right_paths: List[str],
                 path: str,
                 img_transforms: Optional[transforms.Compose] = None,
                 feature_scaler_path: Optional[str] = None,
                 target_scaler_path: Optional[str] = None) -> None:
        self.num_samples, self.num_robot_features = robot_features.shape
        assert force_targets.shape[0] == self.num_samples, \
            f"force_labels size: \
            {force_targets.shape} does not match samples nr: {self.num_samples}"
        assert len(img_left_paths) == self.num_samples
        assert len(img_right_paths) == self.num_samples

        self.robot_features = torch.from_numpy(robot_features).float()
        self.force_targets = torch.from_numpy(force_targets).float()
        self.img_left_paths = img_left_paths
        self.img_right_paths = img_right_paths
        self.transforms = img_transforms
        self.path = Path(path)

        if feature_scaler_path:
            assert os.path.isfile(
                feature_scaler_path), f"{feature_scaler_path=}"
            self.feature_scaler = joblib.load(feature_scaler_path)
            self.robot_features = torch.from_numpy(
                self.feature_scaler.transform(robot_features)
            ).float()
        else:
            self.feature_scaler = None

        if target_scaler_path:
            assert os.path.isfile(
                target_scaler_path), f"{target_scaler_path=}"
            self.target_scaler = joblib.load(target_scaler_path)
            self.force_targets = torch.from_numpy(
                self.target_scaler.transform(force_targets)
            ).float()
        else:
            self.target_scaler = None

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx):
        img_left_path = self.path / self.img_left_paths[idx]
        img_right_path = self.path / self.img_right_paths[idx]
        img_left = Image.open(img_left_path)
        img_right = Image.open(img_right_path)

        assert img_left.size[0] == img_left.size[1] == 256, \
            f"{img_left.size=}, {img_left_path=}"
        assert img_right.size[0] == img_right.size[1] == 256, \
            f"{img_left.size=}, {img_right.size=}"

        if self.transforms:
            img_left = self.transforms(img_left)
            img_right = self.transforms(img_right)

        return {"img_left": img_left, "img_right": img_right, "features": self.robot_features[idx], "target": self.force_targets[idx]}
