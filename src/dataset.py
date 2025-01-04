from typing import List, Optional
import torch
import numpy as np
import joblib
import os

from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler


import constants


class FeatureDataset(Dataset):
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


class SequentialDataset(Dataset):
    """
    Dataset class to handle sequences of robot state features and force targets.
    """

    def __init__(self,
                 robot_features_list: List[np.ndarray],
                 force_targets_list: List[np.ndarray],
                 seq_length: int,
                 normalize_targets: bool,
                 feature_scaler_path: Optional[str] = None,
                 target_scaler_path: Optional[str] = None) -> None:
        assert isinstance(robot_features_list, list)
        assert isinstance(force_targets_list, list)
        self.robot_features = []
        self.force_targets = []
        self.seq_length = seq_length

        for robot_features, force_targets in zip(robot_features_list, force_targets_list):
            self.robot_features.append(
                torch.from_numpy(robot_features).float())
            self.force_targets.append(torch.from_numpy(force_targets).float())

        self.num_samples_per_run = [
            len(features) - seq_length + 1 for features in self.robot_features]
        self.cumulative_samples = np.cumsum(self.num_samples_per_run)

        if feature_scaler_path:
            assert os.path.isfile(
                feature_scaler_path), f"{feature_scaler_path=}"
            self.feature_scaler = joblib.load(feature_scaler_path)
            for i in range(len(self.robot_features)):
                self.robot_features[i] = torch.from_numpy(
                    self.feature_scaler.transform(
                        self.robot_features[i].numpy())
                ).float()
        else:
            self.feature_scaler = StandardScaler()
            self._fit_scaler()
            self._transform_features()
            joblib.dump(self.feature_scaler, constants.FEATURE_SCALER_FN)

        if normalize_targets:
            if target_scaler_path:
                assert os.path.isfile(
                    target_scaler_path), f"{target_scaler_path=}"
                self.target_scaler = joblib.load(target_scaler_path)
                self._transform_targets()
            else:
                self.target_scaler = MinMaxScaler(feature_range=(-1, 1))
                self._fit_target_scaler()
                self._transform_targets()
                joblib.dump(self.target_scaler, constants.TARGET_SCALER_FN)
        else:
            self.target_scaler = None

    def _fit_scaler(self):
        all_features = np.concatenate(
            [features.numpy() for features in self.robot_features])
        self.feature_scaler.fit(all_features)

    def _transform_features(self):
        for i in range(len(self.robot_features)):
            self.robot_features[i] = torch.from_numpy(
                self.feature_scaler.transform(self.robot_features[i].numpy())
            ).float()

    def _fit_target_scaler(self):
        all_targets = np.concatenate([targets.numpy()
                                     for targets in self.force_targets])
        self.target_scaler.fit(all_targets)

    def _transform_targets(self):
        for i in range(len(self.force_targets)):
            self.force_targets[i] = torch.from_numpy(
                self.target_scaler.transform(self.force_targets[i].numpy())
            ).float()

    def __len__(self) -> int:
        return sum(self.num_samples_per_run)

    def __getitem__(self, idx):
        run_idx = np.searchsorted(self.cumulative_samples, idx, side='right')
        if run_idx == 0:
            start = idx
        else:
            start = idx - self.cumulative_samples[run_idx - 1]

        end = start + self.seq_length
        return {
            "features": self.robot_features[run_idx][start:end],
            "target": self.force_targets[run_idx][start:end]
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


class AutoregressiveDataset(Dataset):
    """
    Dataset class to handle autoregressive sequences of robot state features and force targets.
    """

    def __init__(self,
                 robot_features_list: List[np.ndarray],
                 force_targets_list: List[np.ndarray],
                 seq_length: int,
                 normalize_targets: bool,
                 feature_scaler_path: Optional[str] = None,
                 target_scaler_path: Optional[str] = None) -> None:
        assert isinstance(robot_features_list, list)
        assert isinstance(force_targets_list, list)
        self.robot_features = []
        self.force_targets = []
        self.seq_length = seq_length

        for robot_features, force_targets in zip(robot_features_list, force_targets_list):
            self.robot_features.append(
                torch.from_numpy(robot_features).float())
            self.force_targets.append(torch.from_numpy(force_targets).float())

        if feature_scaler_path:
            assert os.path.isfile(
                feature_scaler_path), f"{feature_scaler_path=}"
            self.feature_scaler = joblib.load(feature_scaler_path)
            for i in range(len(self.robot_features)):
                self.robot_features[i] = torch.from_numpy(
                    self.feature_scaler.transform(
                        self.robot_features[i].numpy())
                ).float()
        else:
            self.feature_scaler = StandardScaler()
            self._fit_scaler()
            self._transform_features()
            joblib.dump(self.feature_scaler, constants.FEATURE_SCALER_FN)

        if normalize_targets:
            if target_scaler_path:
                assert os.path.isfile(
                    target_scaler_path), f"{target_scaler_path=}"
                self.target_scaler = joblib.load(target_scaler_path)
                self._transform_targets()
            else:
                self.target_scaler = MinMaxScaler(feature_range=(-1, 1))
                self._fit_target_scaler()
                self._transform_targets()
                joblib.dump(self.target_scaler, constants.TARGET_SCALER_FN)
        else:
            self.target_scaler = None

    def _fit_scaler(self):
        all_features = np.concatenate(
            [features.numpy() for features in self.robot_features])
        self.feature_scaler.fit(all_features)

    def _transform_features(self):
        for i in range(len(self.robot_features)):
            self.robot_features[i] = torch.from_numpy(
                self.feature_scaler.transform(self.robot_features[i].numpy())
            ).float()

    def _fit_target_scaler(self):
        all_targets = np.concatenate([targets.numpy()
                                     for targets in self.force_targets])
        self.target_scaler.fit(all_targets)

    def _transform_targets(self):
        for i in range(len(self.force_targets)):
            self.force_targets[i] = torch.from_numpy(
                self.target_scaler.transform(self.force_targets[i].numpy())
            ).float()

    def __len__(self) -> int:
        return len(self.robot_features[0]) - 1

    def __getitem__(self, idx):
        end = idx+1
        start = max(0, end - self.seq_length)

        return {
            "features": self.robot_features[0][start:end],
            "target": self.force_targets[0][start:end]
        }
