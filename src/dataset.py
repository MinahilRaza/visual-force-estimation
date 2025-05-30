from typing import List, Optional, Tuple
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

        if feature_scaler_path is not None: 
            if os.path.isfile(feature_scaler_path):
                self.feature_scaler = joblib.load(feature_scaler_path)
                print(f"Loading feature scaler from {feature_scaler_path}")
                for i in range(len(self.robot_features)):
                    self.robot_features[i] = torch.from_numpy(
                        self.feature_scaler.transform(
                            self.robot_features[i].numpy())
                    ).float()
            else:
                self.feature_scaler = StandardScaler()
                self._fit_scaler()
                self._transform_features()
                os.makedirs(os.path.dirname(feature_scaler_path), exist_ok=True)
                joblib.dump(self.feature_scaler, feature_scaler_path)
                print(f"Saving feature scaler to {feature_scaler_path}")
        else:
            self.feature_scaler = StandardScaler()
            self._fit_scaler()
            self._transform_features()
            joblib.dump(self.feature_scaler, constants.FEATURE_SCALER_FN)
            print(f"Saving feature scaler to {constants.FEATURE_SCALER_FN}")
            

        if normalize_targets:
            if target_scaler_path is not None:
                if os.path.isfile(target_scaler_path):
                    print(f"Loading target scaler from {target_scaler_path}")
                    self.target_scaler = joblib.load(target_scaler_path)
                    self._transform_targets()
                else:
                    self.target_scaler = MinMaxScaler(feature_range=(-1, 1))
                    self._fit_target_scaler()
                    self._transform_targets()
                    os.makedirs(os.path.dirname(target_scaler_path), exist_ok=True)
                    joblib.dump(self.target_scaler, target_scaler_path)
                    print(f"Saving target scaler to {target_scaler_path}")
            else:
                self.target_scaler = MinMaxScaler(feature_range=(-1, 1))
                self._fit_target_scaler()
                self._transform_targets()
                joblib.dump(self.target_scaler, constants.TARGET_SCALER_FN)
                print(f"Saving target scaler to {constants.TARGET_SCALER_FN}")

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

class SequentialVisionRobotDataset(Dataset):
    """
    Dataset class to handle sequences of left and right images, robot features, and force targets.
    This is used for training models that require temporal context, such as transformers.
    """

    def __init__(
        self,
        *,
        robot_features: np.ndarray,        # Robot features defined in constants.py (N, S)
        force_targets: np.ndarray,         # 3 dimentional forces (N, 3)
        img_left_paths: List[str],
        img_right_paths: List[str],
        path: str,
        img_transforms: Optional[transforms.Compose] = None,
        seq_length: int = 1,
        feature_scaler_path: Optional[str] = None,
        target_scaler_path: Optional[str] = None) -> None:
        super().__init__()
        assert len(robot_features) == len(img_right_paths) == len(force_targets), "sample mismatch"
        self.seq_length = seq_length
        self.root = Path(path)
        self.transforms = img_transforms

        if feature_scaler_path and Path(feature_scaler_path).is_file():
            fscaler: StandardScaler = joblib.load(feature_scaler_path)
            self.robot_features = torch.from_numpy(fscaler.transform(robot_features)).float()
        else:
            self.robot_features = torch.from_numpy(robot_features).float()

        if target_scaler_path and Path(target_scaler_path).is_file():
            tscaler: MinMaxScaler = joblib.load(target_scaler_path)
            self.force_targets = torch.from_numpy(tscaler.transform(force_targets)).float()
        else:
            self.force_targets = torch.from_numpy(force_targets).float()

        self.img_left_paths = img_left_paths
        self.img_right_paths = img_right_paths
        self.N = len(img_right_paths)

    def __len__(self) -> int: 
        return self.N - self.seq_length + 1 if self.seq_length > 1 else self.N

    def _load_pair(self, idx: int) -> Tuple[Image.Image, Image.Image]:
        l = Image.open(self.root / self.img_left_paths[idx]).convert("RGB")
        r = Image.open(self.root / self.img_right_paths[idx]).convert("RGB")
        return l, r
    
    def _load_left(self, idx: int) -> Image.Image:
        img = Image.open(self.root / self.img_left_paths[idx]).convert("RGB")
        return img
    
    def _load_right(self, idx: int) -> Image.Image:
        img = Image.open(self.root / self.img_right_paths[idx]).convert("RGB")
        return img

    def __getitem__(self, idx):
        if self.seq_length == 1:
            img_l, img_r = self._load_pair(idx)
            if self.transforms:
                img_l, img_r = self.transforms(img_l), self.transforms(img_r)
            return {
                "img_left": img_l,
                "img_right": img_r,
                "features": self.robot_features[idx],
                "target": self.force_targets[idx],
            }

        # Load a sequence window with T frames)
        idxs = range(idx, idx + self.seq_length)

        if len(self.img_left_paths) > 0:
            imgs_l = [self._load_left(i) for i in idxs]
            if self.transforms:
                imgs_l = [self.transforms(im) for im in imgs_l]
            imgs_l_tensor = torch.stack(imgs_l) # (T, C, H, W)

        else:
            imgs_l_tensor = None

        if len(self.img_right_paths) > 0:
            imgs_r = [self._load_right(i) for i in idxs]
            if self.transforms:
                imgs_r = [self.transforms(im) for im in imgs_r]
            imgs_r_tensor = torch.stack(imgs_r) # (T, C, H, W)
        else:
            imgs_r_tensor = None

        return {
            "img_left": imgs_l_tensor,
            "img_right": imgs_r_tensor,
            "features": self.robot_features[idx : idx + self.seq_length],   # (T, S)
            "target": self.force_targets[idx : idx + self.seq_length],      # (T, 3)
        }

def custom_collate_fn(batch):
    """
    Stacks dict fields to a single batch.
    Works for both single‑frame and sequence inputs.

    Returns
    -------
    dict with keys:
        img_right  – (B, T, C, H, W) or (B, C, H, W)
        forces     – (B, 3)   (last step if sequence)
        robot_state– (B, T, S) or (B, S)
    """
    imgs = torch.stack([b["img_right"] for b in batch])

    forces_lst = [b["target"] for b in batch]          # (T,3) or (3,)
    if forces_lst[0].dim() == 2:                       # sequence → pick last
        forces = torch.stack([f[-1] for f in forces_lst])
    else:
        forces = torch.stack(forces_lst)

    robot_state = torch.stack([b["features"] for b in batch])
    return {"img_right": imgs, "forces": forces, "robot_state": robot_state}