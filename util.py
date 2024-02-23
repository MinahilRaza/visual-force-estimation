import os
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset

FEATURE_COLUMS = ['PSM1_joint_1', 'PSM1_joint_2', 'PSM1_joint_3', 'PSM1_joint_4',
                  'PSM1_joint_5', 'PSM1_joint_6', 'PSM1_jaw_angle', 'PSM1_ee_x',
                  'PSM1_ee_y', 'PSM1_ee_z', 'PSM1_Orientation_Matrix_[1,1]',
                  'PSM1_Orientation_Matrix_[1,2]', 'PSM1_Orientation_Matrix_[1,3]',
                  'PSM1_Orientation_Matrix_[2,1]', 'PSM1_Orientation_Matrix_[2,2]',
                  'PSM1_Orientation_Matrix_[2,3]', 'PSM1_Orientation_Matrix_[3,1]',
                  'PSM1_Orientation_Matrix_[3,2]', 'PSM1_Orientation_Matrix_[3,3]',
                  'PSM2_joint_1', 'PSM2_joint_2', 'PSM2_joint_3', 'PSM2_joint_4',
                  'PSM2_joint_5', 'PSM2_joint_6', 'PSM2_jaw_angle', 'PSM2_ee_x',
                  'PSM2_ee_y', 'PSM2_ee_z', 'PSM2_Orientation_Matrix_[1,1]',
                  'PSM2_Orientation_Matrix_[1,2]', 'PSM2_Orientation_Matrix_[1,3]',
                  'PSM2_Orientation_Matrix_[2,1]', 'PSM2_Orientation_Matrix_[2,2]',
                  'PSM2_Orientation_Matrix_[2,3]', 'PSM2_Orientation_Matrix_[3,1]',
                  'PSM2_Orientation_Matrix_[3,2]', 'PSM2_Orientation_Matrix_[3,3]']

IMAGE_COLUMS = ['ZED Camera Left', 'ZED Camera Right']

TIME_COLUMN = ["Time (Seconds)"]

VELOCITY_COLUMNS = [f'PSM{nr}_ee_v_{axis}'
                    for axis in ['x', 'y', 'z'] for nr in [1, 2]]

TARGET_COLUMNS = ['Force_x_smooth', 'Force_y_smooth', 'Force_z_smooth']


def get_img_paths(cam: str, excel_df: pd.DataFrame) -> List[str]:
    assert cam in ["Left", "Right"], f"Invalid {cam}"

    col_name = f"ZED Camera {cam}"
    img_paths = []
    for server_path in excel_df[col_name].to_list():
        dirs = server_path.split("/")
        new_path = "/".join(dirs[-3:])
        img_paths.append(new_path)

    return img_paths


def load_data(excel_file_names: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    assert isinstance(excel_file_names, list)

    all_X = []
    all_y = []
    all_img_left_paths = []
    all_img_right_paths = []

    for excel_file_name in excel_file_names:
        print(f"Loading data: {excel_file_name}")
        relevant_cols = FEATURE_COLUMS + IMAGE_COLUMS + TARGET_COLUMNS + TIME_COLUMN
        excel_df = pd.read_excel(excel_file_name, usecols=relevant_cols)
        excel_df = calculate_velocity(excel_df)
        excel_df = excel_df.drop(TIME_COLUMN, axis=1)

        X = excel_df[FEATURE_COLUMS + VELOCITY_COLUMNS].to_numpy()
        y = excel_df[TARGET_COLUMNS].to_numpy()
        img_left_paths = get_img_paths("Left", excel_df)
        img_right_paths = get_img_paths("Right", excel_df)

        all_X.append(X)
        all_y.append(y)
        all_img_left_paths += img_left_paths
        all_img_right_paths += img_right_paths

    all_X = np.concatenate(all_X, axis=0)
    all_y = np.concatenate(all_y, axis=0)

    return all_X, all_y, all_img_left_paths, all_img_right_paths


def calculate_velocity(df: pd.DataFrame) -> pd.DataFrame:
    for axis in ['x', 'y', 'z']:
        for nr in [1, 2]:
            position_col = f'PSM{nr}_ee_{axis}'
            velocity_col = f'PSM{nr}_ee_v_{axis}'
            df[velocity_col] = df[position_col].diff() / \
                df["Time (Seconds)"].diff()
            # first element is nan, as the velocity cannot be computed
            df.loc[df.index[0], velocity_col] = 0
            assert len(df[position_col]) == len(df[velocity_col])
            assert df[velocity_col].isnull().sum() == 0

    return df


def load_dataset(path: str, run_nums: Optional[List[int]] = None):
    assert os.path.isdir(path), f"{path} is not a directory"
    assert os.path.exists(os.path.join(path, "images")), \
        f"{path} does not contain an images directory"
    assert os.path.exists(os.path.join(path, "roll_out")), \
        f"{path} does not contain a roll out directory"

    roll_out_dir = os.path.join(path, "roll_out")

    if run_nums is None:
        excel_files = [os.path.join(roll_out_dir, f)
                       for f in os.listdir(roll_out_dir)]
    else:
        assert isinstance(run_nums, list)
        assert len(run_nums) > 0
        excel_files = [
            f"{roll_out_dir}/dec6_force_no_TA_lastP_randomPosHeight_cs100_run{n}.xlsx" for n in run_nums]

    return load_data(excel_files)


def apply_scaling_to_datasets(train_dataset: Dataset, test_dataset: Dataset) -> None:
    scaler = StandardScaler()

    scaler.fit(train_dataset.robot_features.numpy())

    train_dataset.robot_features = torch.from_numpy(
        scaler.transform(train_dataset.robot_features.numpy())).float()
    test_dataset.robot_features = torch.from_numpy(
        scaler.transform(test_dataset.robot_features.numpy())).float()


def create_weights_path(lr: float, num_epochs: int, base_dir: str = "weights") -> str:
    """
    Creates a directory path for saving weights with a unique run count and specified parameters.

    """
    base_path = Path(base_dir)
    if not base_path.exists():
        base_path.mkdir(parents=True, exist_ok=True)
        highest_count = 0
    else:
        counts = []
        for dir_name in os.listdir(base_dir):
            if dir_name.startswith("run_"):
                parts = dir_name.split("_")
                if len(parts) >= 2 and parts[1].isdigit():
                    counts.append(int(parts[1]))
        highest_count = max(counts) + 1 if counts else 1

    new_dir_path = base_path / \
        f"run_{highest_count}_lr_{lr}_epochs_{num_epochs}"
    return str(new_dir_path)


if __name__ == "__main__":
    all_X, all_y, all_img_left_paths, all_img_right_paths = load_dataset(
        path="data/train", run_nums=[1, 2])

    assert all_X.shape == (2549, 44)
    assert all_y.shape == (2549, 3)

    assert len(all_img_left_paths) == 2549
    assert len(all_img_right_paths) == 2549
