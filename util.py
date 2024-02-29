import os
from typing import List, Optional, Tuple, Dict
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset

import constants


def get_img_paths(cam: str, excel_df: pd.DataFrame) -> List[str]:
    assert cam in ["Left", "Right"], f"Invalid {cam}"

    col_name = f"ZED Camera {cam}"
    img_paths = []
    for server_path in excel_df[col_name].to_list():
        dirs = server_path.split("/")
        new_path = "/".join(dirs[-3:])
        img_paths.append(new_path)

    return img_paths


def load_data(runs: dict[str, List[int]], data_dir: str, plot_forces: bool = False) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    assert isinstance(runs, dict), f"{runs=}"

    all_X = []
    all_y = []
    all_img_left_paths = []
    all_img_right_paths = []

    relevant_cols = constants.FEATURE_COLUMS + constants.IMAGE_COLUMS + \
        constants.TARGET_COLUMNS + constants.TIME_COLUMN

    for policy, runs in runs.items():
        for run in runs:
            print(f"Loading data for run {run} of policy {policy}")
            excel_file_name = constants.EXCEL_FILE_NAMES[policy][run]
            excel_file_path = os.path.join(data_dir, excel_file_name)
            excel_df = pd.read_excel(excel_file_path, usecols=relevant_cols)
            excel_df = calculate_velocity(excel_df)
            excel_df = excel_df.drop(constants.TIME_COLUMN, axis=1)

            if plot_forces:
                forces_arr = excel_df[constants.TARGET_COLUMNS].to_numpy()
                plot_forces(forces_arr, run_nr=run, policy=policy, pdf=True)

            for times in constants.START_END_TIMES[policy][run]:
                start = times[0]
                end = times[1]
                actual_data_df = excel_df.iloc[start:end, :]

                X = actual_data_df[constants.FEATURE_COLUMS +
                                   constants.VELOCITY_COLUMNS].to_numpy()
                y = actual_data_df[constants.TARGET_COLUMNS].to_numpy()
                img_left_paths = get_img_paths("Left", actual_data_df)
                img_right_paths = get_img_paths("Right", actual_data_df)

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


def load_dataset(path: str, force_policy_runs: List[int], no_force_policy_runs: List[int]):
    assert os.path.isdir(path), f"{path} is not a directory"
    assert os.path.exists(os.path.join(path, "images")), \
        f"{path} does not contain an images directory"
    assert os.path.exists(os.path.join(path, "roll_out")), \
        f"{path} does not contain a roll out directory"

    roll_out_dir = os.path.join(path, "roll_out")

    runs = {
        "force_policy": force_policy_runs,
        "no_force_policy": no_force_policy_runs
    }

    return load_data(runs, roll_out_dir)


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

    run_name = f"run_{highest_count}_lr_{lr}_epochs_{num_epochs}"
    new_dir_path = base_path / run_name
    return str(new_dir_path)


def plot_forces(forces: np.ndarray, run_nr: int, policy: str, pdf: bool):
    assert forces.shape[1] == 3
    os.makedirs('plots', exist_ok=True)
    time_axis = np.arange(forces.shape[0])

    plt.figure()
    plt.plot(time_axis, forces[:, 0],
             label='X', linestyle='-', marker='')
    plt.plot(time_axis, forces[:, 1],
             label='X', linestyle='-', marker='')
    plt.plot(time_axis, forces[:, 2],
             label='X', linestyle='-', marker='')
    policy_name = "Force Policy" if policy == "force_policy" else "No Force Policy"
    plt.title(f"{policy_name}, Run {run_nr}")
    plt.xlabel('Time')
    plt.ylabel('Force [N]')
    plt.legend()
    save_path = f"plots/rollout_{policy}_{run_nr}.{'pdf' if pdf else 'png'}"
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    data = load_dataset(path="data/test",
                        force_policy_runs=[11], no_force_policy_runs=[1])
    exit(0)
    all_X, all_y, all_img_left_paths, all_img_right_paths = load_dataset(
        path="data/train", force_policy_runs=[1, 2, 3, 4, 8, 9, 10], no_force_policy_runs=[4])

    assert all_X.shape == (2549, 44), f"{all_X.shape=}"
    assert all_y.shape == (2549, 3)

    assert len(all_img_left_paths) == 2549
    assert len(all_img_right_paths) == 2549
