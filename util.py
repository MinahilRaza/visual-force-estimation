import os
from typing import List, Optional, Tuple, Dict
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import joblib

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

import torch
from torchvision import transforms
from dataset import VisionRobotDataset
from models.vision_robot_net import VRNConfig
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


def load_data(runs: dict[str, List[int]],
              data_dir: str,
              create_plots: bool = False,
              crop_runs: bool = True,
              use_acceleration: bool = False) -> Tuple[np.ndarray,
                                                       np.ndarray,
                                                       List[str],
                                                       List[str]]:
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
            if use_acceleration:
                excel_df = calculate_acceleration(excel_df)

            excel_df = excel_df.drop(constants.TIME_COLUMN, axis=1)
            X_cols = constants.FEATURE_COLUMS + constants.VELOCITY_COLUMNS
            if use_acceleration:
                X_cols += constants.ACCELERATION_COLUMNS

            if create_plots:
                forces_arr = excel_df[constants.TARGET_COLUMNS].to_numpy()
                plot_forces(forces_arr, run_nr=run, policy=policy, pdf=False)

            if crop_runs:
                for times in constants.START_END_TIMES[policy][run]:
                    start = times[0]
                    end = times[1]
                    actual_data_df = excel_df.iloc[start:end, :]

                    X = actual_data_df[X_cols].to_numpy()
                    y = actual_data_df[constants.TARGET_COLUMNS].to_numpy()
                    img_left_paths = get_img_paths("Left", actual_data_df)
                    img_right_paths = get_img_paths("Right", actual_data_df)

                    all_X.append(X)
                    all_y.append(y)
                    all_img_left_paths += img_left_paths
                    all_img_right_paths += img_right_paths
            else:
                X = excel_df[X_cols].to_numpy()
                y = excel_df[constants.TARGET_COLUMNS].to_numpy()
                img_left_paths = get_img_paths("Left", excel_df)
                img_right_paths = get_img_paths("Right", excel_df)

                all_X.append(X)
                all_y.append(y)
                all_img_left_paths += img_left_paths
                all_img_right_paths += img_right_paths

    all_X = np.concatenate(all_X, axis=0)
    all_y = np.concatenate(all_y, axis=0)

    return all_X, all_y, all_img_left_paths, all_img_right_paths


def calculate_acceleration(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure velocity is calculated first
    assert all(column in df.columns for column in constants.VELOCITY_COLUMNS)

    for nr in [1, 2]:
        for axis in ['x', 'y', 'z']:
            velocity_col = f'PSM{nr}_ee_v_{axis}'
            acceleration_col = f'PSM{nr}_ee_a_{axis}'
            set_acceleration(df, velocity_col, acceleration_col)
        for joint in range(1, 7):
            velocity_col = f'PSM{nr}_joint_{joint}_v'
            acceleration_col = f'PSM{nr}_joint_{joint}_a'
            set_acceleration(df, velocity_col, acceleration_col)
        velocity_col = f'PSM{nr}_jaw_angle_v'
        acceleration_col = f'PSM{nr}_jaw_angle_a'
        set_acceleration(df, velocity_col, acceleration_col)

    return df


def set_acceleration(df: pd.DataFrame, velocity_col: str, acceleration_col: str):
    df[acceleration_col] = df[velocity_col].diff() / df["Time (Seconds)"].diff()
    df.loc[df.index[0], acceleration_col] = 0
    assert len(df[acceleration_col]) == len(df[velocity_col])
    assert df[acceleration_col].isnull().sum() == 0


def calculate_velocity(df: pd.DataFrame) -> pd.DataFrame:
    for nr in [1, 2]:
        for axis in ['x', 'y', 'z']:
            position_col = f'PSM{nr}_ee_{axis}'
            velocity_col = f'PSM{nr}_ee_v_{axis}'
            set_velocity(df, position_col, velocity_col)
        for joint in range(1, 7):
            position_col = f'PSM{nr}_joint_{joint}'
            velocity_col = f'PSM{nr}_joint_{joint}_v'
            set_velocity(df, position_col, velocity_col)
        position_col = f'PSM{nr}_jaw_angle'
        velocity_col = f'PSM{nr}_jaw_angle_v'
        set_velocity(df, position_col, velocity_col)

    return df


def set_velocity(df: pd.DataFrame, position_col: str, velocity_col: str):
    df[velocity_col] = df[position_col].diff() / \
        df["Time (Seconds)"].diff()
    # first element is nan, as the velocity cannot be computed
    df.loc[df.index[0], velocity_col] = 0
    assert len(df[position_col]) == len(df[velocity_col])
    assert df[velocity_col].isnull().sum() == 0


def load_dataset(path: str,
                 force_policy_runs: List[int],
                 no_force_policy_runs: List[int],
                 create_plots: bool = False,
                 crop_runs: bool = True,
                 use_acceleration: bool = False) -> Tuple[np.ndarray,
                                                          np.ndarray,
                                                          List[str],
                                                          List[str]]:
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

    return load_data(runs, roll_out_dir, create_plots=create_plots, crop_runs=crop_runs, use_acceleration=use_acceleration)


def apply_scaling_to_datasets(train_dataset: VisionRobotDataset,
                              test_dataset: VisionRobotDataset,
                              normalize_targets: Optional[bool] = False) -> None:
    feature_scaler = StandardScaler()

    feature_scaler.fit(train_dataset.robot_features.numpy())

    train_dataset.robot_features = torch.from_numpy(
        feature_scaler.transform(train_dataset.robot_features.numpy())).float()
    test_dataset.robot_features = torch.from_numpy(
        feature_scaler.transform(test_dataset.robot_features.numpy())).float()

    # Save scaler to file to load it during eval
    joblib.dump(feature_scaler, constants.FEATURE_SCALER_FN)

    if normalize_targets:
        target_scaler = MinMaxScaler(feature_range=(-1, 1))
        target_scaler.fit(train_dataset.force_targets.numpy())

        train_dataset.force_targets = torch.from_numpy(
            target_scaler.transform(train_dataset.force_targets.numpy())).float()
        test_dataset.force_targets = torch.from_numpy(
            target_scaler.transform(test_dataset.force_targets.numpy())).float()

        # Save scaler to file to load it during eval
        joblib.dump(target_scaler, constants.TARGET_SCALER_FN)


def create_weights_path(model: str, num_epochs: int, base_dir: str = "weights") -> str:
    """
    Creates a directory path for saving weights with a unique run count and specified parameters.

    """
    assert isinstance(model, str)
    assert isinstance(num_epochs, int)
    if os.path.isdir(model):
        dir_name = model.split("/")[2]
        dir_name_split = dir_name.split("_")
        cnn_name = dir_name_split[2] if len(
            dir_name_split) > 1 else dir_name_split[0]
        model_name = f"pretrained_{cnn_name}"
    else:
        model_name = model

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

    run_name = f"run_{highest_count}_{model_name}_epochs_{num_epochs}"
    new_dir_path = base_path / run_name
    return str(new_dir_path)


def plot_forces(forces: np.ndarray, run_nr: int, policy: str, pdf: bool):
    assert forces.shape[1] == 3
    os.makedirs('plots', exist_ok=True)
    time_axis = np.arange(forces.shape[0])

    plt.figure()
    plt.plot(time_axis, forces[:, 0],
             label='x-axis', linestyle='-', marker='')
    plt.plot(time_axis, forces[:, 1],
             label='y-axis', linestyle='-', marker='')
    plt.plot(time_axis, forces[:, 2],
             label='z-axis', linestyle='-', marker='')
    policy_name = "Force Policy" if policy == "force_policy" else "No Force Policy"
    plt.title(f"{policy_name}, Run {run_nr}")
    plt.xlabel('Time')
    plt.ylabel('Force [N]')
    plt.legend()
    save_path = f"plots/rollout_{policy}_{run_nr}.{'pdf' if pdf else 'png'}"
    plt.savefig(save_path)
    plt.close()


def get_log_dir(args: argparse.Namespace) -> str:
    if os.path.isdir(args.model):
        cnn_name = args.model.split("/")[2]
        model_name = f"finetuned_{cnn_name}"
    else:
        model_name = args.model
    num_ep = args.num_epochs
    lr = args.lr
    batch_size = args.batch_size
    accel = int(args.use_acceleration)
    normalized = int(args.normalize_targets)
    pretrained = int(args.use_pretrained)
    scheduled = "_scheduled" if args.lr_scheduler else ""
    overfit = "_overfit" if args.overfit else ""
    log_dir = f"runs/force_est{overfit}_{model_name}_{num_ep}ep_lr_{lr}{scheduled}_bs_{batch_size}_accel_{accel}_normalized_{normalized}_pretrained_{pretrained}_real"
    return log_dir


def get_run_numbers(args: argparse.Namespace) -> dict[str, List[List[int]]]:
    train_runs = [args.force_runs, args.no_force_runs]
    test_runs = train_runs if args.overfit else constants.DEFAULT_TEST_RUNS
    return {"train": train_runs, "test": test_runs}


def get_num_robot_features(args: argparse.Namespace) -> int:
    if args.use_acceleration:
        return constants.NUM_ROBOT_FEATURES_INCL_ACCEL
    else:
        return constants.NUM_ROBOT_FEATURES


def get_image_transforms(args: argparse.Namespace) -> dict[str, transforms.Compose]:
    train_transform = constants.RES_NET_TEST_TRANSFORM if args.overfit else constants.RES_NET_TRAIN_TRANSFORM
    return {"train": train_transform,
            "test": constants.RES_NET_TEST_TRANSFORM}


def get_vrn_config(args: argparse.Namespace) -> VRNConfig:
    if args.overfit:
        return VRNConfig(cnn_model_version=args.model,
                         num_image_features=constants.NUM_IMAGE_FEATURES,
                         num_robot_features=get_num_robot_features(
                             args),
                         use_pretrained=args.use_pretrained,
                         dropout_rate=0.0,
                         use_batch_norm=False)
    return VRNConfig(cnn_model_version=args.model,
                     num_image_features=constants.NUM_IMAGE_FEATURES,
                     num_robot_features=get_num_robot_features(
                         args),
                     use_pretrained=args.use_pretrained,
                     dropout_rate=0.2,
                     use_batch_norm=True)


if __name__ == "__main__":
    all_X, all_y, all_img_left_paths, all_img_right_paths = load_dataset(
        path="data", force_policy_runs=[1, 2, 3, 4, 6], no_force_policy_runs=[1, 4], create_plots=False, )

    assert all_X.shape == (2549, 44), f"{all_X.shape=}"
    assert all_y.shape == (2549, 3)

    assert len(all_img_left_paths) == 2549
    assert len(all_img_right_paths) == 2549
