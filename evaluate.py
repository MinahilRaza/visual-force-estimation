import argparse
import os
import joblib
from tqdm import tqdm
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from models.vision_robot_net import VisionRobotNet
from models.robot_state_transformer import RobotStateTransformer
from transforms import CropBottom
from dataset import VisionRobotDataset, SequentialDataset
import constants
import util


def parse_cmd_line() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--weights", required=True)
    parser.add_argument("-r", "--run", required=True, type=int)
    parser.add_argument("-m", "--model", required=True, type=str)
    parser.add_argument("--pdf", action='store_true', default=False,
                        help='stores the plots as pdf instead of png')
    parser.add_argument('--use_acceleration',
                        action='store_true', default=False)
    parser.add_argument("--overfit", action='store_true', default=False)
    parser.add_argument('--state',
                        choices=['both', 'robot', 'vision'],
                        help='Set the model state: both for VISION_AND_ROBOT, robot for ROBOT_ONLY, vision for VISION_ONLY')
    parser.add_argument("--model_type", required=True,
                        choices=['vision_robot', 'transformer'])

    return parser.parse_args()


def save_predictions(dir: str, forces_pred: np.ndarray, forces_smooth: np.ndarray, forces_gt: np.ndarray):
    os.makedirs(dir, exist_ok=True)
    pred_file = os.path.join(dir, "predicted_forces.txt")
    gt_file = os.path.join(dir, "true_forces.txt")
    smooth_file = os.path.join(dir, "smoothed_forces.txt")
    files = [pred_file, smooth_file, gt_file]
    forces = [forces_pred, forces_smooth, forces_gt]

    for write_file, force_array in zip(files, forces):
        with open(write_file, 'w', encoding='utf-8') as file:
            file.write("F_X,F_Y,F_Z\n")
            for force in force_array:
                line = "{},{},{}\n".format(force[0], force[1], force[2])
                file.write(line)


def moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
    """
    Computes the moving average for each column of data separately.

    Parameters:
    data (np.ndarray): A 2D array where each column represents a series of data points.
    window_size (int): The number of data points in each moving average window.

    Returns:
    np.ndarray: A 2D array with the same shape as data, containing the moving averages.
    """
    if window_size > data.shape[0]:
        raise ValueError(
            "window_size is larger than the number of rows in data.")

    zero_pad = np.zeros((window_size-1, data.shape[1]))
    data_padded = np.insert(data, 0, zero_pad, axis=0)
    cumsum_vec = np.cumsum(data_padded, axis=0)
    moving_avg = (cumsum_vec[window_size:] -
                  cumsum_vec[:-window_size]) / window_size

    assert moving_avg.shape[1] == data.shape[
        1], f"The output shape {moving_avg.shape} does not match the input shape {data.shape}"
    return moving_avg


def plot_forces(forces_pred: np.ndarray,
                forces_smooth: np.ndarray,
                forces_gt: np.ndarray,
                avg_rmse: float,
                run: int,
                pdf: bool):
    assert forces_pred.shape == forces_gt.shape
    assert forces_pred.shape[1] == 3

    os.makedirs('plots', exist_ok=True)

    time_axis = np.arange(forces_pred.shape[0])

    axes = ["X", "Y", "Z"]

    for i, ax in enumerate(axes):
        plt.figure()
        plt.plot(time_axis, forces_pred[:, i],
                 label='Predicted', linestyle='-', marker='')
        plt.plot(time_axis, forces_gt[:, i],
                 label='Ground Truth', linestyle='-', marker='')
        title = f"Force in {ax} Direction, Run {run}, Avg RMSE: {avg_rmse:.4f}"
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Force [N]')
        plt.ylim(-1, 1)
        plt.legend()
        save_path = f"plots/pred_run_{run}_force_{ax}.{'pdf' if pdf else 'png'}"
        plt.savefig(save_path)
        plt.close()

    for i, ax in enumerate(axes):
        plt.figure()
        plt.plot(
            time_axis[:-1], forces_smooth[:, i], label='Smoothed Predictions', linestyle='-', marker='')
        plt.plot(time_axis[:-1], forces_gt[:-1, i],
                 label='Ground Truth', linestyle='-', marker='')
        title = f"Force in {ax} Direction, Run {run}"
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Force [N]')
        plt.ylim(-1, 1)
        plt.legend()
        save_path = f"plots/pred_smooth_run_{run}_force_{ax}.{'pdf' if pdf else 'png'}"
        plt.savefig(save_path)
        plt.close()


@torch.no_grad()
def eval_model(model: VisionRobotNet,
               data_loader: DataLoader,
               target_scaler: MinMaxScaler,
               device: torch.device,
               model_type: str) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    n_samples = len(data_loader.dataset)
    batch_size = data_loader.batch_size
    forces_pred = torch.zeros((n_samples, 3), device=device)
    forces_gt = torch.zeros((n_samples, 3), device=device)
    i = 0
    for batch in tqdm(data_loader):
        if model_type == 'vision_robot':
            img_left = batch["img_left"].to(device)
            img_right = batch["img_right"].to(device)
            features = batch["features"].to(device)
            target = batch["target"].to(device)
            out: torch.Tensor = model(img_left, img_right, features)
            len_batch = target.size(0)
            forces_pred[i*batch_size: i*batch_size + len_batch] = out
            forces_gt[i * batch_size: i * batch_size + len_batch] = target

        elif model_type == 'transformer':
            robot_state = batch["features"].to(device)
            target = batch["target"].to(device)
            # Shape: [batch_size, seq_length, 3]
            out: torch.Tensor = model(robot_state)
            len_batch = target.size(0)
            # Take the last value in the sequence of predictions
            forces_pred[i*batch_size: i*batch_size + len_batch] = out[:, -1, :]
            forces_gt[i * batch_size: i * batch_size +
                      len_batch] = target[:, -1, :]
        i += 1
    forces_pred = forces_pred.cpu().detach().numpy()
    forces_pred = target_scaler.inverse_transform(forces_pred)
    forces_gt = forces_gt.cpu().detach().numpy()
    avg_rmse = np.sqrt(mean_squared_error(
        forces_gt, forces_pred, multioutput='uniform_average'))
    return forces_pred, forces_gt, avg_rmse


def eval() -> None:
    args = parse_cmd_line()
    args.use_pretrained = False

    if os.path.isdir(args.weights):
        weights_path = os.path.join(args.weights, "best_params.pth")
    else:
        weights_path = args.weights
    if not os.path.exists(weights_path) or not os.path.isfile(weights_path):
        raise ValueError(f"Invalid weights: {weights_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_type == 'vision_robot':
        model_config = util.get_vrn_config(args)
        model = VisionRobotNet(model_config)
    elif args.model_type == 'transformer':
        config = util.get_transformer_config(args)
        model = RobotStateTransformer(config)

    model.load_state_dict(torch.load(weights_path))

    model.to(device)
    model.eval()
    print(f"[INFO] Loaded model from: {weights_path}")
    print(f"[INFO] Using Device: {device}")

    batch_size = 32
    path = "data"

    if args.model_type == 'vision_robot':
        data = util.load_dataset(path,
                                 force_policy_runs=[args.run],
                                 no_force_policy_runs=[],
                                 crop_runs=False,
                                 use_acceleration=args.use_acceleration)
        dataset = VisionRobotDataset(*data,
                                     path=path,
                                     img_transforms=constants.RES_NET_TEST_TRANSFORM,
                                     feature_scaler_path=constants.FEATURE_SCALER_FN)
    elif args.model_type == 'transformer':

        features, targets, _, _ = util.load_dataset(path,
                                                    force_policy_runs=[
                                                        args.run],
                                                    no_force_policy_runs=[],
                                                    crop_runs=False,
                                                    use_acceleration=args.use_acceleration)
        dataset = SequentialDataset(robot_features=features,
                                    force_targets=targets,
                                    seq_length=constants.SEQ_LENGTH,
                                    feature_scaler_path=constants.FEATURE_SCALER_FN)

    print(f"[INFO] Loaded Dataset with {len(dataset)} samples!")
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    target_scaler = joblib.load(constants.TARGET_SCALER_FN)
    forces_pred, forces_gt, avg_rmse = eval_model(
        model, data_loader, target_scaler, device, model_type=args.model_type)
    forces_pred_smooth = moving_average(
        forces_pred, window_size=constants.MOVING_AVG_WINDOW_SIZE)
    save_predictions("predictions", forces_pred, forces_pred_smooth, forces_gt)
    plot_forces(forces_pred, forces_pred_smooth,
                forces_gt, avg_rmse, args.run, args.pdf)


if __name__ == "__main__":
    eval()
