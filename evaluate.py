import argparse
import os
import joblib
from tqdm import tqdm
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from models.vision_robot_net import VisionRobotNet
from transforms import CropBottom
from dataset import VisionRobotDataset
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
    return parser.parse_args()


def save_predictions(dir: str, forces_pred: np.ndarray, forces_gt: np.ndarray):
    if not os.path.exists(dir):
        os.makedirs(dir)
    pred_file = os.path.join(dir, "predicted_forces.txt")
    gt_file = os.path.join(dir, "true_forces.txt")
    for write_file, force_array in zip([pred_file, gt_file], [forces_pred, forces_gt]):
        with open(write_file, 'w', encoding='utf-8') as file:
            file.write("F_X,F_Y,F_Z\n")
            for force in force_array:
                line = "{},{},{}\n".format(force[0], force[1], force[2])
                file.write(line)


def plot_forces(forces_pred: np.ndarray, forces_gt: np.ndarray, run: int, pdf: bool):
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
        title = f"Force in {ax} Direction, Run {run}"
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Force [N]')
        plt.ylim(-1, 1)
        plt.legend()
        save_path = f"plots/pred_run_{run}_force_{ax}.{'pdf' if pdf else 'png'}"
        plt.savefig(save_path)
        plt.close()


@ torch.no_grad()
def eval_model(model: VisionRobotNet, data_loader: DataLoader, target_scaler: MinMaxScaler, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    n_samples = len(data_loader.dataset)
    batch_size = data_loader.batch_size
    forces_pred = torch.zeros((n_samples, 3), device=device)
    forces_gt = torch.zeros((n_samples, 3), device=device)
    i = 0
    for batch in tqdm(data_loader):
        img_left = batch["img_left"].to(device)
        img_right = batch["img_right"].to(device)
        features = batch["features"].to(device)
        target = batch["target"].to(device)

        out: torch.Tensor = model(img_left, img_right, features)
        len_batch = batch["img_left"].size(0)
        forces_pred[i*batch_size: i*batch_size + len_batch] = out
        forces_gt[i * batch_size: i * batch_size + len_batch] = target
        i += 1
    forces_pred = forces_pred.cpu().detach().numpy()
    forces_pred = target_scaler.inverse_transform(forces_pred)
    forces_gt = forces_gt.cpu().detach().numpy()
    return forces_pred, forces_gt


def eval() -> None:
    args = parse_cmd_line()

    if os.path.isdir(args.weights):
        weights_path = os.path.join(args.weights, "best_params.pth")
    else:
        weights_path = args.weights
    if not os.path.exists(weights_path) or not os.path.isfile(weights_path):
        raise ValueError(f"Invalid weights: {weights_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_config = util.get_vrn_config(args)
    model = VisionRobotNet(model_config)
    model.load_state_dict(torch.load(weights_path))

    model.to(device)
    model.eval()
    print(f"[INFO] Loaded model from: {weights_path}")
    print(f"[INFO] Using Device: {device}")

    batch_size = 8

    path = "data"
    data = util.load_dataset(path, force_policy_runs=[
        args.run], no_force_policy_runs=[], crop_runs=False, use_acceleration=args.use_acceleration)
    dataset = VisionRobotDataset(
        *data,
        path=path,
        img_transforms=constants.RES_NET_TEST_TRANSFORM,
        feature_scaler_path=constants.FEATURE_SCALER_FN)

    print(f"[INFO] Loaded Dataset with {len(dataset)} samples!")
    data_loader = DataLoader(dataset, batch_size=batch_size)
    target_scaler = joblib.load(constants.TARGET_SCALER_FN)
    forces_pred, forces_gt = eval_model(
        model, data_loader, target_scaler, device)
    save_predictions("predictions", forces_pred, forces_gt)
    plot_forces(forces_pred, forces_gt, args.run, args.pdf)


if __name__ == "__main__":
    eval()
