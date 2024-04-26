import argparse
import os
from tqdm import tqdm
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt

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


def plot_forces(forces_pred: np.ndarray, forces_gt: np.ndarray, pdf: bool):
    assert forces_pred.shape == forces_gt.shape
    assert forces_pred.shape[1] == 3

    if not os.path.exists('plots'):
        os.makedirs('plots')

    time_axis = np.arange(forces_pred.shape[0])

    titles = ['Force in X Direction',
              'Force in Y Direction', 'Force in Z Direction']
    y_labels = ['Force X (N)', 'Force Y (N)', 'Force Z (N)']

    for i in range(3):
        plt.figure()
        plt.plot(time_axis, forces_pred[:, i],
                 label='Predicted', linestyle='-', marker='')
        plt.plot(time_axis, forces_gt[:, i],
                 label='Ground Truth', linestyle='-', marker='')
        plt.title(titles[i])
        plt.xlabel('Time')
        plt.ylabel(y_labels[i])
        plt.legend()
        save_path = f"plots/force_{i}.{'pdf' if pdf else 'png'}"
        plt.savefig(save_path)
        plt.close()


@torch.no_grad()
def eval_model(model: VisionRobotNet, data_loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
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
    forces_pred = forces_pred.to("cpu").numpy()
    forces_gt = forces_gt.to("cpu").numpy()
    return forces_pred, forces_gt


def eval() -> None:
    args = parse_cmd_line()

    if os.path.isdir(args.weights):
        weights_path = os.path.join(args.weights, "best_params.pth")
    else:
        weights_path = args.weights
    if not os.path.exists(weights_path) or not os.path.isfile(weights_path):
        raise Warning(f"Invalid weights: {weights_path}")

    model = VisionRobotNet(cnn_model_version=args.model,
                           num_image_features=constants.NUM_IMAGE_FEATURES,
                           num_robot_features=util.get_num_robot_features(
                               args),
                           use_pretrained=False)
    model.eval()
    model.load_state_dict(torch.load(weights_path))
    print(f"[INFO] Loaded model from: {weights_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Using Device: {device}")
    model.to(device)

    batch_size = 8

    path = "data"
    data = util.load_dataset(path, force_policy_runs=[
        args.run], no_force_policy_runs=[], crop_runs=False, use_acceleration=args.use_acceleration)
    dataset = VisionRobotDataset(
        *data, path=path, transforms=constants.RES_NET_TEST_TRANSFORM)
    print(f"[INFO] Loaded Dataset with {len(dataset)} samples!")
    data_loader = DataLoader(dataset, batch_size=batch_size)
    forces_pred, forces_gt = eval_model(model, data_loader, device)
    save_predictions("predictions", forces_pred, forces_gt)
    plot_forces(forces_pred, forces_gt, args.pdf)


if __name__ == "__main__":
    eval()
