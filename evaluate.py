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
from util import load_dataset
import constants


def parse_cmd_line() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--weights", required=True)
    parser.add_argument("-r", "--run", required=True, type=int)
    parser.add_argument("--pdf", action='store_true', default=False,
                        help='stores the plots as pdf instead of png')
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


def eval(args: argparse.Namespace) -> None:
    weights_path = args.weights
    if not os.path.exists(weights_path) or not os.path.isfile(weights_path):
        raise Warning(f"Invalid weights file: {weights_path}")
    model = VisionRobotNet(cnn_model_version=constants.CNN_MODEL_VERSION,
                           num_image_features=constants.NUM_IMAGE_FEATURES,
                           num_robot_features=constants.NUM_ROBOT_FEATURES)
    model.eval()
    model.load_state_dict(torch.load(weights_path))
    print(f"[INFO] Loaded model from: {weights_path}")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"[INFO] Using Device: {device}")
    model.to(device)

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        CropBottom((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    batch_size = 8

    path = "data"
    data = load_dataset(path, force_policy_runs=[
                        args.run], no_force_policy_runs=[])
    dataset = VisionRobotDataset(
        *data, path=path, transforms=val_transform)
    print(f"Loaded Dataset with {len(dataset)} samples!")
    data_loader = DataLoader(dataset, batch_size=batch_size)
    forces_pred, forces_gt = eval_model(model, data_loader, device)
    save_predictions("predictions", forces_pred, forces_gt)
    plot_forces(forces_pred, forces_gt, args.pdf)


if __name__ == "__main__":
    args = parse_cmd_line()
    eval(args)
