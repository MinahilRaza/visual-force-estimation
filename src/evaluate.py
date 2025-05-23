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
from models.robot_state_transformer import RobotStateTransformer
from transforms import CropBottom
from dataset import VisionRobotDataset, SequentialDataset
import constants
import util
import signal_processing_utils as sp_utils


def parse_cmd_line() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--weights", required=True)
    parser.add_argument("-r", "--run", required=True, type=int)
    parser.add_argument("-m", "--model", required=True, type=str)
    parser.add_argument("--plot_type", choices=['plotly', 'matplotlib', 'none'], default='none',
                        help='Type of plot to generate: plotly, matplotlib or none')
    parser.add_argument("--pdf", action='store_true', default=False,
                        help='stores the plots as pdf instead of png')
    parser.add_argument('--use_acceleration',
                        action='store_true', default=False)
    parser.add_argument("--overfit", action='store_true', default=False)
    parser.add_argument('--state',
                        choices=['both', 'robot', 'vision', 'linear', 'conv'],
                        help='Set the model state: both for VISION_AND_ROBOT, robot for ROBOT_ONLY, vision for VISION_ONLY')
    parser.add_argument("--model_type", required=True,
                        choices=['vision_robot', 'transformer'])
    parser.add_argument("--seq_length", default=10, type=int, help="Length of the input sequences")
    parser.add_argument("--draw_crop_intervals", action='store_true', default=False,
                        help="Draw the crop intervals on the plots")
    parser.add_argument("--loss_criterion", type=str, default="mse",
                        help="Loss function to use: mse, rmse, l1, weighted_mse, mixed, custom")
    parser.add_argument("--analyze_results", action='store_true', default=True,    
                        help="Run analysis on the results, e.g. run-wise or peak-wise analysis")
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
    forces_gt = target_scaler.inverse_transform(forces_gt)
    avg_rmse = sp_utils.rmse(forces_gt, forces_pred)
    avg_nrmse = sp_utils.normalized_rmse(forces_gt, forces_pred)
    return forces_pred, forces_gt, avg_rmse, avg_nrmse


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

    checkpoint = torch.load(weights_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

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
                                 sequential=False,
                                 crop_runs=False,
                                 use_acceleration=args.use_acceleration)
        dataset = VisionRobotDataset(*data,
                                     path=path,
                                     img_transforms=constants.RES_NET_TEST_TRANSFORM,
                                     feature_scaler_path=constants.FEATURE_SCALER_FN)
    elif args.model_type == 'transformer':
        weights_dir = args.weights
        transformations_path = weights_dir.replace("weights/", "transformations/")
        feature_scaler_path = transformations_path + "/feature_scaler.joblib"
        target_scaler_path = transformations_path + "/target_scaler.joblib"

        features, targets, _, _ = util.load_dataset(path,
                                                    force_policy_runs=[
                                                        args.run],
                                                    no_force_policy_runs=[],
                                                    sequential=True,
                                                    crop_runs=False,
                                                    use_acceleration=args.use_acceleration)
        dataset = SequentialDataset(robot_features_list=features,
                                    force_targets_list=targets,
                                    normalize_targets=True,
                                    seq_length= args.seq_length,
                                    feature_scaler_path=feature_scaler_path,
                                    target_scaler_path=target_scaler_path)

    print(f"[INFO] Loaded Dataset with {len(dataset)} samples!")
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    target_scaler = joblib.load(target_scaler_path)
    forces_pred, forces_gt, avg_rmse, avg_nrmse = eval_model(
        model, data_loader, target_scaler, device, model_type=args.model_type)
    forces_pred_smooth = sp_utils.moving_average(
        forces_pred, window_size=constants.MOVING_AVG_WINDOW_SIZE)
    save_predictions("predictions", forces_pred, forces_pred_smooth, forces_gt)
    crop_intervals = None if not args.draw_crop_intervals else constants.START_END_TIMES["force_policy"][args.run] 
    
    if args.plot_type == 'matplotlib':
        sp_utils.plot_forces_matplotlib(forces_pred, forces_pred_smooth,
                forces_gt, avg_rmse, avg_nrmse, args.run,
                f'plots/{weights_path.split("/")[-3]}',
                crop_intervals=crop_intervals,
                loss_criterion=args.loss_criterion,
                pdf=args.pdf)
    elif args.plot_type == 'plotly':
        sp_utils.plot_forces_plotly(forces_pred, forces_pred_smooth,
                    forces_gt, avg_rmse, avg_nrmse, args.run,
                    f'plots/{weights_path.split("/")[-3]}',
                    crop_intervals=crop_intervals,
                    loss_criterion=args.loss_criterion)
    else:
        print("[INFO] No plots will be generated.")

    if args.analyze_results:

        print("[INFO] Running run-wise analysis on the results...")
        result_run = sp_utils.run_wise_analysis(forces_gt[:-1, :],
                                                forces_pred_smooth,
                                                args.run)
        sp_utils.save_results_to_csv(result_run, args.run,
                                 f'plots/{weights_path.split("/")[-3]}',
                                 peak_wise=False)
        
        print("[INFO] Running peak-wise analysis on the results...")
        result = sp_utils.peak_wise_analysis(forces_gt[:-1, :], 
                                              forces_pred_smooth,
                                              args.run)
        sp_utils.save_results_to_csv(result, args.run, 
                                 f'plots/{weights_path.split("/")[-3]}')

if __name__ == "__main__":
    eval()
