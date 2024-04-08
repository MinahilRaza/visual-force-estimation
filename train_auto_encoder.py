import os
import socket
import argparse
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import util
import constants
from dataset import AutoEncoderDataset
from models.auto_encoder import ResNetAutoencoder
from models.var_auto_encoder import VarAutoEncoder
from trainer.trainer import AutoEncoderTrainer, VarAutoEncoderTrainer


def parse_cmd_line() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", required=True, type=int)
    parser.add_argument("--lr", required=True, type=float)
    parser.add_argument("--num_epochs", required=True, type=int)
    parser.add_argument("--base_model", required=True, type=str)
    parser.add_argument("--out_dir", default=None, type=str)
    return parser.parse_args()


def train():
    print("Training Auto Encoder Network")
    args = parse_cmd_line()
    data_transforms = {"train": constants.RES_NET_TEST_TRANSFORM,
                       "test": constants.RES_NET_TRAIN_TRANSFORM}

    if socket.gethostname() == "Tim-ThinkPad":
        run_nums = {"train": [[1, 2], [1]],
                    "test": [[], [4]]}
    else:
        run_nums = {"train": [[1, 2, 3, 4, 6, 8, 9], [1, 4]],
                    "test": [[10, 11], []]}

    data_dir = "data"
    sets = ["train", "test"]
    data_loaders: dict[str, DataLoader] = {}

    for s in sets:
        _, _, img_left_paths, img_right_paths = util.load_dataset(
            data_dir, force_policy_runs=run_nums[s][0], no_force_policy_runs=run_nums[s][1], crop_runs=True)
        dataset = AutoEncoderDataset(
            img_left_paths=img_left_paths,
            img_right_paths=img_right_paths,
            transforms=data_transforms[s],
            path=data_dir)
        print(f"[INFO] Loaded Dataset {s} with {len(dataset)} images!")
        data_loaders[s] = DataLoader(
            dataset, batch_size=args.batch_size, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Using Device: {device}")
    print(f"[INFO] Base Model: {args.base_model}")
    print(f"[INFO] Batch Size: {args.batch_size}")
    print(f"[INFO] Learning rate: {args.lr}")

    # model = ResNetAutoencoder(base_model=args.base_model, use_pretrained=True)
    model = VarAutoEncoder(enc_dim=constants.NUM_IMAGE_FEATURES)
    model.to(device)

    weights_dir = util.create_weights_path(
        args.base_model, args.num_epochs, base_dir="weights/auto_encoder") if not args.out_dir else args.out_dir
    lr_scheduler_config = None
    writer = SummaryWriter()

    trainer = VarAutoEncoderTrainer(model,
                                    data_loaders,
                                    device,
                                    criterion="mse",
                                    lr=args.lr,
                                    regularized=True,
                                    weights_dir=weights_dir,
                                    writer=writer,
                                    lr_scheduler_config=lr_scheduler_config,
                                    use_acceleration=args.use_acceleration)
    trainer.train(num_epochs=args.num_epochs)
    encoder_state_dict = trainer.model.encoder.state_dict()
    encoder_weights_path = os.path.join(
        weights_dir, constants.ENCODER_WEIGHTS_FN)
    torch.save(encoder_state_dict, encoder_weights_path)


if __name__ == "__main__":
    train()
