import os
import argparse
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import util
import constants
from dataset import AutoEncoderDataset
from models.auto_encoder import ResNetAutoencoder
from trainer.trainer import AutoEncoderTrainer


def parse_cmd_line() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", required=True, type=int)
    parser.add_argument("--lr", required=True, type=float)
    parser.add_argument("--num_epochs", required=True, type=int)
    parser.add_argument("--base_model", required=True, type=str)
    return parser.parse_args()


def train():
    args = parse_cmd_line()
    data_transforms = {"train": constants.RES_NET_TEST_TRANSFORM,
                       "test": constants.RES_NET_TRAIN_TRANSFORM}
    # run_nums = {"train": [[1, 2, 3, 4, 6, 8, 9, 10], [1, 3]],
    #             "test": [[11, 13], [4]]}

    run_nums = {"train": [[1], []],
                "test": [[], [4]]}

    data_dir = "data"
    sets = ["train", "test"]
    data_loaders: dict[DataLoader] = {}

    for s in sets:
        _, _, img_left_paths, img_right_paths = util.load_dataset(
            data_dir, force_policy_runs=run_nums[s][0], no_force_policy_runs=run_nums[s][1], crop_runs=True)
        dataset = AutoEncoderDataset(
            img_left_paths=img_left_paths,
            img_right_paths=img_right_paths,
            transforms=data_transforms[s],
            path=data_dir)
        print(f"Loaded Dataset {s} with {len(dataset)} samples!")
        data_loaders[s] = DataLoader(
            dataset, batch_size=args.batch_size, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Using Device: {device}")
    print(f"[INFO] Base Model: {args.base_model}")

    model = ResNetAutoencoder(base_model=args.base_model, use_pretrained=True)
    model.to(device)

    weights_dir = util.create_weights_path(
        "res_net", args.num_epochs, base_dir="weights/auto_encoder")
    lr_scheduler_config = None
    writer = SummaryWriter()

    trainer = AutoEncoderTrainer(model,
                                 data_loaders,
                                 device,
                                 criterion="mse",
                                 lr=args.lr,
                                 regularized=True,
                                 weights_dir=weights_dir,
                                 writer=writer,
                                 lr_scheduler_config=lr_scheduler_config)
    trainer.train(num_epochs=args.num_epochs)
    encoder_state_dict = trainer.model.encoder.state_dict()
    encoder_weights_path = os.path.join(weights_dir, "encoder_weights.pth")
    torch.save(encoder_state_dict, encoder_weights_path)


if __name__ == "__main__":
    train()
