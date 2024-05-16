import os
import torch
import argparse
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.robot_state_transformer import RobotStateTransformer, TransformerConfig, EncoderState
from trainer.trainer import TransformerTrainer, LRSchedulerConfig
from dataset import SequentialDataset
import util
import constants


def parse_cmd_line() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", required=True, type=int)
    parser.add_argument("--lr", required=True, type=float)
    parser.add_argument("--num_epochs", required=True, type=int)
    parser.add_argument('--force_runs', nargs='+', type=int, required=True)
    parser.add_argument('--no_force_runs', nargs='*', type=int, default=[])
    parser.add_argument('--lr_scheduler', action='store_true', default=False)
    parser.add_argument('--use_acceleration',
                        action='store_true', default=False)
    parser.add_argument('--normalize_targets',
                        action='store_true', default=False)
    parser.add_argument("--out_dir", default=None, type=str)
    parser.add_argument("--overfit", action='store_true', default=False)
    parser.add_argument("--seq_length", required=True,
                        type=int, help="Length of the input sequences")

    return parser.parse_args()


def train():
    args = parse_cmd_line()
    args.model = "transformer"
    args.use_pretrained = False
    args.state = "linear"

    print("Training Robot State Transformer Network")

    run_nums = util.get_run_numbers(args)
    log_dir = util.get_log_dir(args)
    hparams = {
        'batch_size': args.batch_size,
        'lr': args.lr,
        'num_epochs': args.num_epochs,
        'seq_length': args.seq_length
    }
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_hparams(hparams, {})

    data_dir = "data"
    sets = ["train", "test"]
    data_loaders = {}

    for s in sets:
        features, targets, _, _ = util.load_dataset(path=data_dir,
                                                    force_policy_runs=run_nums[s][0],
                                                    no_force_policy_runs=run_nums[s][1],
                                                    use_acceleration=args.use_acceleration,
                                                    crop_runs=False)
        dataset = SequentialDataset(robot_features=features,
                                    force_targets=targets,
                                    seq_length=args.seq_length)
        print(
            f"[INFO] Loaded Sequential Dataset {s} with {len(dataset)} samples!")
        data_loaders[s] = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=(s == "train"), drop_last=True)

    util.apply_scaling_to_datasets(
        data_loaders["train"].dataset,
        data_loaders["test"].dataset,
        normalize_targets=args.normalize_targets)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = util.get_transformer_config(args)
    model = RobotStateTransformer(config)
    model.to(device)

    print(f"[INFO] Using Device: {device}")
    print(f"[INFO] Training Model with Seq Length: {constants.SEQ_LENGTH}")
    print(f"[INFO] Batch Size: {args.batch_size}")
    print(f"[INFO] Learning Rate: {args.lr}")

    weights_dir = util.create_weights_path(
        "robot_state_transformer", args.num_epochs) if not args.out_dir else args.out_dir
    lr_scheduler_config = LRSchedulerConfig() if args.lr_scheduler else None
    trainer = TransformerTrainer(model,
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


if __name__ == "__main__":
    train()
