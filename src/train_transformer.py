import torch
import argparse
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from models.robot_state_transformer import RobotStateTransformer
from trainer.trainer import TransformerTrainer, LRSchedulerConfig
from dataset import SequentialDataset

import util
import constants
from datetime import datetime
import wandb
wandb.login()


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
    parser.add_argument('--state',
                        choices=['linear', 'conv'],
                        required=True,
                        help='Set the model state: linear for using a linear feature extractor, conv for a Conv1 Layer'
                        )
    parser.add_argument("--use_kfold", action='store_true', default=False,
                        help="Enable k-fold cross-validation")
    parser.add_argument("--k_folds", type=int, default=5,
                        help="Number of folds for k-fold cross-validation")

    return parser.parse_args()


def train():
    """
    Train a Robot State Transformer model.

    This function trains a Robot State Transformer model based on the command line arguments.

    The following hyperparameters can be set through command line arguments:

    - batch_size: The batch size to use.
    - lr: The learning rate to use.
    - num_epochs: The number of epochs to train for.
    - seq_length: The length of the input sequences.
    - state: The model state to use, either 'linear' for a linear feature extractor or 'conv' for a Conv1 Layer feature extractor.
    - force_runs: The runs to use for training.
    - no_force_runs: The runs not to use for training.
    - lr_scheduler: Whether or not to use a learning rate scheduler.
    - use_acceleration: Whether or not to use acceleration as an input feature.
    - normalize_targets: Whether or not to normalize the targets.
    - out_dir: The directory to save the trained model weights in. If not specified, will save in
        "weights/robot_state_transformer/<num_epochs>_epochs_lr_<lr>".
    - overfit: If set, will only use one batch for training and testing.

    The function will print out various information about the training process, including the batch size, learning rate,
    number of epochs, sequence length, model state, and whether or not acceleration is being used.

    The function will also save the trained model weights in the specified directory.

    :return: None
    """
    args = parse_cmd_line()
    args.model = "transformer"
    args.use_pretrained = False
    loss_criterion = "mse"

    print("Training Robot State Transformer Network")

    run_nums = util.get_run_numbers(args)
    current_time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = util.get_log_dir(args) + current_time
    hparams = {
        'batch_size': args.batch_size,
        'lr': args.lr,
        'num_epochs': args.num_epochs,
        'seq_length': args.seq_length
    }
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_hparams(hparams, {})

    dataset_name = "force-runs" if not run_nums["train"][1] else "all-runs"

    data_dir = "data"
    sets = ["train", "test"]

    if args.use_kfold:
        fold_splits = util.prepare_kfold_datasets(
            args, run_nums, data_dir, sets)

        for fold in range(args.k_folds):
            wandb.init(
                project="force-transformer",
                group=f"{args.k_folds}_fold_seq_{args.seq_length}",
                name=f"fold_{fold}_seq_{args.seq_length}",
                config={
                    'architecture': 'state-transformer',
                    'dataset': "force-runs" if not run_nums["train"][1] else "all-runs",
                    'validation-method': f'{args.k_folds}-fold',
                    'batch_size': args.batch_size,
                    'lr': args.lr,
                    'num_epochs': args.num_epochs,
                    'seq_length': args.seq_length,
                    'loss_criterion': loss_criterion
                },
                reinit=True
            )

            wandb.config["val_indices"] = fold_splits[fold]["test"][0]
            k_fold_weights_dir = util.create_kfolds_weights_path(
                util.create_weights_path(
                    args.out_dir, args.num_epochs), fold, args.k_folds
            )

            transformations_path = k_fold_weights_dir.replace("weights/", "transformations/")
            feature_scaler_path = transformations_path + "/feature_scaler.joblib"
            target_scaler_path = transformations_path + "/target_scaler.joblib"
            kfold_dataloaders = util.prepare_standard_datasets(
                args, fold_splits[fold], data_dir, sets, feature_scaler_path, target_scaler_path)

            model = RobotStateTransformer(util.get_transformer_config(args))
            model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

            trainer = TransformerTrainer(
                model=model,
                data_loaders=kfold_dataloaders,
                device=torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"),
                criterion=loss_criterion,
                lr=args.lr,
                regularized=True,
                weights_dir=k_fold_weights_dir,
                writer=writer,
                lr_scheduler_config=LRSchedulerConfig() if args.lr_scheduler else None,
                use_acceleration=args.use_acceleration
            )
            trainer.train(num_epochs=args.num_epochs)
    else:
        wandb.init(
            project="force-transformer",
            config={
                'architecture': 'state-transformer',
                'dataset': "force-runs" if not run_nums["train"][1] else "all-runs",
                'batch_size': args.batch_size,
                'lr': args.lr,
                'num_epochs': args.num_epochs,
                'seq_length': args.seq_length,
                'loss_criterion': loss_criterion,
                'val_indices': constants.DEFAULT_TEST_RUNS[0]
            }
        )
        weights_path = util.create_weights_path("robot_state_transformer", args.num_epochs) # TODO
        transformations_path = weights_path.replace("weights/", "transformations/")
        feature_scaler_path = transformations_path + "/feature_scaler.joblib"
        target_scaler_path = transformations_path + "/target_scaler.joblib"
        
        data_loaders = util.prepare_standard_datasets(
            args, run_nums, data_dir, sets, feature_scaler_path, target_scaler_path)

        model = RobotStateTransformer(util.get_transformer_config(args))
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        trainer = TransformerTrainer(
            model=model,
            data_loaders=data_loaders,
            device=torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"),
            criterion=loss_criterion,
            lr=args.lr,
            regularized=True,
            weights_dir= weights_path,
            writer=writer,
            lr_scheduler_config=LRSchedulerConfig() if args.lr_scheduler else None,
            use_acceleration=args.use_acceleration
        )
        trainer.train(num_epochs=args.num_epochs)


if __name__ == "__main__":
    train()
