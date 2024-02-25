import unittest
from unittest.mock import MagicMock, patch
import torch
import torch.nn as nn
from trainer.trainer import Trainer
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter


class TestTrainer(unittest.TestCase):
    def setUp(self):
        self.mock_model = nn.Linear(10, 2)  # Simple model for testing
        self.mock_device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.mock_criterion = 'mse'
        self.mock_lr = 0.001
        self.mock_regularized = False
        self.mock_weights_dir = 'weights'
        self.mock_writer = SummaryWriter()

        self.mock_dataset = TensorDataset(
            torch.randn(100, 10), torch.randn(100, 2))
        self.mock_data_loader = DataLoader(self.mock_dataset, batch_size=10)
        self.mock_data_loaders = {
            'train': self.mock_data_loader, 'test': self.mock_data_loader}

    @patch('os.path.isdir', return_value=True)
    def test_initialization(self, mock_isdir):
        trainer = Trainer(
            model=self.mock_model,
            data_loaders=self.mock_data_loaders,
            device=self.mock_device,
            criterion=self.mock_criterion,
            lr=self.mock_lr,
            regularized=self.mock_regularized,
            weights_dir=self.mock_weights_dir,
            writer=self.mock_writer
        )
        self.assertIsInstance(trainer, Trainer)

    def test_criterion_selection(self):
        trainer = Trainer(
            model=self.mock_model,
            data_loaders=self.mock_data_loaders,
            device=self.mock_device,
            criterion='mse',
            lr=self.mock_lr,
            regularized=self.mock_regularized,
            weights_dir=self.mock_weights_dir,
            writer=self.mock_writer
        )
        self.assertIsInstance(trainer.criterion, nn.MSELoss)

    def test_invalid_criterion(self):
        with self.assertRaises(ValueError):
            Trainer(
                model=self.mock_model,
                data_loaders=self.mock_data_loaders,
                device=self.mock_device,
                criterion='invalid_criterion',
                lr=self.mock_lr,
                regularized=self.mock_regularized,
                weights_dir=self.mock_weights_dir,
                writer=self.mock_writer
            )


if __name__ == '__main__':
    unittest.main()
