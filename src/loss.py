import torch
import torch.nn as nn


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, y_hat, y):
        loss = torch.sqrt(self.mse(y_hat, y) + self.eps)
        return loss


class WeightedAxisMSELoss(nn.Module):
    def __init__(self, weights=(0.3, 0.3, 0.4), eps=1e-6):
        """
        Loss that computes a weighted sum of axis-wise MSE values.

        Args:
            weights: Tuple of weights for X, Y, Z axes (default: 0.25, 0.25, 0.5).
            eps: Small epsilon value to prevent numerical instability (default: 1e-6).
        """
        super().__init__()
        self.weights = torch.tensor(weights)  # Weights for X, Y, Z
        self.eps = eps

    def forward(self, y_hat, y):
        """
        Computes the weighted MSE loss for each axis.

        Args:
            y_hat: Predicted tensor (Nx3).
            y: Ground truth tensor (Nx3).

        Returns:
            Weighted MSE loss value.
        """
        # Compute element-wise squared differences
        squared_diff = (y_hat - y) ** 2

        # Compute MSE for each axis
        mse_per_axis = squared_diff.mean(dim=0)  # Shape: (3,)

        # Apply weights to the axis-wise MSE and sum them
        weighted_loss = torch.sum(self.weights.to(
            y.device) * mse_per_axis) + self.eps

        return weighted_loss
