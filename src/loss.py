import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, y_hat, y):
        loss = torch.sqrt(self.mse(y_hat, y) + self.eps)
        return loss

class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, prediction, target):
        diff = prediction - target
        abs_diff = torch.abs(diff)

        quadratic = torch.min(abs_diff, torch.tensor(self.delta))
        linear = abs_diff - quadratic

        loss = 0.5 * quadratic ** 2 + self.delta * linear
        return loss.mean()

class WeightedHuberLoss(nn.Module):
    def __init__(self, delta=1.0, l1_weight=2.0):
        super().__init__()
        self.delta = delta
        self.l1_weight = l1_weight  # Scale for linear (L1-like) region

    def forward(self, y_pred, y_true):
        error = y_true - y_pred
        abs_error = torch.abs(error)
        is_small_error = abs_error <= self.delta

        squared_loss = 0.5 * error**2
        linear_loss = self.l1_weight * self.delta * (abs_error - 0.5 * self.delta)

        return torch.where(is_small_error, squared_loss, linear_loss).mean()
    
class WeightedMSELoss(nn.Module):
    def __init__(self, w1=1.0, w2=1.0):
        super(WeightedMSELoss, self).__init__()
        self.w1 = w1
        self.w2 = w2

    def forward(self, prediction, target):
        mse = (prediction - target) ** 2

        # Boolean masks
        zero_mask = (target == 0)
        nonzero_mask = (target != 0) & (prediction != 0)

        # Apply weights
        loss = torch.zeros_like(mse)
        loss[zero_mask] = mse[zero_mask] * self.w1
        loss[nonzero_mask] = mse[nonzero_mask] * self.w2

        return loss.mean()
    
class AdvancedForceLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.5, gamma=1.0, delta=0.5, 
                 base_weight=1.0, spike_weight=5.0, spike_threshold=0.1,
                 kernel_sizes=[3, 7, 15]):
        super().__init__()
        self.alpha = alpha  # Derivative loss weight
        self.beta = beta    # Fourier loss weight
        self.gamma = gamma  # Event-aware loss weight
        self.delta = delta  # Multiscale loss weight
        self.base_weight = base_weight
        self.spike_weight = spike_weight
        self.spike_threshold = spike_threshold
        self.kernel_sizes = kernel_sizes

    def forward(self, pred, target):
        base = F.mse_loss(pred, target)
        d_loss = self.derivative_loss(pred, target)
        f_loss = self.fourier_loss(pred, target)
        e_loss = self.event_aware_loss(pred, target)
        m_loss = self.multiscale_loss(pred, target)

        return base + self.alpha * d_loss + self.beta * f_loss + self.gamma * e_loss + self.delta * m_loss

    def derivative_loss(self, pred, target):
        pred_diff = pred[:, 1:] - pred[:, :-1]
        target_diff = target[:, 1:] - target[:, :-1]
        return F.mse_loss(pred_diff, target_diff)

    def fourier_loss(self, pred, target):
        loss = 0
        for i in range(3):  # x, y, z
            pred_fft = torch.fft.fft(pred[:, i, :], dim=-1)
            target_fft = torch.fft.fft(target[:, i, :], dim=-1)
            loss += F.mse_loss(torch.abs(pred_fft), torch.abs(target_fft))
        return loss / 3

    def event_aware_loss(self, pred, target):
        target_diff = torch.abs(target[:, 1:] - target[:, :-1])
        spike_mask = (target_diff > self.spike_threshold).float()
        spike_mask = F.pad(spike_mask, (1, 0))  # match original shape

        loss = (pred - target) ** 2
        weights = self.base_weight + spike_mask * (self.spike_weight - self.base_weight)
        return (loss * weights).mean()

    def multiscale_loss(self, pred, target):
        loss = 0.0
        for k in self.kernel_sizes:
            pred_s = self.gaussian_smooth(pred, k)
            target_s = self.gaussian_smooth(target, k)
            loss += F.mse_loss(pred_s, target_s)
        return loss / len(self.kernel_sizes)

    def gaussian_smooth(self, x, kernel_size):
        padding = kernel_size // 2
        weights = torch.exp(-torch.linspace(-2, 2, steps=kernel_size) ** 2)
        weights = weights / weights.sum()
        weights = weights.to(x.device).unsqueeze(0).unsqueeze(0)  # [1, 1, K]

        x = x.unsqueeze(1)  # [B, 1, T]
        x_smoothed = F.conv1d(x, weights, padding=padding)
        return x_smoothed.squeeze(1)