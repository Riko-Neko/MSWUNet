import numpy as np
import torch
from torch import nn
from torchmetrics.image import StructuralSimilarityIndexMeasure


class CombinedLoss(nn.Module):
    def __init__(self, device, alpha=0.7, beta=0.3, gamma=0., delta=0.1, adjust_threshold=20, momentum=0.99):
        super(CombinedLoss, self).__init__()
        self.alpha_init = alpha  # initial spectrum detection weight
        self.beta_init = beta  # initial RFI detection weight
        self.gamma = nn.Parameter(torch.tensor(gamma, device=device)) if gamma != 0. else torch.tensor(0.,
                                                                                                       device=device)  # ssim loss weight (0 for observation)
        self.delta = nn.Parameter(torch.tensor(delta, device=device))  # BCE loss for physical detection
        self.adjust_threshold = adjust_threshold  # window size for adjusting weights
        self.momentum = momentum
        self.jitter_period = 30  # cosine jitter period
        self.jitter_amplitude = 0.1  # jitter amplitude
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

        self.mse = nn.MSELoss(reduction='none')
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0], device=device))
        self.dice = DiceLoss()

        # register buffers for moving average and weights
        self.register_buffer('mse_moving_avg', torch.tensor(1.0))
        self.register_buffer('alpha', torch.tensor(alpha))
        self.register_buffer('beta', torch.tensor(beta))
        self.step = 0

        # history for trend detection
        self.spec_history = []
        self.rfi_history = []

        self.trend_threshold = 0.001  # threshold for trend detection

    def forward(self, denoised, rfi_pred, plogits, clean, mask, pprob):
        # mse loss for spectrum detection
        signal_weight = 10.0
        background_weight = 1.
        weight_map = torch.where(clean > 0.,
                                 torch.full_like(clean, signal_weight),
                                 torch.full_like(clean, background_weight))
        spectrum_loss = (self.mse(denoised, clean) * weight_map)
        ssim_loss = 1 - self.ssim(denoised, clean)

        # initialize moving average if it's the first step
        if self.step == 0:
            self.mse_moving_avg.fill_(spectrum_loss.detach().mean())

        # apply moving average normalization to spectrum loss
        normalized_spectrum_loss = spectrum_loss / torch.clamp(self.mse_moving_avg, min=1e-6)
        gamma = torch.relu(self.gamma)
        spectrum_loss_scalar = normalized_spectrum_loss.mean() * (1 - gamma) + ssim_loss * gamma

        # dice loss for RFI detection
        rfi_loss = self.dice(rfi_pred, mask)

        # update history
        self.spec_history.append(spectrum_loss_scalar.detach().item())
        self.rfi_history.append(rfi_loss.detach().item())

        # ensure history size is within reasonable range
        if len(self.spec_history) > self.adjust_threshold:
            self.spec_history.pop(0)
            self.rfi_history.pop(0)

        # dynamically adjust weights based on trend detection
        if len(self.spec_history) >= 2:  # at least two history entries required
            spec_trend = self._compute_trend(self.spec_history)
            rfi_trend = self._compute_trend(self.rfi_history)

            # case1: both loss decrease - do nothing
            if spec_trend < -self.trend_threshold and rfi_trend < -self.trend_threshold:
                pass

            # case2: spectrum loss decrease but RFI loss increase - decrease spectrum weight
            elif spec_trend < -self.trend_threshold <= rfi_trend:
                new_alpha = max(0.0, self.alpha.item() - 0.01)
                self.alpha.fill_(new_alpha)
                self.beta.fill_(1.0 - new_alpha)

            # case3: rfi loss decrease but spectrum loss increase - decrease RFI weight
            elif rfi_trend < -self.trend_threshold <= spec_trend:
                new_alpha = min(1.0, self.alpha.item() + 0.01)
                self.alpha.fill_(new_alpha)
                self.beta.fill_(1.0 - new_alpha)

            # case4: both loss not decrease - apply jitter
            else:
                # cosine jitter to adjust weights
                jitter = self.jitter_amplitude * np.cos(2 * np.pi * self.step / self.jitter_period)
                new_alpha = np.clip(self.alpha_init + jitter, 0.0, 1.0)
                self.alpha.fill_(new_alpha)
                self.beta.fill_(1.0 - new_alpha)

        # detection loss for physical detection
        detection_loss = self.bce(plogits, pprob)
        delta = torch.relu(self.delta)

        # compute total loss
        total_loss = + (self.alpha * spectrum_loss_scalar + self.beta * rfi_loss) * (1 - delta) + delta * detection_loss

        # update moving average
        current_mse_avg = spectrum_loss_scalar.detach()
        self.mse_moving_avg.data = (
                self.momentum * self.mse_moving_avg +
                (1 - self.momentum) * current_mse_avg
        )

        self.step += 1
        return total_loss, spectrum_loss_scalar, ssim_loss, rfi_loss, detection_loss, self.alpha.item(), self.beta.item(), self.gamma.item(), self.delta.item()

    def _compute_trend(self, history):
        n = len(history)
        if n < 2:
            return 0.0

        # compute trend using linear regression
        x = np.arange(n)
        y = np.array(history)
        x_mean = np.mean(x)
        y_mean = np.mean(y)

        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)

        if abs(denominator) < 1e-8:
            return 0.0

        slope = numerator / denominator
        return slope


class DiceLoss(nn.Module):
    def forward(self, pred, target):
        pred = pred.sigmoid()
        intersection = (pred * target).sum()
        return 1.0 - (2.0 * intersection) / (pred.sum() + target.sum() + 1e-6)
