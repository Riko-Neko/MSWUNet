from typing import Optional, List, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torchmetrics.image import StructuralSimilarityIndexMeasure

"""
Seperate loss for 1D (frequency) feature extraction & yolo-based object detection.

Provides:
- build_target_yolo(targets_list, S, C, device):
    maps ground-truth boxes (per image) -> target tensor (B, S, S, 5+C)

Design notes:
- This is a simplified version of the YOLO target map format, where each cell
  contains a single object (no multi-object support).

"""


# ----- Build target map from list of (N,5) -> (B,S,S,5+C) -----
def build_target_yolo(targets_list, S: int, C: int, device: torch.device):
    """
    Convert a batch of ground truth boxes (per image) into YOLO-style target map.

    Args:
        targets_list (List[Tensor]): List of length B.
            Each entry is a tensor of shape (N, 5) or (5,) containing:
            [class_id, cx, cy, w, h] where (cx, cy, w, h) are normalized in [0,1].
        S (int): Grid size (S x S).
        C (int): Number of classes.
        device (torch.device): Target device.

    Returns:
        target_map (Tensor): Shape (B, S, S, 5 + C)
            Channels:
                0: objectness flag (1.0 if object exists)
                1: x-center relative to cell [0,1]
                2: y-center relative to cell [0,1]
                3: width (normalized to image)
                4: height (normalized to image)
                5:5+C: one-hot class vector
    """
    B = len(targets_list)
    target_map = torch.zeros((B, S, S, 5 + C), dtype=torch.float32, device=device)

    for bi, gt in enumerate(targets_list):
        if gt is None or gt.numel() == 0:
            continue

        # Ensure (N, 5) float tensor on correct device
        if gt.ndim == 1 and gt.shape[0] == 5:
            gt = gt.unsqueeze(0)
        gt = gt.to(dtype=torch.float32, device=device)

        # --------------------------------------------------------------
        # 1. Filter invalid GT boxes using _valid_gt_mask
        # --------------------------------------------------------------
        def _valid_gt_mask(gt_starts: torch.Tensor, gt_stops: torch.Tensor) -> torch.Tensor:
            """Return boolean mask of valid GTs (finite and in range)."""
            valid = torch.isfinite(gt_starts) & torch.isfinite(gt_stops)
            return valid

        gt_cx = gt[:, 1]  # center x
        gt_cy = gt[:, 2]  # center y
        valid_mask = _valid_gt_mask(gt_cx, gt_cy)

        # Also ensure width and height are positive
        valid_mask = valid_mask & (gt[:, 3] > 0.0) & (gt[:, 4] > 0.0)

        if not valid_mask.any():
            continue  # No valid boxes in this image
        gt = gt[valid_mask]

        # --------------------------------------------------------------
        # 2. Map valid boxes to YOLO grid cells
        # --------------------------------------------------------------
        cls = gt[:, 0].long()  # class ID
        cx = gt[:, 1].clamp_(0.0, 0.9999)
        cy = gt[:, 2].clamp_(0.0, 0.9999)
        w = gt[:, 3]
        h = gt[:, 4]

        # Compute grid cell indices
        cell_x = (cx * S).long()
        cell_y = (cy * S).long()

        # Relative coordinates within the cell
        cx_in_cell = cx * S - cell_x.float()
        cy_in_cell = cy * S - cell_y.float()

        # Fill target map (vectorized over all boxes in this image)
        target_map[bi, cell_y, cell_x, 0] = 1.0  # objectness
        target_map[bi, cell_y, cell_x, 1] = cx_in_cell
        target_map[bi, cell_y, cell_x, 2] = cy_in_cell
        target_map[bi, cell_y, cell_x, 3] = w
        target_map[bi, cell_y, cell_x, 4] = h

        # One-hot class encoding (safe modulo C)
        cls = cls % C
        class_indices = 5 + cls
        rows = torch.arange(bi, bi + 1, device=device).unsqueeze(-1).expand(-1, len(cls))
        cols = cell_y.unsqueeze(0), cell_x.unsqueeze(0), class_indices.unsqueeze(0)
        target_map.index_put_(
            (rows, cell_y.unsqueeze(0), cell_x.unsqueeze(0), class_indices.unsqueeze(0)),
            torch.ones_like(class_indices, dtype=torch.float32),
            accumulate=True
        )

    return target_map


"""
Conbined loss for 1D (frequency) metrics regression.

Provides:
- RegressionHeadLoss(w_loc, w_class, w_conf, eps):
    loss function for RegressionDetector head that consumes raw logits

    Returns:
        loss_dict (dict): Dictionary of scalar losses.

"""


def build_target_regression(gt_boxes: torch.Tensor, num_classes: int = 2, N: int = 10):
    assert gt_boxes.dim() == 3, "gt_boxes should be of shape (B, max_num_signals, 3)"
    B, K, _ = gt_boxes.shape
    device = gt_boxes.device
    if N is None:
        N = K
    f_start = gt_boxes[..., 0]  # (B, K)
    f_end = gt_boxes[..., 1]  # (B, K)
    cls = gt_boxes[..., 2]  # (B, K)
    presence = (~torch.isnan(f_start)) & (~torch.isnan(f_end)) & (~torch.isnan(cls))
    presence = presence.float()  # (B, K)
    f_start = torch.nan_to_num(f_start, nan=0.0)
    f_end = torch.nan_to_num(f_end, nan=0.0)
    cls = torch.nan_to_num(cls, nan=0.0)
    cls_idx = cls.long().clamp(0, num_classes - 1)  # 防止越界
    class_onehot = F.one_hot(cls_idx, num_classes).float()  # (B, K, C)

    def pad_or_trunc(x, dim, value=0.0):
        size = x.size(1)
        if size == N:
            return x
        elif size > N:
            return x[:, :N]
        else:
            pad_shape = list(x.shape)
            pad_shape[1] = N - size
            pad = torch.full(pad_shape, value, device=x.device, dtype=x.dtype)
            return torch.cat([x, pad], dim=1)

    f_start = pad_or_trunc(f_start, 1, value=0.0)
    f_end = pad_or_trunc(f_end, 1, value=0.0)
    presence = pad_or_trunc(presence, 1, value=0.0)
    class_onehot = pad_or_trunc(class_onehot, 1, value=0.0)

    target = {"f_start": f_start.to(device), "f_end": f_end.to(device), "class": class_onehot.to(device),
              "presence": presence.to(device)}
    return target


class RegressionHeadLoss(nn.Module):
    def __init__(self, num_classes: int = 2, N: int = 10, w_loc: float = 1.0, w_class: float = 1.0, w_conf: float = 1.0,
                 eps: float = 1e-8):
        super().__init__()
        self.num_classes = num_classes
        self.N_bins = N
        self.w_loc = float(w_loc)
        self.w_class = float(w_class)
        self.w_conf = float(w_conf)
        self.eps = float(eps)
        self.bce_logits = nn.BCEWithLogitsLoss(reduction="none")

    @staticmethod
    def _iou_1d_interval(s1: torch.Tensor, e1: torch.Tensor, s2: torch.Tensor, e2: torch.Tensor,
                         eps: float = 1e-8) -> torch.Tensor:
        """Compute 1D IoU for intervals (tensors). Handles unordered start/end."""
        a1 = torch.min(s1, e1)
        b1 = torch.max(s1, e1)
        a2 = torch.min(s2, e2)
        b2 = torch.max(s2, e2)
        inter_left = torch.max(a1, a2)
        inter_right = torch.min(b1, b2)
        inter = torch.clamp(inter_right - inter_left, min=0.0)
        union_left = torch.min(a1, a2)
        union_right = torch.max(b1, b2)
        union = torch.clamp(union_right - union_left, min=eps)
        return inter / union

    @staticmethod
    def _giou_1d_interval(s1: torch.Tensor, e1: torch.Tensor, s2: torch.Tensor, e2: torch.Tensor,
                          eps: float = 1e-8) -> torch.Tensor:
        a1 = torch.min(s1, e1)
        b1 = torch.max(s1, e1)
        a2 = torch.min(s2, e2)
        b2 = torch.max(s2, e2)
        inter_left = torch.max(a1, a2)
        inter_right = torch.min(b1, b2)
        inter = torch.clamp(inter_right - inter_left, min=0.0)
        len1 = b1 - a1
        len2 = b2 - a2
        union = len1 + len2 - inter
        union = torch.clamp(union, min=eps)
        c_left = torch.min(a1, a2)
        c_right = torch.max(b1, b2)
        convex = torch.clamp(c_right - c_left, min=eps)
        iou = inter / union
        giou = iou - (convex - union) / convex
        return giou

    def forward(self, pred: Dict[str, torch.Tensor], gt_boxes: torch.Tensor) -> Tuple[
        torch.Tensor, Dict[str, torch.Tensor]]:
        # Unpack preds (raw logits)
        p_start = pred["f_start"]  # (B, N)
        p_end = pred["f_end"]  # (B, N)
        p_class_logits = pred["class_logits"]  # (B, N, C)
        p_conf_logits = pred.get("confidence", None)  # (B, N) logits or None

        device = p_start.device
        B, N = p_start.shape
        C = p_class_logits.shape[-1]

        # Unpack targets
        target = build_target_regression(gt_boxes, num_classes=self.num_classes, N=self.N_bins)
        f_start_all = target["f_start"].to(device)  # (B, K)
        f_end_all = target["f_end"].to(device)  # (B, K)
        classes_all = target["class"].to(device)  # (B, K, C)
        presence_all = target["presence"].to(device).float()  # (B, K) 0/1

        total_loc = 0.0
        total_class = 0.0
        total_conf = 0.0
        total_matched = 0

        for b in range(B):
            pres = presence_all[b]  # (K,)
            valid_idx = torch.nonzero(pres, as_tuple=False).view(-1)
            K = valid_idx.numel()
            if K == 0:
                continue

            # gather valid GTs
            f_start = f_start_all[b, valid_idx]  # (K,)
            f_end = f_end_all[b, valid_idx]  # (K,)
            s_class = classes_all[b, valid_idx, :]  # (K, C)

            # Predictions for this sample
            ps = p_start[b]  # (N,) logits
            pe = p_end[b]  # (N,)
            pcl = p_class_logits[b]  # (N, C) logits
            pconf = p_conf_logits[b] if p_conf_logits is not None else None  # (N,)

            # Build cost matrix
            ps_sig = torch.sigmoid(ps).unsqueeze(0).expand(K, N)  # (K,N)
            pe_sig = torch.sigmoid(pe).unsqueeze(0).expand(K, N)  # (K,N)
            f_start_mat = f_start.unsqueeze(1).expand(K, N)
            f_end_mat = f_end.unsqueeze(1).expand(K, N)
            loc_mse = (ps_sig - f_start_mat).pow(2) + (pe_sig - f_end_mat).pow(2)  # (K,N)

            pcl_sig = torch.sigmoid(pcl).unsqueeze(0).expand(K, N, C)  # (K,N,C)
            f_class_mat = s_class.unsqueeze(1).expand(K, N, C)  # (K,N,C)
            class_l2 = (pcl_sig - f_class_mat).pow(2).sum(dim=-1)  # (K,N)

            cost = loc_mse + class_l2  # (K,N)

            # Hungarian matching
            row_ind, col_ind = linear_sum_assignment(cost.cpu().numpy())
            pairs = [(r, c) for r, c in zip(row_ind, col_ind)]

            matched_preds = set(col_ind)
            unmatched_preds = [pj for pj in range(N) if pj not in matched_preds]

            loc_losses = []
            class_losses = []
            conf_losses = []

            for (gi, pj) in pairs:
                pred_s = torch.sigmoid(ps[pj])
                pred_e = torch.sigmoid(pe[pj])
                gt_s = f_start[gi]
                gt_e = f_end[gi]

                # localization loss: 1 - GIoU
                giou = self._giou_1d_interval(pred_s, pred_e, gt_s, gt_e, self.eps)
                loc_loss = 1 - giou
                loc_losses.append(loc_loss)

                # classification loss: BCEWithLogits between logits and target vector (sum over classes)
                gt_class_vec = s_class[gi]  # (C,)
                pred_class_logits = pcl[pj].unsqueeze(0)  # (1,C)
                gt_class_vec = gt_class_vec.unsqueeze(0)  # (1,C)
                class_elem = self.bce_logits(pred_class_logits, gt_class_vec)  # (1,C)
                class_loss = class_elem.sum()
                class_losses.append(class_loss)

                # confidence obj loss for positive
                if pconf is not None:
                    conf_loss_pos = self.bce_logits(pconf[pj].unsqueeze(0), torch.tensor([1.0], device=device))
                    conf_losses.append(conf_loss_pos)

            # confidence obj loss for negative (unmatched preds)
            if pconf is not None:
                for pj in unmatched_preds:
                    conf_loss_neg = self.bce_logits(pconf[pj].unsqueeze(0), torch.tensor([0.0], device=device))
                    conf_losses.append(conf_loss_neg)

            loc_sum = torch.stack(loc_losses).sum() if len(loc_losses) > 0 else torch.tensor(0.0, device=device)
            class_sum = torch.stack(class_losses).sum() if len(class_losses) > 0 else torch.tensor(0.0, device=device)
            conf_sum = torch.stack(conf_losses).sum() if len(conf_losses) > 0 else torch.tensor(0.0, device=device)
            total_loc += loc_sum
            total_class += class_sum
            total_conf += conf_sum
            total_matched += len(pairs)

        if total_matched == 0:
            zero = torch.tensor(0.0, device=device)
            return zero, {"loc_loss": zero, "class_loss": zero, "conf_loss": zero, "total_loss": zero}

        loc_loss_avg = total_loc / (total_matched + self.eps)
        class_loss_avg = total_class / (total_matched + self.eps)
        conf_loss_avg = total_conf / (B * N + self.eps)  # average over all predictions

        total_loss = self.w_loc * loc_loss_avg + self.w_class * class_loss_avg + self.w_conf * conf_loss_avg

        loss_dict = {"loc_loss": loc_loss_avg.detach(), "class_loss": class_loss_avg.detach(),
                     "conf_loss": conf_loss_avg.detach(), "total_loss": total_loss.detach(),
                     "matched_count": total_matched, }
        return total_loss, loss_dict


class DetectionCombinedLoss(nn.Module):
    """
    Combined loss for denoising and frequency detection tasks.

    Combines:
    1. Frequency detection loss (FreqDetectionLoss)
    2. Denoising loss (MSE or Dice)

    Args:
        lambda_denoise: Weight factor for denoising loss
        loss_type: Type of denoising loss ('mse' or 'dice')
        regression_loss_kwargs: Optional kwargs for RegressionHeadLoss
        device: Torch device
    """

    def __init__(self, lambda_denoise: float = 1.0, loss_type: str = 'mse', lambda_learnable: bool = False,
                 regression_loss_kwargs: Optional[dict] = None, device: Optional[torch.device] = None):
        super().__init__()

        assert loss_type in ('mse', 'dice'), "denoise_loss_type must be 'mse' or 'dice'"

        self.lambda_denoise = nn.Parameter(
            torch.tensor(lambda_denoise, device=device)) if lambda_learnable else lambda_denoise
        self.loss_type = loss_type
        self.device = device

        # Initialize frequency regression loss
        regression_loss_kwargs = regression_loss_kwargs or {}
        self.detection_loss = RegressionHeadLoss(**regression_loss_kwargs)

        # Initialize denoising loss
        if loss_type == 'mse':
            self.denoise_loss = nn.MSELoss(reduction='mean')
        else:  # dice
            self.denoise_loss = DiceLoss()

    def forward(self, raw_preds, denoised, clean, gt_boxes) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass for combined loss calculation.

        Args:
            raw_preds: Frequency detection predictions (single tensor or list of tensors)
            denoised: Denoised spectrum prediction (B, C, T, F)
            clean: Clean target spectrum (B, C, T, F)
            gt_boxes: Ground truth boxes for frequency detection (B, CLS, N, 2)

        Returns:
            total_loss: Combined loss value
            metrics: Dictionary containing all loss components and metrics
        """
        # Calculate frequency detection loss
        detection_loss, detection_metrics = self.detection_loss(raw_preds, gt_boxes)
        denoised_loss = self.denoise_loss(denoised, clean)

        # Combine losses
        total_loss = detection_loss + self.lambda_denoise * denoised_loss

        # Prepare metrics
        metrics = {
            'total_loss': total_loss.item(),
            'detection_loss': detection_loss.item(),
            'denoise_loss': denoised_loss.item(),
            'lambda_denoise': self.lambda_denoise,
            'loss_type': self.loss_type
        }

        # Add frequency detection metrics with prefix
        for k, v in detection_metrics.items():
            metrics[f'{k}'] = v

        return total_loss, metrics


"""
Loss for combined denoising, RFI detection, and physical detection.

Provides:
- MaskCombinedLoss(nn.Module):
    nn.Module loss that combines MSE+SSIM for spectrum denoising, Dice loss for RFI detection, 
    and BCEWithLogits for physical detection. Supports dynamic weight adjustment and learnable parameters.

Design decisions / assumptions (can be changed in code):
- Uses weighted MSE with signal/background weighting and SSIM for spectrum reconstruction quality.
- Applies Dice loss for RFI mask prediction to handle class imbalance.
- Dynamically adjusts alpha/beta weights based on loss trends using linear regression.
- Supports fixed or learnable gamma (SSIM ratio) and delta (detection ratio) parameters.
- Includes cosine jitter to escape local minima when both losses stagnate.

Usage summary:
    loss_module = MaskCombinedLoss(device=device, alpha=0.7, beta=0.3)
    total_loss, metrics = loss_module(denoised, rfi_pred, plogits, clean, mask, pprob)
"""


class MaskCombinedLoss(nn.Module):
    """
    Loss for combined denoising, RFI detection, and physical detection.

    forward(denoised, rfi_pred, plogits, clean, mask, pprob) -> (total_loss, metrics)

    Args：
        device: torch.device for tensors.
        alpha: initial weight for spectrum denoising loss.
        beta: initial weight for RFI detection loss.
        gamma: SSIM loss mixing factor (0-1).
        delta: physical detection loss mixing factor (0-1).
        adjust_threshold: window size for dynamic weight adjustment.
        momentum: momentum for moving average normalization.
        fixed_g_d: whether gamma/delta are fixed or learnable.
    """

    def __init__(self, device, alpha=0.7, beta=0.3, gamma=0., delta=0.1, adjust_threshold=20, momentum=0.99,
                 fixed_g_d=False):
        super(MaskCombinedLoss, self).__init__()
        self.alpha_init = alpha  # initial spectrum detection weight
        self.beta_init = beta  # initial RFI detection weight
        gamma, delta = torch.tensor(gamma, device=device), torch.tensor(delta, device=device)
        fixed_gamma = fixed_delta = bool(fixed_g_d)
        if isinstance(fixed_g_d, (list, tuple)) and len(fixed_g_d) == 2:
            fixed_gamma, fixed_delta = map(bool, fixed_g_d)
        self.gamma = gamma if fixed_gamma else nn.Parameter(gamma)
        self.delta = delta if fixed_delta else nn.Parameter(delta)

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
        """
        Compute combined loss.

        Args:
            denoised: (B, 1, T, F) tensor of denoised spectrum.
            rfi_pred: (B, 1, T, F) tensor of RFI mask prediction.
            plogits: (B, 1) tensor of physical detection logits.
            clean: (B, 1, T, F) tensor of clean spectrum.
            mask: (B, 1, T, F) tensor of RFI mask.
            pprob: (B, 1) tensor of physical detection probability.

        Returns:
            total_loss: scalar tensor
            metrics: dict with keys {'spectrum_loss','ssim_loss', 'rfi_loss', 'detection_loss', 'alpha', 'beta', 'gamma', 'delta'}

        """
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

        metrics = {
            "spectrum_loss": spectrum_loss_scalar,
            "ssim_loss": ssim_loss,
            "rfi_loss": rfi_loss,
            "detection_loss": detection_loss,
            "alpha": self.alpha.item(),
            "beta": self.beta.item(),
            "gamma": self.gamma.item(),
            "delta": self.delta.item(),
        }

        return total_loss, metrics

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


"""
Dice loss (optional), this may be helpful in certain tasks.
"""


class DiceLoss(nn.Module):
    def forward(self, pred, target):
        pred = pred.sigmoid()
        intersection = (pred * target).sum()
        return 1.0 - (2.0 * intersection) / (pred.sum() + target.sum() + 1e-6)
