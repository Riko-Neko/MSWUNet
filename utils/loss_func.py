from typing import Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure

"""
Loss and target-building utilities for 1D (frequency) object detection.

Provides:
- build_targets_freq_vectorized_no_time(gt_boxes, P, F, device, clip=True):
    maps ground-truth boxes (B, N, 2) -> target tensor (B, P, 3, F) (no time dim)

- build_targets_freq_vectorized_with_time(...): same but returns (B,P,3,T,F) by broadcasting

- FreqDetectionLoss(nn.Module):
    supports training that either duplicates GT across time (old behavior) or aggregates
    predictions across time (recommended when GT has no time info). Aggregation modes:
      - 'soft'  : softmax-weighted regression across time (differentiable)
      - 'argmax': pick time index with max logit and use its reg preds (sparse gradient)
      - 'none'  : do not aggregate; expect targets with T dimension (legacy behavior)

Design notes:
- If your GT does not include time, prefer aggregation modes ('soft' or 'argmax') so
  we don't naively duplicate supervision across all T steps.
- Vectorized target building operates per-batch but fully vectorized across frequency F
  (and across the GT list for each sample), so it's efficient for large F.

"""


def _valid_gt_mask(gt_starts: torch.Tensor, gt_stops: torch.Tensor) -> torch.Tensor:
    """Return boolean mask of valid GTs (finite and in range)."""
    valid = torch.isfinite(gt_starts) & torch.isfinite(gt_stops)
    return valid


def build_targets_freq_vectorized_no_time(gt_boxes: torch.Tensor, P: int, F: int, device: Optional[torch.device] = None,
                                          clip: bool = True) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Vectorized target builder that does NOT expand across time.

    Args:
        gt_boxes: (B, N, 2) with normalized f_start,f_stop in [0,1]
        P: slots per (f) cell
        F: number of frequency cells
        device: torch device
        clip: clamp bounds to [0,1]

    Returns:
        targets_no_time: (B, P, 3, F) where 3=(f_start,f_stop,conf)
        pos_mask: (B, P, F) boolean
        n_pos: int
    """
    if device is None:
        device = gt_boxes.device if isinstance(gt_boxes, torch.Tensor) else torch.device('cpu')

    if gt_boxes.dim() != 3 or gt_boxes.size(-1) != 2:
        raise ValueError("gt_boxes must have shape (B, N, 2)")

    B, N, _ = gt_boxes.shape
    targets = torch.zeros((B, P, 3, F), dtype=torch.float32, device=device)

    # frequency cell boundaries (float tensors)
    cell_idx = torch.arange(F, dtype=torch.float32, device=device)
    cell_left = cell_idx / float(F)  # (F,)
    cell_right = (cell_idx + 1.0) / float(F)  # (F,)

    for b in range(B):
        starts = gt_boxes[b, :, 0].float().to(device)
        stops = gt_boxes[b, :, 1].float().to(device)
        if clip:
            starts = torch.clamp(starts, 0.0, 1.0)
            stops = torch.clamp(stops, 0.0, 1.0)

        valid = _valid_gt_mask(starts, stops)
        if not valid.any():
            continue
        starts_v = starts[valid]  # (M,)
        stops_v = stops[valid]  # (M,)
        M = starts_v.size(0)

        # compute overlap (M, F)
        # overlap = clamp(min(stop, cell_right) - max(start, cell_left), min=0)
        # use broadcasting
        starts_exp = starts_v.unsqueeze(1)  # (M,1)
        stops_exp = stops_v.unsqueeze(1)  # (M,1)
        left = cell_left.unsqueeze(0)  # (1,F)
        right = cell_right.unsqueeze(0)  # (1,F)

        # (M,F)
        overlap = (torch.min(stops_exp, right) - torch.max(starts_exp, left)).clamp(min=0.0)

        # if no overlap anywhere, continue
        if overlap.sum() == 0:
            continue

        # choose top-k (k = min(P, M)) for each f cell along M dimension
        k = min(P, M)
        if k <= 0:
            continue
        # topk on dim=0 -> returns (k, F)
        topk_vals, topk_idx = overlap.topk(k, dim=0, largest=True, sorted=False)
        # topk_idx has indices in [0, M-1]

        # prepare per-slot targets (P,F)
        f_starts_slots = torch.zeros((P, F), device=device)
        f_stops_slots = torch.zeros((P, F), device=device)
        conf_slots = torch.zeros((P, F), device=device)

        # fill first k rows
        f_starts_slots[:k, :] = starts_v[topk_idx]  # (k,F)
        f_stops_slots[:k, :] = stops_v[topk_idx]
        conf_slots[:k, :] = (topk_vals > 0.0).float()

        # write into targets[b]
        targets[b, :, 0, :] = f_starts_slots
        targets[b, :, 1, :] = f_stops_slots
        targets[b, :, 2, :] = conf_slots

    pos_mask = targets[:, :, 2, :] == 1.0  # (B,P,F)
    n_pos = int(pos_mask.sum().item())
    return targets, pos_mask, n_pos


def build_targets_freq_vectorized_with_time(gt_boxes: torch.Tensor, P: int, T: int, F: int,
                                            device: Optional[torch.device] = None, clip: bool = True) -> Tuple[
    torch.Tensor, torch.Tensor, int]:
    """
    Vectorized builder that returns targets with a time dimension by broadcasting the
    no-time targets across T. Keeps backward-compatible shape (B,P,3,T,F).
    """
    targets_no_time, pos_mask, n_pos = build_targets_freq_vectorized_no_time(gt_boxes, P=P, F=F, device=device,
                                                                             clip=clip)
    # targets_no_time: (B,P,3,F) -> expand to (B,P,3,T,F)
    B = targets_no_time.size(0)
    targets_time = targets_no_time.unsqueeze(3).expand(B, -1, -1, T, -1).contiguous()
    pos_mask_time = pos_mask.unsqueeze(2).expand(B, -1, T, -1).contiguous()
    return targets_time, pos_mask_time, n_pos


def _create_edge_weights(edge_factor, T: int, device: torch.device, mode: str = 'start') -> torch.Tensor:
    """
    Create edge-biased weights for time dimension.
    Weights are higher at the edges (first and last time steps) and lower in the middle.
    edge_factor controls the steepness: higher values -> more concentration at edges.

    Args:
        edge_factor: Controls steepness of edge weights
        T: Number of time steps
        device: Device to create tensor on
        mode: 'start' for weights biased towards beginning, 'stop' for weights biased towards end
    """
    if T == 1:
        return torch.ones(1, device=device)

    # Create time indices
    t = torch.arange(T, device=device, dtype=torch.float32)
    # Normalize to [0, 1]
    t_norm = t / (T - 1) if T > 1 else t

    if mode == 'start':
        # For start predictions: higher weights at the beginning (T=0)
        # Using a decreasing function from 1 to 0
        weights = 1.0 - torch.sigmoid(edge_factor * (t_norm - 0.5) * 10)
    else:  # mode == 'stop'
        # For stop predictions: higher weights at the end (T=T-1)
        # Using an increasing function from 0 to 1
        weights = torch.sigmoid(edge_factor * (t_norm - 0.5) * 10)

    # Normalize to sum to 1
    weights = weights / weights.sum()

    return weights  # shape: (T,)


class FreqDetectionLoss(nn.Module):
    """
    Loss supporting both time-duplicated targets and aggregated-time training.

    Temporal aggregation options:
      - 'none': Expect targets with time dimension and compute per-time losses (legacy).
      - 'soft': Compute softmax weights over time from confidence logits and use them to
                produce weighted-regression predictions (differentiable).
      - 'argmax': Pick the time index with largest confidence logit and use its regression
                  predictions (sparse gradients; non-smooth).
      - 'edge_soft': Similar to 'soft' but with higher weights at the edges of time dimension,
                     controlled by edge_factor parameter.
      - 'edge': Directly use values from both ends of time dimension (first and last time steps).

    Additional parameters:
      - reg_loss_type: Type of regression loss ('smoothl1' (default) or 'mse')
      - temp: Temperature parameter for softmax when temporal_agg == 'soft'
      - edge_factor: Controls steepness of edge weights when temporal_agg == 'edge_soft'
    """

    def __init__(self, P: int, lambda_coord: float = 5.0, noobj_weight: float = 0.5, temporal_agg: str = 'soft',
                 reg_loss_type: str = 'smoothl1', temp: float = 1.0, edge_factor: float = 1.0,
                 device: Optional[torch.device] = None):
        super().__init__()
        assert temporal_agg in ('soft', 'argmax', 'none', 'edge_soft', 'edge')  # 添加新模式
        assert reg_loss_type in ('smoothl1', 'mse')
        self.P = P
        self.lambda_coord = float(lambda_coord)
        self.noobj_weight = float(noobj_weight)
        self.temporal_agg = temporal_agg
        self.reg_loss_type = reg_loss_type
        self.temp = float(temp)
        self.edge_factor = float(edge_factor)  # 控制边缘权重的陡峭度
        self.device = device
        self.gt_norm = None

        if reg_loss_type == 'mse':
            self.reg_crit = nn.MSELoss(reduction='sum')
        else:
            self.reg_crit = nn.SmoothL1Loss(reduction='sum')

        self.conf_crit = nn.BCEWithLogitsLoss(reduction='none')

    def _forward_single(self, raw_preds: torch.Tensor, gt_boxes: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Single-level loss computation extracted from original forward.
        raw_preds: (B, P, 3, T, F)
        gt_boxes: (B, N, 2) normalized (f_start, f_stop)
        Returns: (total_loss, metrics_dict)
        """
        if self.device is None:
            self.device = raw_preds.device

        if raw_preds.dim() != 5 or raw_preds.size(2) != 3:
            raise ValueError("raw_preds must have shape (B, P, 3, T, F)")

        B, Pp, _, T, _F = raw_preds.shape
        if Pp != self.P:
            raise ValueError(f"raw_preds has P={Pp} but loss was initialized with P={self.P}")

        # choose behavior according to temporal aggregation mode
        if self.temporal_agg == 'none':
            # build per-time targets
            targets, pos_mask, n_pos = build_targets_freq_vectorized_with_time(gt_boxes, P=self.P, T=T, F=_F,
                                                                               device=self.device)

            pred_start = raw_preds[:, :, 0, :, :]  # (B,P,T,F)
            pred_stop = raw_preds[:, :, 1, :, :]
            pred_conf_logits = raw_preds[:, :, 2, :, :]

            tgt_start = targets[:, :, 0, :, :]
            tgt_stop = targets[:, :, 1, :, :]
            tgt_conf = targets[:, :, 2, :, :]

            # regression loss on positives (per-time)
            n_pos_val = int(pos_mask.sum().item())
            if n_pos_val > 0:
                reg_loss_start = self.reg_crit(pred_start[pos_mask], tgt_start[pos_mask])
                reg_loss_stop = self.reg_crit(pred_stop[pos_mask], tgt_stop[pos_mask])
                reg_loss = (reg_loss_start + reg_loss_stop) / float(n_pos_val)
            else:
                reg_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)

            # confidence loss over all slots (B,P,T,F), downweight negatives
            bce_all = self.conf_crit(pred_conf_logits, tgt_conf)  # (B,P,T,F)
            weight_mask = tgt_conf.float() + (1.0 - tgt_conf.float()) * self.noobj_weight
            conf_loss = (bce_all * weight_mask).sum() / float(B * self.P * T * _F)

            total_loss = self.lambda_coord * reg_loss + conf_loss

            metrics = {
                'loss_reg': reg_loss.detach().cpu().item() if isinstance(reg_loss, torch.Tensor) else float(reg_loss),
                'loss_conf': conf_loss.detach().cpu().item() if isinstance(conf_loss, torch.Tensor) else float(
                    conf_loss),
                'loss_total': total_loss.detach().cpu().item() if isinstance(total_loss, torch.Tensor) else float(
                    total_loss),
                'n_pos': n_pos_val
            }
            return total_loss, metrics

        else:
            # build no-time targets (B,P,3,F)
            targets_nt, pos_mask_nt, n_pos = build_targets_freq_vectorized_no_time(gt_boxes, P=self.P, F=_F,
                                                                                   device=self.device)

            pred_start = raw_preds[:, :, 0, :, :]  # (B,P,T,F)
            pred_stop = raw_preds[:, :, 1, :, :]
            pred_conf_logits = raw_preds[:, :, 2, :, :]  # (B,P,T,F)

            # aggregate confidence across time: max logit over time for conf loss
            pred_conf_logits_max = pred_conf_logits.max(dim=2).values  # (B,P,F)

            # regression aggregation across time
            if self.temporal_agg == 'argmax':
                # pick argmax time index for each (B,P,F)
                t_star = pred_conf_logits.argmax(dim=2)  # (B,P,F)
                b_idx = torch.arange(B, device=self.device)[:, None, None].expand(B, self.P, _F)
                p_idx = torch.arange(self.P, device=self.device)[None, :, None].expand(B, self.P, _F)
                f_idx = torch.arange(_F, device=self.device)[None, None, :].expand(B, self.P, _F)
                pred_start_agg = pred_start[b_idx, p_idx, t_star, f_idx]
                pred_stop_agg = pred_stop[b_idx, p_idx, t_star, f_idx]
            elif self.temporal_agg == 'edge_soft':
                # 'edge_soft' aggregation: use different edge-biased weights for start and stop
                start_weights = _create_edge_weights(self.edge_factor, T, self.device, mode='start')  # (T,)
                stop_weights = _create_edge_weights(self.edge_factor, self.device, mode='stop')  # (T,)
                # Expand to (1,1,T,1) for broadcasting to (B,P,T,F)
                start_weights = start_weights.view(1, 1, T, 1)
                stop_weights = stop_weights.view(1, 1, T, 1)
                # Use start-biased weights for start predictions, stop-biased weights for stop predictions
                pred_start_agg = (pred_start * start_weights).sum(dim=2)  # (B,P,F)
                pred_stop_agg = (pred_stop * stop_weights).sum(dim=2)  # (B,P,F)
            elif self.temporal_agg == 'edge':
                # 'edge' aggregation: use start from first time step and stop from last time step
                pred_start_agg = pred_start[:, :, 0, :]  # (B,P,F) - first time step
                pred_stop_agg = pred_stop[:, :, -1, :]  # (B,P,F) - last time step
            else:
                # 'soft' aggregation (differentiable) with temperature
                if self.temp <= 0.0:
                    raise ValueError("temp must be > 0 for soft aggregation")
                weights = F.softmax(pred_conf_logits / (self.temp + 1e-12), dim=2)  # (B,P,T,F)
                pred_start_agg = (pred_start * weights).sum(dim=2)  # (B,P,F)
                pred_stop_agg = (pred_stop * weights).sum(dim=2)

            tgt_start = targets_nt[:, :, 0, :]
            tgt_stop = targets_nt[:, :, 1, :]
            tgt_conf = targets_nt[:, :, 2, :]

            # regression loss only on positive slots (no-time)
            n_pos_val = int(pos_mask_nt.sum().item())
            if n_pos_val > 0:
                reg_loss_start = self.reg_crit(pred_start_agg[pos_mask_nt], tgt_start[pos_mask_nt])
                reg_loss_stop = self.reg_crit(pred_stop_agg[pos_mask_nt], tgt_stop[pos_mask_nt])
                reg_loss = (reg_loss_start + reg_loss_stop) / float(n_pos_val)
            else:
                reg_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)

            # confidence loss on aggregated logits (B,P,F)
            bce_all = self.conf_crit(pred_conf_logits_max, tgt_conf)  # (B,P,F)
            weight_mask = tgt_conf.float() + (1.0 - tgt_conf.float()) * self.noobj_weight
            conf_loss = (bce_all * weight_mask).sum() / float(B * self.P * _F)

            total_loss = self.lambda_coord * reg_loss + conf_loss

            metrics = {
                'loss_reg': reg_loss.detach().cpu().item() if isinstance(reg_loss, torch.Tensor) else float(reg_loss),
                'loss_conf': conf_loss.detach().cpu().item() if isinstance(conf_loss, torch.Tensor) else float(
                    conf_loss),
                'loss_total': total_loss.detach().cpu().item() if isinstance(total_loss, torch.Tensor) else float(
                    total_loss),
                'n_pos': n_pos_val
            }
            return total_loss, metrics

    def forward(self, raw_preds, gt_boxes, level_weights: Optional[List[float]] = None) -> Tuple[torch.Tensor, dict]:
        """
        Forward supports either:
          - raw_preds: single Tensor (B,P,3,T,F)  -> returns (_forward_single output)
          - raw_preds: list/tuple of Tensors [(B,P,3,T1,F1), (B,P,3,T2,F2), ...] -> per-level loss + weighted sum
        gt_boxes: (B,N,2) (normalized or non-normalized)
        fchans: Number of frequency channels (F)
        level_weights: optional list of floats to weight each level's loss (length must match number of preds)
        """

        # single tensor path
        if isinstance(raw_preds, torch.Tensor):
            return self._forward_single(raw_preds, gt_boxes)

        # list/tuple path
        if not isinstance(raw_preds, (list, tuple)):
            raise TypeError("raw_preds must be a Tensor or a list/tuple of Tensors")

        L = len(raw_preds)
        if L == 0:
            raise ValueError("raw_preds list is empty")

        if level_weights is None:
            level_weights = [1.0] * L
        if len(level_weights) != L:
            raise ValueError("level_weights length must match number of prediction levels")

        # ensure device set
        if self.device is None:
            # pick device of first level
            first = raw_preds[0]
            if isinstance(first, torch.Tensor):
                self.device = first.device

        total_loss = None
        metrics_acc = {'loss_reg': 0.0, 'loss_conf': 0.0, 'loss_total': 0.0, 'n_pos': 0}
        for pred, w in zip(raw_preds, level_weights):
            loss_l, metrics_l = self._forward_single(pred, gt_boxes)
            # accumulate weighted loss (loss_l is a tensor)
            if total_loss is None:
                total_loss = w * loss_l
            else:
                total_loss = total_loss + w * loss_l
            metrics_acc['loss_reg'] += w * metrics_l.get('loss_reg', 0.0)
            metrics_acc['loss_conf'] += w * metrics_l.get('loss_conf', 0.0)
            metrics_acc['n_pos'] += metrics_l.get('n_pos', 0)

        # finalize metrics
        metrics_acc['loss_total'] = total_loss.detach().cpu().item() if isinstance(total_loss, torch.Tensor) else float(
            total_loss)
        return total_loss, metrics_acc


class DetectionCombinedLoss(nn.Module):
    """
    Combined loss for denoising and frequency detection tasks.

    Combines:
    1. Frequency detection loss (FreqDetectionLoss)
    2. Denoising loss (MSE or Dice)

    Args:
        P: Number of slots for frequency detection
        lambda_denoise: Weight factor for denoising loss
        loss_type: Type of denoising loss ('mse' or 'dice')
        detection_loss_kwargs: Keyword arguments for FreqDetectionLoss
        device: Torch device
    """

    def __init__(self, P: int, lambda_denoise: float = 1.0, loss_type: str = 'mse', avg_time: bool = False,
                 lambda_learnable: bool = False, detection_loss_kwargs: Optional[dict] = None,
                 device: Optional[torch.device] = None):
        super().__init__()

        assert loss_type in ('mse', 'dice'), "denoise_loss_type must be 'mse' or 'dice'"

        self.P = P
        self.lambda_denoise = nn.Parameter(
            torch.tensor(lambda_denoise, device=device)) if lambda_learnable else lambda_denoise
        self.loss_type = loss_type
        self.avg_time = avg_time
        self.device = device

        # Initialize frequency detection loss
        detection_loss_kwargs = detection_loss_kwargs or {}
        self.detection_loss = FreqDetectionLoss(P=P, device=device, **detection_loss_kwargs)

        # Initialize denoising loss
        if loss_type == 'mse':
            self.denoise_loss = nn.MSELoss(reduction='mean')
        else:  # dice
            self.denoise_loss = DiceLoss()

    def forward(self, raw_preds, denoised, clean, gt_boxes,
                level_weights: Optional[List[float]] = None) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass for combined loss calculation.

        Args:
            raw_preds: Frequency detection predictions (single tensor or list of tensors)
            denoised: Denoised spectrum prediction (B, F) or (B, T, F)
            clean: Clean target spectrum (B, F) or (B, T, F)
            gt_boxes: Ground truth boxes for frequency detection (B, N, 2)
            level_weights: Optional weights for different prediction levels

        Returns:
            total_loss: Combined loss value
            metrics: Dictionary containing all loss components and metrics
        """
        # Calculate frequency detection loss
        detection_loss, detection_metrics = self.detection_loss(raw_preds, gt_boxes, level_weights)

        # Calculate denoising loss
        if self.avg_time:
            # Average over time dimension T to get (B, C, F)
            denoised_mean = denoised.mean(dim=2)  # (B, C, T, F) -> (B, C, F)
            clean_mean = clean.mean(dim=2)  # (B, C, T, F) -> (B, C, F)
            if denoised_mean.dim() == 3 and denoised_mean.size(1) > 1:
                denoised_mean = denoised_mean.mean(dim=1)  # (B, C, F) -> (B, F)
                clean_mean = clean_mean.mean(dim=1)  # (B, C, F) -> (B, F)
            if self.loss_type == 'dice':
                clean_mean = clean_mean.sigmoid()
            denoised_loss = self.denoise_loss(denoised_mean, clean_mean)
        else:
            if self.loss_type == 'dice':
                clean = clean.sigmoid()
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
            metrics[f'det_{k}'] = v

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


# ----------------------
# Unit-test / smoke demo (updated for list-of-levels support)
# ----------------------
if __name__ == '__main__':
    torch.manual_seed(0)
    B = 2
    C = 1
    P = 2
    T = 6
    _F = 16
    N = 3
    device = torch.device('cpu')

    # create random raw preds (logits/regression) for single-level test
    raw_preds = torch.randn((B, P, 3, T, _F), device=device) * 0.5  # smaller noise

    # make a synthetic pattern: for sample 0 put one GT around f~0.3-0.45, sample1 one at 0.6-0.7
    gt_boxes = torch.full((B, N, 2), float('nan'), device=device)
    gt_boxes[0, 0, :] = torch.tensor([0.28, 0.45])
    gt_boxes[0, 1, :] = torch.tensor([0.9, 0.95])  # narrow
    gt_boxes[1, 0, :] = torch.tensor([0.58, 0.72])

    print(
        "[\033[32mInfo\033[0m] Running single-level demo for temporal_agg modes with reg_loss_type='smoothl1' and temp=0.5")
    for mode in ('soft', 'argmax', 'none'):
        loss_module = FreqDetectionLoss(P=P, lambda_coord=5.0, noobj_weight=0.5,
                                        temporal_agg=mode, reg_loss_type='smoothl1', temp=0.5, device=device)
        loss, metrics = loss_module(raw_preds.clone(), gt_boxes)
        print(
            f"mode={mode:6s} -> total={metrics['loss_total']:.6f}, reg={metrics['loss_reg']:.6f}, conf={metrics['loss_conf']:.6f}, n_pos={metrics['n_pos']}")

    # show effect of regression loss type (single-level)
    print("\n[\033[32mInfo\033[0m] Compare reg loss types (soft agg):")
    for regt in ('smoothl1', 'mse'):
        loss_module = FreqDetectionLoss(P=P, temporal_agg='soft', reg_loss_type=regt, temp=0.8, device=device)
        loss, metrics = loss_module(raw_preds.clone(), gt_boxes)
        print(f"reg={regt:8s} -> total={metrics['loss_total']:.6f}, reg={metrics['loss_reg']:.6f}")

    # ----------------------
    # Multi-level (list) test: build three levels with different (T,F)
    # ----------------------
    print("\n[\033[32mInfo\033[0m] Running multi-level demo (list of preds with different T/F)")

    # level A: same as single-level above (T=6, F=16)
    pred_A = raw_preds.clone()

    # level B: coarser freq (F=8) and shorter time (T=4)
    pred_B = torch.randn((B, P, 3, 4, 8), device=device) * 0.5

    # level C: even coarser freq (F=4) and shorter time (T=2)
    pred_C = torch.randn((B, P, 3, 2, 4), device=device) * 0.5

    preds_list = [pred_A, pred_B, pred_C]

    # Use the loss module (supports list input). Use default equal level weights here.
    loss_module = FreqDetectionLoss(P=P, lambda_coord=5.0, noobj_weight=0.5,
                                    temporal_agg='soft', reg_loss_type='smoothl1', temp=0.6, device=device)

    loss, metrics = loss_module(preds_list, gt_boxes)  # list input supported
    print(
        f"multi-level -> total={metrics['loss_total']:.6f}, reg={metrics['loss_reg']:.6f}, conf={metrics['loss_conf']:.6f}, n_pos_total={metrics['n_pos']}")

    # optionally: inspect per-level losses by calling internal _forward_single (for debugging)
    print("\n[\033[32mInfo\033[0m] Per-level breakdown (debug):")
    for i, p in enumerate(preds_list):
        loss_i, met_i = loss_module._forward_single(p, gt_boxes)
        print(
            f" level {i}: total={met_i['loss_total']:.6f}, reg={met_i['loss_reg']:.6f}, conf={met_i['loss_conf']:.6f}, n_pos={met_i['n_pos']}")

    # Initialize the combined loss
    print("\n[\033[32mInfo\033[0m] Running combined loss demo (detection + denoising)")
    denoised = torch.randn((B, C, T, _F), device=device)
    clean = torch.randn((B, C, T, _F), device=device)
    combined_loss = DetectionCombinedLoss(P=2, lambda_denoise=10, loss_type='mse', avg_time=True, device=device,
                                          detection_loss_kwargs={'lambda_coord': 5.0,
                                                                 'noobj_weight': 0.5,
                                                                 'temporal_agg': 'soft',
                                                                 'reg_loss_type': 'smoothl1',
                                                                 'temp': 0.5})

    # Calculate loss
    total_loss, metrics = combined_loss(raw_preds=raw_preds, denoised=denoised, clean=clean, gt_boxes=gt_boxes)

    print(f"Total loss: {metrics['total_loss']:.4f}")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print("\n[\033[32mInfo\033[0m] Demo complete.")
