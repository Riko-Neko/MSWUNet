from typing import Optional, List

import numpy as np
import torch
import torch.nn.functional as F

from utils.loss_func import _create_edge_weights


def decode_F(raw: torch.Tensor, mode='none', temp: float = 0.5, edge_factor: float = 1.0) -> torch.Tensor:
    """
    Decode raw predictions for global normalized [0,1] f_start/f_stop with temporal aggregation.

    Args:
        raw: (B, P, 3, T, F) where 3=(f_start_logits, f_stop_logits, conf_logits)
        mode: temporal aggregation options ('soft', 'argmax', 'edge', 'edge_soft' or 'none')
        temp: temperature for softmax temporal aggregation (used only in 'soft' mode)
        edge_factor: controls steepness of edge weights (used only in 'edge_soft' mode)

    Returns:
        det_out: (B, N, 3) -> [f_start, f_stop, conf], normalized to [0,1]
                 where N depends on mode:
                 - 'soft'/'argmax'/'edge'/'edge_soft': N = F * P (time dimension aggregated)
                 - 'none': N = T * F * P (time dimension preserved)
    """
    B, P, _, T, _F = raw.shape
    device = raw.device
    dtype = raw.dtype

    if mode == 'none':
        # No temporal aggregation - preserve time dimension
        # (B, P, 3, T, F) -> (B, T, F, P, 3)
        r = raw.permute(0, 3, 4, 1, 2).contiguous()

        # extract channels
        f_start = r[..., 0]  # (B, T, F, P)
        f_stop = r[..., 1]
        conf_logits = r[..., 2]

        # Flatten T, F, and P dims -> N = T * F * P
        Ncells = T * _F * P
        f_start = f_start.view(B, Ncells)  # (B, N)
        f_stop = f_stop.view(B, Ncells)
        conf = conf_logits.view(B, Ncells)

        det_out = torch.stack([f_start, f_stop, torch.sigmoid(conf)], dim=2)  # (B, N, 3)
        return det_out

    elif mode == 'argmax':
        # (B, P, 3, T, F) -> (B, T, F, P, 3)
        r = raw.permute(0, 3, 4, 1, 2).contiguous()

        # extract channels
        f_start = r[..., 0]  # (B, T, F, P)
        f_stop = r[..., 1]
        conf_logits = r[..., 2]

        # Pick argmax time index for each (B, F, P)
        t_star = conf_logits.argmax(dim=1)  # (B, F, P)

        # Gather predictions using argmax indices
        b_idx = torch.arange(B, device=device)[:, None, None].expand(B, _F, P)
        f_idx = torch.arange(_F, device=device)[None, :, None].expand(B, _F, P)
        p_idx = torch.arange(P, device=device)[None, None, :].expand(B, _F, P)

        f_start_agg = f_start[b_idx, t_star, f_idx, p_idx]  # (B, F, P)
        f_stop_agg = f_stop[b_idx, t_star, f_idx, p_idx]  # (B, F, P)
        conf = conf_logits[b_idx, t_star, f_idx, p_idx]  # (B, F, P)

    elif mode == 'soft':
        # (B, P, 3, T, F) -> (B, T, F, P, 3)
        r = raw.permute(0, 3, 4, 1, 2).contiguous()

        # extract channels
        f_start = r[..., 0]  # (B, T, F, P)
        f_stop = r[..., 1]
        conf_logits = r[..., 2]

        # soft temporal aggregation (matches soft mode in FreqDetectionLoss)
        conf_weights = F.softmax(conf_logits / (temp + 1e-12), dim=1)  # (B, T, F, P)

        # weighted sum over time (dim=1)
        f_start_agg = (f_start * conf_weights).sum(dim=1)  # (B, F, P)
        f_stop_agg = (f_stop * conf_weights).sum(dim=1)  # (B, F, P)
        conf = conf_logits.max(dim=1)[0]  # (B, F, P), max conf over time

    elif mode == 'edge':
        # (B, P, 3, T, F) -> (B, T, F, P, 3)
        r = raw.permute(0, 3, 4, 1, 2).contiguous()

        # extract channels
        f_start = r[..., 0]  # (B, T, F, P)
        f_stop = r[..., 1]
        conf_logits = r[..., 2]

        # Use start from first time step and stop from last time step
        f_start_agg = f_start[:, 0, :, :]  # (B, F, P) - 取第一个时间步
        f_stop_agg = f_stop[:, -1, :, :]  # (B, F, P) - 取最后一个时间步
        conf = conf_logits.max(dim=1)[0]  # (B, F, P), max conf over time

    elif mode == 'edge_soft':
        # (B, P, 3, T, F) -> (B, T, F, P, 3)
        r = raw.permute(0, 3, 4, 1, 2).contiguous()

        # extract channels
        f_start = r[..., 0]  # (B, T, F, P)
        f_stop = r[..., 1]
        conf_logits = r[..., 2]

        # Create edge-biased weights for start and stop predictions
        start_weights = _create_edge_weights(edge_factor, T, device, mode='start')  # (T,)
        stop_weights = _create_edge_weights(edge_factor, T, device, mode='stop')  # (T,)

        # Expand to (1, T, 1, 1) for broadcasting to (B, T, F, P)
        start_weights = start_weights.view(1, T, 1, 1)
        stop_weights = stop_weights.view(1, T, 1, 1)

        # Use start-biased weights for start predictions, stop-biased weights for stop predictions
        f_start_agg = (f_start * start_weights).sum(dim=1)  # (B, F, P)
        f_stop_agg = (f_stop * stop_weights).sum(dim=1)  # (B, F, P)
        conf = conf_logits.max(dim=1)[0]  # (B, F, P), max conf over time

    else:
        raise ValueError(f"Unsupported mode: {mode}. Must be 'soft', 'argmax', 'edge', 'edge_soft' or 'none'")

    # For 'argmax', 'soft', 'edge', and 'edge_soft' modes: flatten F and P dims -> N = F * P
    Ncells = _F * P
    f_start_flat = f_start_agg.view(B, Ncells)  # (B, N)
    f_stop_flat = f_stop_agg.view(B, Ncells)
    conf_flat = conf.view(B, Ncells)

    det_out = torch.stack([f_start_flat, f_stop_flat, torch.sigmoid(conf_flat)], dim=2)  # (B, N, 3)
    return det_out


def nms_1d(det_out: torch.Tensor, iou_thresh: float = 0.5, top_k: Optional[int] = None,
           score_thresh: Optional[float] = None, eps: float = 1e-13) -> List[torch.Tensor]:
    """
    Batched 1D NMS for intervals.

    Args:
        det_out: (B, N, 3) or (N, 3) -> [start, stop, score]
        iou_thresh: IoU threshold for suppression
        top_k: max number of intervals to keep per batch (after NMS)
        score_thresh: if not None, drop intervals with score < score_thresh before NMS
        eps: small number to avoid div-by-zero

    Returns:
        results: list of length B, each is tensor (M,3) on same device/dtype as det_out
    """
    if det_out.ndim == 2:
        det_out = det_out.unsqueeze(0)  # make it (1, N, 3)

    B, N, C = det_out.shape
    assert C >= 3, "det_out last dim should be >= 3 (start, stop, score)"

    device = det_out.device
    dtype = det_out.dtype
    results = []

    for b in range(B):
        boxes = det_out[b]  # (N,3)
        starts = boxes[:, 0]
        stops = boxes[:, 1]
        scores = boxes[:, 2]

        # basic validity mask: stop >= start and finite values
        valid = (stops >= starts) & torch.isfinite(starts) & torch.isfinite(stops) & torch.isfinite(scores)
        if score_thresh is not None:
            valid &= (scores >= score_thresh)

        if not valid.any():
            results.append(torch.empty((0, 3), dtype=dtype, device=device))
            continue

        boxes = boxes[valid]
        starts = boxes[:, 0]
        stops = boxes[:, 1]
        scores = boxes[:, 2]

        # sort by score descending
        order = torch.argsort(scores, descending=True)
        keep_idx = []  # indices into `boxes`

        # standard greedy NMS loop (works with indices into boxes)
        while order.numel() > 0:
            i = int(order[0].item())  # index into boxes
            keep_idx.append(i)

            if order.numel() == 1:
                break

            others = order[1:]  # tensor of indices
            # compute intersection/union with current box i
            inter_start = torch.maximum(starts[i], starts[others])
            inter_stop = torch.minimum(stops[i], stops[others])
            inter = (inter_stop - inter_start).clamp(min=0.0)

            union = (stops[i] - starts[i]) + (stops[others] - starts[others]) - inter
            ious = inter / (union + eps)

            # keep those with IoU <= threshold
            keep_mask = ious <= iou_thresh
            order = order[1:][keep_mask]

        # gather kept boxes (preserves device/dtype)
        if len(keep_idx) == 0:
            kept = torch.empty((0, 3), dtype=dtype, device=device)
        else:
            keep_idx_tensor = torch.as_tensor(keep_idx, dtype=torch.long, device=device)
            kept = boxes[keep_idx_tensor]

        # optionally trim top_k
        if top_k is not None and kept.size(0) > top_k:
            kept = kept[:top_k]

        results.append(kept)

    return results


def plot_F_lines(ax, freqs, pred_boxes, normalized=True, color=['red', 'green'], linestyle='--', linewidth=0.5):
    """
    Plot detected frequency intervals as vertical lines on a 1D frequency plot.

    Args:
        ax (matplotlib.axes.Axes): The axis to plot on.
        freqs (array-like): Array of frequency values.
        pred_boxes (tuple): (N, classes, f_starts, f_stops)
            - N: number of intervals
            - classes: list/array of class ids (0 or 1)
            - f_starts: list/array of start indices (or normalized [0,1] values)
            - f_stops: list/array of stop indices (or normalized [0,1] values)
        normalized (bool): Whether the boxes are normalized to [0, 1].
        color (str | list[str]): Either a single color string (e.g., 'red')
                                 or a list of two colors [color_class0, color_class1].
        linestyle (str): Line style.
        linewidth (float): Line width.
    """
    if len(pred_boxes) == 4:
        N, classes, f_starts, f_stops = pred_boxes
    elif len(pred_boxes) == 3:
        N, f_starts, f_stops = pred_boxes
        classes = np.ones_like(f_starts, dtype=int)
    else:
        raise ValueError(f"[\033[31mError\033[0m] Invalid pred_boxes format: {pred_boxes}")

    if N == 0:
        return

    # --- 统一颜色配置 ---
    if isinstance(color, str):
        colors = [color, color]
    elif isinstance(color, (list, tuple)) and len(color) == 2:
        colors = list(color)
    else:
        raise ValueError("[\033[31mError\033[0m] `color` must be a string or a list/tuple of two color strings.")

    def to_numpy(x):
        if isinstance(x, list):
            x = torch.cat([t.flatten() for t in x]) if len(x) > 0 else torch.tensor([])
            return x.cpu().numpy()
        elif torch.is_tensor(x):
            return x.cpu().numpy()
        else:
            return np.asarray(x)

    classes = to_numpy(classes)
    f_starts = to_numpy(f_starts)
    f_stops = to_numpy(f_stops)

    valid_mask = (np.isfinite(f_starts) & np.isfinite(f_stops))
    if normalized:
        valid_mask &= ((f_starts >= 0) & (f_starts <= 1) &
                       (f_stops >= 0) & (f_stops <= 1))
    else:
        valid_mask &= ((f_starts >= 0) & (f_starts < len(freqs)) &
                       (f_stops >= 0) & (f_stops < len(freqs)))

    f_starts = f_starts[valid_mask]
    f_stops = f_stops[valid_mask]
    classes = classes[valid_mask]
    N_valid = len(f_starts)
    if N_valid == 0:
        print(f"[\033[33mWarn\033[0m] No valid boxes after filtering in plot_F_lines")
        return

    if normalized:
        f_starts = np.rint(f_starts * (len(freqs) - 1)).astype(int)
        f_stops = np.rint(f_stops * (len(freqs) - 1)).astype(int)
    else:
        f_starts = f_starts.astype(int)
        f_stops = f_stops.astype(int)

    for cls, f_start, f_stop in zip(classes, f_starts, f_stops):
        col = colors[int(cls)] if int(cls) < len(colors) else colors[0]
        ax.axvline(freqs[f_start], color=col, linestyle=linestyle, linewidth=linewidth)
        ax.axvline(freqs[f_stop], color=col, linestyle=linestyle, linewidth=linewidth)
