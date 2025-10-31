from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F


def decode_F(out: Dict[str, torch.Tensor], iou_thresh: float = 0.5, score_thresh: float = 0.2) -> Dict[
    str, torch.Tensor]:
    """
    Decode raw logits to normalized probabilities (0-1).
    Args:
        out: dict with keys "f_start", "f_end", "class_logits", "confidence"
        iou_thresh: IoU threshold for filtering overlapping signals in f dimension.
        score_thresh: Confidence threshold for filtering low-score signals.
    Returns: dict with same keys but values possibly transformed to probabilities,
             and confidence adjusted to 0 for filtered signals.
    """
    f_start = out["f_start"]
    f_end = out["f_end"]
    class_logits = out["class_logits"]
    confidence = out["confidence"]
    f_start = torch.sigmoid(f_start)
    f_end = torch.sigmoid(f_end)
    confidence = torch.sigmoid(confidence)
    class_prob = F.softmax(class_logits, dim=-1)

    # Apply filtering: set confidence to 0 for low scores or overlapping signals
    B, N = confidence.shape[:2]  # Assume shape [batch, N, ...] but for confidence [B, N]
    for i in range(B):
        conf = confidence[i]

        # Apply score threshold
        low_mask = conf < score_thresh
        confidence[i][low_mask] = 0

        # Apply IoU filtering (NMS-like, set conf to 0 for suppressed)
        active_mask = confidence[i] > 0
        if not active_mask.any():
            continue

        active_indices = torch.nonzero(active_mask).squeeze(-1)
        scores = confidence[i][active_indices]
        starts = f_start[i][active_indices]
        ends = f_end[i][active_indices]

        # Sort by scores descending
        sort_idx = torch.argsort(scores, descending=True)
        active_indices = active_indices[sort_idx]
        starts = starts[sort_idx]
        ends = ends[sort_idx]
        scores = scores[sort_idx]

        # Perform NMS
        suppress = torch.zeros(len(scores), dtype=torch.bool, device=scores.device)
        for j in range(len(scores)):
            if suppress[j]:
                continue
            for k in range(j + 1, len(scores)):
                if suppress[k]:
                    continue
                inter = max(0.0, min(ends[j], ends[k]) - max(starts[j], starts[k]))
                union = max(ends[j], ends[k]) - min(starts[j], starts[k]) + 1e-10
                iou = inter / union
                if iou > iou_thresh:
                    suppress[k] = True

        # Set confidence to 0 for suppressed
        supp_indices = active_indices[suppress]
        confidence[i][supp_indices] = 0

    return {"f_start": f_start, "f_end": f_end, "class": class_prob, "confidence": confidence}


def plot_F_lines(ax, freqs, pred_boxes, normalized=True, color=['red', 'green'], linestyle='--', linewidth=0.5):
    """
    Plot detected frequency start/stop as **vertical lines** (not boxes).
    Preserves physical meaning: f_start > f_stop means negative drift.
    Supports arbitrary class IDs with strict color validation.

    Args:
        ax (matplotlib.axes.Axes): The axis to plot on.
        freqs (array-like): Array of frequency values (length = fchans).
        pred_boxes (tuple): (N, classes, f_starts, f_stops)
            - N: int, number of signals
            - classes: list/array of int class IDs (>=0)
            - f_starts: list/array of start freq (normalized or pixel)
            - f_stops:  list/array of stop freq
        normalized (bool): If True, f_starts/f_stops in [0,1]
        color (list[str]): List of colors, must have length >= (max_class_id + 1)
        linestyle, linewidth: matplotlib line style
    """
    if len(pred_boxes) == 4:
        N, classes, f_starts, f_stops = pred_boxes
    elif len(pred_boxes) == 3:
        N, f_starts, f_stops = pred_boxes
        classes = np.zeros(N, dtype=int)  # default class 0
    else:
        raise ValueError(f"[\033[31mError\033[0m] Invalid pred_boxes format: {pred_boxes}")

    if N == 0:
        return

    def to_numpy(x):
        if isinstance(x, list):
            if len(x) > 0 and isinstance(x[0], torch.Tensor):
                x = torch.cat([t.flatten() for t in x])
            else:
                x = np.array(x)
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        return np.asarray(x)

    classes = to_numpy(classes).astype(int)
    f_starts = to_numpy(f_starts)
    f_stops = to_numpy(f_stops)

    if np.any(classes < 0):
        raise ValueError(f"[\033[31mError\033[0m] Class IDs must be >= 0. Got: {classes}")

    max_class_id = int(classes.max())
    if color is None or not isinstance(color, (list, tuple)):
        raise ValueError("[\033[31mError\033[0m] `color` must be a list of color strings.")
    if len(color) <= max_class_id:
        raise ValueError(
            f"[\033[31mError\033[0m] Not enough colors: need {max_class_id + 1}, got {len(color)}. "
            f"Class IDs present: {np.unique(classes).tolist()}"
        )
    colors = list(color)

    valid_mask = (np.isfinite(f_starts) & np.isfinite(f_stops))
    if normalized:
        valid_mask &= (f_starts >= 0) & (f_starts <= 1) & (f_stops >= 0) & (f_stops <= 1)
    else:
        valid_mask &= (f_starts >= 0) & (f_starts < len(freqs)) & (f_stops >= 0) & (f_stops < len(freqs))

    f_starts = f_starts[valid_mask]
    f_stops = f_stops[valid_mask]
    classes = classes[valid_mask]

    if len(f_starts) == 0:
        print(f"[\033[33mWarn\033[0m] No valid frequency lines to plot after filtering.")
        return

    if normalized:
        scale = len(freqs) - 1
        start_idx = np.rint(f_starts * scale).astype(int)
        stop_idx = np.rint(f_stops * scale).astype(int)
    else:
        start_idx = f_starts.astype(int)
        stop_idx = f_stops.astype(int)

    start_idx = np.clip(start_idx, 0, len(freqs) - 1)
    stop_idx = np.clip(stop_idx, 0, len(freqs) - 1)

    for cls, s_idx, e_idx in zip(classes, start_idx, stop_idx):
        col = colors[cls]
        ax.axvline(freqs[s_idx], color=col, linestyle=linestyle, linewidth=linewidth, alpha=0.8)
        ax.axvline(freqs[e_idx], color=col, linestyle=linestyle, linewidth=linewidth, alpha=0.8)
