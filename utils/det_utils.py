from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F


def decode_F(out: Dict[str, torch.Tensor], iou_thresh: float = 0.5, score_thresh: float = 0.2,
             apply_filtering: bool = True) -> Dict[str, torch.Tensor]:
    """
    Decode raw logits to normalized probabilities (0-1).
    Args:
        out: dict with keys "f_start", "f_end", "class_logits", "confidence"
        iou_thresh: IoU threshold for filtering overlapping signals in f dimension.
        score_thresh: Confidence threshold for filtering low-score signals.
        apply_filtering: If True, apply filtering overlapping signals.
    Returns:
        dict with keys:
            - "f_start": (B, M)
            - "f_end": (B, M)
            - "class_id": (B, M) integer class indices
            - "confidence": (B, M)
        Where M is the number of valid signals after filtering for the first batch.
        Other batches are cropped or padded to match M.
        Padded entries have confidence=0, class_id=-1, f_start=0, f_end=0.
    """
    DEBUG = False
    f_start = torch.sigmoid(out["f_start"])
    f_end = torch.sigmoid(out["f_end"])
    confidence = torch.sigmoid(out["confidence"])
    class_prob = F.softmax(out["class_logits"], dim=-1)
    class_id = torch.argmax(class_prob, dim=-1)  # shape: (B, N)
    B, N = confidence.shape
    device = confidence.device
    results = {"f_start": [], "f_end": [], "class": [], "confidence": []}

    for i in range(B):
        if not apply_filtering:
            kept_indices = torch.arange(N, device=device)
        else:
            conf = confidence[i]
            low_mask = conf < score_thresh
            active_mask = ~low_mask
            if not active_mask.any():
                kept_indices = torch.tensor([], dtype=torch.long, device=device)
            else:
                active_indices = torch.where(active_mask)[0]
                scores = conf[active_indices]
                starts = f_start[i][active_indices]
                ends = f_end[i][active_indices]

                # Sort descending by score
                sort_idx = torch.argsort(scores, descending=True)
                active_indices = active_indices[sort_idx]
                starts = starts[sort_idx]
                ends = ends[sort_idx]
                scores = scores[sort_idx]

                suppress = torch.zeros(len(scores), dtype=torch.bool, device=scores.device)

                # Precompute nondirectional intervals (retain direction info in starts/ends)
                low = torch.minimum(starts, ends)
                high = torch.maximum(starts, ends)
                lengths = (high - low).clamp(min=0.0)

                for j in range(len(scores)):
                    if suppress[j]:
                        continue
                    for k in range(j + 1, len(scores)):
                        if suppress[k]:
                            continue

                        # --- IoU computation using nondirectional intervals ---
                        inter = torch.clamp(torch.minimum(high[j], high[k]) - torch.maximum(low[j], low[k]), min=0.0)
                        union = lengths[j] + lengths[k] - inter + 1e-10  # numerically stable
                        iou = inter / union

                        if iou > iou_thresh:
                            suppress[k] = True

                kept = ~suppress
                kept_indices = active_indices[kept]

        # Append the filtered tensors for this batch
        results["f_start"].append(f_start[i][kept_indices])
        results["f_end"].append(f_end[i][kept_indices])
        results["class"].append(class_id[i][kept_indices])
        results["confidence"].append(confidence[i][kept_indices])

    # Compress to batched tensors cropped/padded based on first batch's M
    if B == 0:
        return {
            "f_start": torch.empty((0, 0), dtype=f_start.dtype, device=device),
            "f_end": torch.empty((0, 0), dtype=f_end.dtype, device=device),
            "class": torch.empty((0, 0), dtype=class_id.dtype, device=device),
            "confidence": torch.empty((0, 0), dtype=confidence.dtype, device=device),
        }

    M = len(results["f_start"][0])

    batched = {
        "f_start": torch.zeros((B, M), dtype=f_start.dtype, device=device),
        "f_end": torch.zeros((B, M), dtype=f_end.dtype, device=device),
        "class": torch.full((B, M), -1, dtype=class_id.dtype, device=device),
        "confidence": torch.zeros((B, M), dtype=confidence.dtype, device=device),
    }

    for i in range(B):
        for key in batched:
            curr = results[key][i]
            len_curr = len(curr)
            crop_len = min(len_curr, M)
            batched[key][i, :crop_len] = curr[:crop_len]
            # Padded parts remain as initialized (0 or -1)

    # Debug print
    if DEBUG:
        print(f"[\033[36mDebug\033[0m] Filtering, f_start: {[t.tolist() for t in results['f_start']]}, "
              f"f_end: {[t.tolist() for t in results['f_end']]}, "
              f"class_id: {[t.tolist() for t in results['class']]}, "
              f"confidence: {[t.tolist() for t in results['confidence']]}")

    return batched


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
        print(f"[\033[33mWarn\033[0m] No valid frequency lines to plot after filtering.")
        return

    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        elif isinstance(x, list):
            x = np.array(x)
        return np.asarray(x)

    f_starts, f_stops, classes = map(to_numpy, (f_starts, f_stops, classes))

    valid_mask = (np.isfinite(f_starts) & np.isfinite(f_stops))
    if normalized:
        valid_mask &= (f_starts >= 0) & (f_starts <= 1) & (f_stops >= 0) & (f_stops <= 1)
    else:
        valid_mask &= (f_starts >= 0) & (f_starts < len(freqs)) & (f_stops >= 0) & (f_stops < len(freqs))

    classes, f_starts, f_stops = classes[valid_mask], f_starts[valid_mask], f_stops[valid_mask]
    classes = classes.astype(int)
    if len(classes) == 0:
        print(f"[\033[33mWarn\033[0m] No valid frequency lines in generated boxes.")
        return

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
