import numpy as np
import torch


def decode_F(raw: torch.Tensor, w_scale: float = 4.0) -> torch.Tensor:
    """
    raw: (B, P, 3, T, F)
    return: (B, N, 3) -> [f_start, f_stop, conf], normalized to [0,1]
    NOTE: This method does NOT remove time information from features; it flattens time axis
          only in output sequence. The head still saw full (T,F) context.
    """
    B, P, _, T, F = raw.shape
    device = raw.device
    dtype = raw.dtype

    # (B, T, F, P, 3)
    r = raw.permute(0, 3, 4, 1, 2).contiguous()

    # frequency cell indices, shape (1,1,F,1)
    f_idx = torch.arange(F, device=device, dtype=dtype).view(1, 1, F, 1)

    # extract channels
    f_off_logits = r[..., 0]  # (B, T, F, P)
    log_f_w = r[..., 1]
    conf_logits = r[..., 2]

    # decode
    f_off = torch.sigmoid(f_off_logits)  # inside cell offset [0,1]
    f_center_abs = (f_idx + f_off) / float(F)  # normalized center in [0,1]

    cell_freq = float(1.0 / float(F))
    # convert to same dtype device (not strictly necessary for math but keeps FP consistent)
    cell_freq = raw.new_tensor(cell_freq)

    f_half = 0.5 * torch.exp(log_f_w) * cell_freq * float(w_scale)

    f_start = (f_center_abs - f_half).clamp(0.0, 1.0)
    f_stop = (f_center_abs + f_half).clamp(0.0, 1.0)
    conf = torch.sigmoid(conf_logits)

    # flatten spatial + P dims -> N = T * F * P
    Ncells = T * F * P
    f_start = f_start.view(B, Ncells)  # (B, N)
    f_stop = f_stop.view(B, Ncells)
    conf = conf.view(B, Ncells)

    det_out = torch.stack([f_start, f_stop, conf], dim=2)  # (B, N, 3)
    return det_out


def nms_1d(det_out, iou_thresh=0.5, top_k=None):
    """
    Batched 1D NMS for frequency intervals.

    Args:
        det_out: (B, N, 3) -> [f_start, f_stop, conf]
        iou_thresh: float, IoU threshold for suppression
        top_k: int or None, max number of intervals to keep per batch
    Returns:
        List of tensors, each shape (M, 3) for one batch
    """
    B, N, _ = det_out.shape
    device = det_out.device
    results = []

    for b in range(B):
        boxes = det_out[b]  # (N, 3)
        starts, stops, scores = boxes[:, 0], boxes[:, 1], boxes[:, 2]
        order = torch.argsort(scores, descending=True)
        starts, stops, scores = starts[order], stops[order], scores[order]
        keep = []
        suppressed = torch.zeros(N, dtype=torch.bool, device=device)

        for i in range(N):
            if suppressed[i]:
                continue

            keep.append([starts[i].item(), stops[i].item(), scores[i].item()])
            inter_start = torch.maximum(starts[i], starts[i + 1:])
            inter_stop = torch.minimum(stops[i], stops[i + 1:])
            inter = torch.clamp(inter_stop - inter_start, min=0.0)
            union = (stops[i] - starts[i]) + (stops[i + 1:] - starts[i + 1:]) - inter
            ious = inter / union.clamp(min=1e-9)
            suppressed[i + 1:] |= (ious > iou_thresh)
        keep = torch.tensor(keep, device=device)

        if top_k is not None and keep.size(0) > top_k:
            keep = keep[:top_k]
        results.append(keep)

    return results


def plot_F_lines(ax, freqs, pred_boxes, normalized=True, color='red', linestyle='--', linewidth=0.5):
    """
    Plot detected frequency intervals as vertical lines on a 1D frequency plot.

    Args:
        ax (matplotlib.axes.Axes): The axis to plot on.
        freqs (array-like): Array of frequency values.
        pred_boxes (tuple): (N, f_starts, f_stops)
            - N: number of intervals
            - f_starts: list/array of start indices (or normalized [0,1] values)
            - f_stops: list/array of stop indices (or normalized [0,1] values)
        normalized (bool): Whether the boxes are normalized to [0, 1].
        color (str): Color of the lines.
        linestyle (str): Line style.
        linewidth (float): Line width.
    """
    N, f_starts, f_stops = pred_boxes
    if normalized:
        f_starts = np.rint(f_starts * (len(freqs) - 1)).astype(int)
        f_stops = np.rint(f_stops * (len(freqs) - 1)).astype(int)
    for f_start, f_stop in zip(f_starts, f_stops):
        ax.axvline(freqs[f_start], color=color, linestyle=linestyle, linewidth=linewidth)
        ax.axvline(freqs[f_stop], color=color, linestyle=linestyle, linewidth=linewidth)
