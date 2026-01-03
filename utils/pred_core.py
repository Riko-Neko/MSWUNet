import os
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.det_utils import decode_F, plot_F_lines, extract_F_slice
from utils.metrics_utils import SNR_filter

DEBUG = True


def _process_batch_core(model, batch, device, mode, *args):
    """
    Core processing function: Handles input batch and returns model outputs

    Args:
        model: The neural network model to use for prediction
        batch: Input data batch (can be tuple/list or single tensor)
        device: Device to run computation on (e.g., 'cuda' or 'cpu')
        mode: 'mask' or 'detection'
        args: Waterfall data index information (if available)

    Returns:
        Dictionary containing:
        - inputs: Original noisy input
        - denoised: Model's denoised output
        - pred_mask: Predicted RFI mask (if available, for mask mode)
        - clean: Ground truth clean spectrum (if available)
        - rfi_mask: Ground truth RFI mask (if available, for mask mode)
        - gt_boxes: Ground truth boxes (if available, for detection mode)
        - raw_preds: Raw predictions (if available, for detection mode)
    """
    # Handle different input formats (single tensor or tuple/list)
    clean = None
    rfi_mask = None
    gt_boxes = None
    f_min, f_max, start_t, end_t = None, None, None, None
    ascending = True
    if isinstance(batch, (list, tuple)):
        inputs = batch[0].to(device)

        if mode == 'detection':
            if len(batch) > 1 and isinstance(batch[1], torch.Tensor):
                clean = batch[1].to(device)
            elif isinstance(batch[1], list):
                f_min, f_max, start_t, end_t, ascending = args
                if DEBUG:
                    f1, f2 = (f_min, f_max) if ascending else (f_max, f_min)
                    print(
                        f"[\033[36mDebug\033[0m] Extracting from {f1} MHz to {f2} MHz.")
            if len(batch) > 2 and isinstance(batch[2], torch.Tensor):
                gt_boxes = batch[2].to(device)

        else:  # 'mask' as default
            if len(batch) > 1 and isinstance(batch[1], torch.Tensor):
                clean = batch[1].to(device)
            if len(batch) > 2 and isinstance(batch[2], torch.Tensor):
                rfi_mask = batch[2].to(device)
    else:
        inputs = batch.to(device)

    # Get model predictions
    outputs = model(inputs)

    # Parse different output formats from model
    logits = None
    pred_mask = None
    raw_preds = None
    if isinstance(outputs, (tuple, list)):
        if len(outputs) == 3:
            denoised, pred_mask, logits = outputs  # (denoised, mask, logits) for mask
        elif len(outputs) == 2:
            if mode == 'detection':
                denoised, raw_preds = outputs  # (denoised, raw_preds)
            else:  # 'mask' as default
                denoised, pred_mask = outputs  # (denoised, mask)
        else:
            denoised = outputs[0]  # Assume first is denoised
    else:
        denoised = outputs  # Single output

    return {
        "inputs": inputs,
        "clean": clean,
        "denoised": denoised,
        "rfi_mask": rfi_mask,
        "pred_mask": pred_mask,
        "logits": logits,
        "gt_boxes": gt_boxes,
        "raw_preds": raw_preds,
        "f_info": (f_min, f_max),
        "t_info": (start_t, end_t),
        "ascending": ascending
    }


def _save_batch_results(results, idx, save_dir, model_class_name, mode='detection', save_npy=False, plot=True,
                        **kwarg_groups):
    """
    Save and visualize results for a single batch

    Args:
        results: Dictionary from _process_batch_core
        idx: Batch index (used for filenames)
        save_dir: Directory to save outputs
        model_class_name: Name of model class (for plot subfolder)
        mode: 'mask' or 'detection'
        save_npy: Whether to save numpy arrays
        plot: Whether to generate visualization plots
        **kwarg_groups: Additional arguments for NMS (group1), SNR calculation (group2) and dedrift peak (group3) (if applicable)
    """
    # Create plot directory if needed
    plot_dir = Path(save_dir) / "plots" / model_class_name
    if plot:
        plot_dir.mkdir(parents=True, exist_ok=True)

    # Save numpy arrays
    if save_npy:
        np.save(os.path.join(save_dir, "npy", f"pred_denoised_{idx:04d}.npy"),
                results["denoised"].cpu().numpy())
        if mode == 'mask' and results["pred_mask"] is not None:
            np.save(os.path.join(save_dir, "npy", f"pred_mask_{idx:04d}.npy"),
                    results["pred_mask"].cpu().numpy())
        elif mode == 'detection' and results["raw_preds"] is not None:
            np.save(os.path.join(save_dir, "npy", f"raw_preds_{idx:04d}.npy"),
                    results["raw_preds"].cpu().numpy())
        print(f"Saved denoised spectrum: {os.path.join(save_dir, "npy", f'pred_denoised_{idx:04d}.npy')}")

    # Generate visualization plots
    if plot:
        # Extract first sample from batch
        noisy_spec = results["inputs"][0].cpu().squeeze().numpy()
        denoised_spec = results["denoised"][0].cpu().squeeze().numpy()
        clean_spec = (results["clean"][0].cpu().squeeze().numpy()
                      if results["clean"] is not None else None)
        kwarg_groups["group_fsnr"].pop("fsnr_threshold", None)
        pad_fraction = kwarg_groups["group_fsnr"].pop("pad_fraction", None)
        SNR_est = SNR_filter(results["denoised"][0], **kwarg_groups["group_fsnr"])
        if DEBUG:
            print(f"[\033[36mDebug\033[0m] Estimated SNR={SNR_est:.2f}")
        freq_frames = noisy_spec.shape[1]
        time_frames = denoised_spec.shape[0]
        f_min, f_max = results["f_info"]
        t_start, t_end = results["t_info"]
        dt = kwarg_groups["group_dedrift"]["dt_s"]
        ascending = results["ascending"]
        if f_min is not None:
            freq_axis = np.arange(f_min, f_max, (f_max - f_min) / freq_frames) if ascending else np.arange(
                f_max, f_min, (f_min - f_max) / freq_frames)
        else:
            freq_axis = np.arange(freq_frames)
        time_axis = dt * np.arange(t_start, t_end) if t_start is not None else np.arange(time_frames)
        time_duration = time_frames * dt
        freq_duration = freq_frames * kwarg_groups["group_dedrift"]["df_hz"]

        if mode == 'mask':
            pred_mask_np = (results["pred_mask"][0].cpu().squeeze().numpy()
                            if results["pred_mask"] is not None else None)
            rfi_mask_np = (results["rfi_mask"][0].cpu().squeeze().numpy()
                           if results["rfi_mask"] is not None else None)
            logits_value = None
            if results["logits"] is not None:
                logits_tensor = results["logits"][0].cpu()
                probability = torch.sigmoid(logits_tensor).item()
                logits_value = probability > 0.5

            # Create figure with subplots
            figlen = 9 if freq_frames <= 512 else 14
            fig, axs = plt.subplots(5, 1, figsize=(figlen, figlen * 2.3))

            def plot_spec(ax, data, title, cmap='viridis'):
                """Helper function to plot spectrum data"""
                im = ax.imshow(data, aspect='auto', origin='lower',
                               extent=[freq_axis[0], freq_axis[-1], 0, time_frames], cmap=cmap)
                ax.set_title(title)
                ax.set_ylabel("Time Frame")
                fig.colorbar(im, ax=ax, label="Intensity")

            # Plot all components
            plot_spec(axs[0], clean_spec if clean_spec is not None else np.zeros_like(noisy_spec),
                      "Clean Spectrum")
            if logits_value is not None:
                axs[0].text(0.95, 0.05, f"Flag: {logits_value}", transform=axs[0].transAxes, fontsize=12,
                            horizontalalignment='right', verticalalignment='bottom',
                            bbox=dict(facecolor='white', alpha=0.8))
            plot_spec(axs[1], noisy_spec, "Noisy Spectrum", cmap='viridis')
            plot_spec(axs[2], rfi_mask_np if rfi_mask_np is not None else np.zeros_like(noisy_spec),
                      "Ground Truth RFI Mask", cmap='Reds')
            plot_spec(axs[3], denoised_spec, "Denoised Spectrum", cmap='viridis')
            axs[3].text(0.98, 0.95, f"SNR={SNR_est:.2f}", transform=axs[3].transAxes, ha='right', va='top', fontsize=12,
                        bbox=dict(facecolor='white', alpha=0.7))
            plot_spec(axs[4], pred_mask_np if pred_mask_np is not None else np.zeros_like(noisy_spec),
                      "Predicted RFI Mask", cmap='Reds')

            axs[-1].set_xlabel("Frequency Channel")
            plt.tight_layout()

        elif mode == 'detection':
            def _get_snrs(event, fraction, starts, stops):
                """Helper function to get SNRs for boxes"""
                to_slice = None
                snr_vals = []
                drift_rates = []
                for i, (f_start, f_stop) in enumerate(zip(starts, stops)):
                    if starts[i] != starts[i]:  # nan != nan
                        continue
                    snr_val = 0.0
                    freq_change_hz = (f_stop - f_start) * freq_duration
                    relative_drift_rate = float(freq_change_hz / time_duration) if time_duration > 0 else 0.0
                    drift_rate = -relative_drift_rate if not ascending else relative_drift_rate
                    if event.ndim == 3:
                        to_slice = event[0]
                    if to_slice.ndim != 2:
                        raise ValueError(
                            f"[\033[31mError\033[0m] input_patch must be 2D after squeeze, got {tuple(to_slice.shape)}")
                    if to_slice is not None:
                        try:
                            roi, f_start_pad, f_stop_pad, _, _ = extract_F_slice(event, f_start, f_stop,
                                                                                 pad_fraction=fraction)
                            snr_val = SNR_filter(roi, mode="dedrift_peak", drift_hz_per_s=relative_drift_rate,
                                                 **kwarg_groups["group_dedrift"])
                        except Exception as e:
                            print(f"[\033[31mError\033[0m] Failed to estimate SNR for detection: {e}")
                            snr_val = 0.0
                    snr_vals.append(snr_val)
                    if DEBUG:
                        drift_rates.append(drift_rate)
                if DEBUG:
                    print(f"[\033[36mDebug\033[0m] SNR: {snr_vals}, drift_rate: {drift_rates}")
                return snr_vals, drift_rates

            # Process predictions
            pred_boxes_tuple = None
            snr_vals, drift_rates = None, None
            events_patch = results["inputs"][0]
            if results["raw_preds"] is not None:
                det_outs = decode_F(results["raw_preds"], **kwarg_groups["group_nms"])  # dict
                f_starts = det_outs["f_start"][0].cpu().numpy()
                f_stops = det_outs["f_end"][0].cpu().numpy()
                N_pred = det_outs["confidence"].shape[1]
                classes = det_outs["class"][0].cpu().numpy()
                pred_boxes_tuple = (N_pred, classes, f_starts, f_stops)
                if results["gt_boxes"] is None:
                    snr_vals, drift_rates = _get_snrs(events_patch, pad_fraction, f_starts, f_stops)

            # Process ground truth boxes
            gt_boxes_tuple = None
            if results["gt_boxes"] is not None:
                gt_boxes = results["gt_boxes"][0].cpu().numpy()  # Assuming (N_gt, 2) or similar, [start, stop]
                gt_starts = gt_boxes[:, 0]
                gt_stops = gt_boxes[:, 1]
                gt_classes = gt_boxes[:, 2]
                N_gt = len(gt_starts)
                gt_boxes_tuple = (N_gt, gt_classes, gt_starts, gt_stops)
                snr_vals, drift_rates = _get_snrs(events_patch, pad_fraction, gt_starts, gt_stops)

            # Create figure with subplots
            figlen = 9 if freq_frames <= 512 else 14
            fig, axs = plt.subplots(3, 1, figsize=(figlen, 9))

            def plot_spec(ax, data, title, cmap='viridis', boxes=None, normalized=False, snrs=None,
                          color=['red', 'green'], linestyle='--', linewidth=2):
                """Helper function to plot spectrum data with optional boxes"""
                im = ax.imshow(data, aspect='auto', origin='lower',
                               extent=[freq_axis[0], freq_axis[-1], time_axis[0], time_axis[-1]], cmap=cmap)
                ax.set_title(title)
                ax.set_ylabel("Time Frame")
                fig.colorbar(im, ax=ax, label="Intensity")
                if boxes is not None:
                    plot_F_lines(ax, freq_axis, boxes, normalized=normalized, snrs=snrs, color=color,
                                 linestyle=linestyle, linewidth=linewidth)

            # Plot all components
            plot_spec(axs[0], clean_spec if clean_spec is not None else np.zeros_like(noisy_spec),
                      "Clean Spectrum")
            plot_spec(axs[1], noisy_spec, "Noisy Spectrum", cmap='viridis',
                      boxes=gt_boxes_tuple if gt_boxes_tuple is not None else pred_boxes_tuple,
                      normalized=True, snrs=snr_vals)
            plot_spec(axs[2], denoised_spec, "Denoised Spectrum", cmap='viridis', boxes=pred_boxes_tuple,
                      normalized=True)
            axs[2].text(0.98, 0.95, f"SNR={SNR_est:.2f}", transform=axs[2].transAxes, ha='right', va='top', fontsize=12,
                        bbox=dict(facecolor='white', alpha=0.7))

            axs[-1].set_xlabel("Frequency Channel")
            plt.tight_layout()

        else:
            raise ValueError(f"Unsupported mode: {mode}")

        # Save figure
        plot_path = plot_dir / f"pred_{idx:04d}.png"
        plt.savefig(plot_path, dpi=480, bbox_inches='tight')
        plt.close()
        print(f"[\033[32mInfo\033[0m] Saved plot: {plot_path}")


def pred(model: torch.nn.Module,
         data: Union[DataLoader, torch.Tensor],
         save_dir: Union[str, Path],
         device: torch.device,
         mode: str = 'detection',
         data_mode: str = "dataloader",
         max_steps: Optional[int] = None,
         idx: Optional[int] = None,
         save_npy: bool = True,
         plot: bool = False,
         **kwarg_groups):
    """
    Unified prediction function that handles both batch and dataloader processing, and pipeline mode

    Args:
        model: Model to use for prediction
        data: Either a DataLoader (for dataloader mode) or a batch tensor (for batch mode)
        save_dir: Directory to save outputs
        device: Computation device
        mode: 'mask' or 'detection'
        data_mode: Processing mode:
              "dbl" for batch mode (process single batch),
              otherwise dataloader mode (process entire dataloader)
        max_steps: For dataloader mode, maximum number of batches to process
        idx: For batch mode, batch index (for filenames)
        save_npy: Whether to save numpy outputs
        plot: Whether to generate plots
        **kwarg_groups: Additional arguments for NMS (group1), SNR calculation (group2) and dedrift peak (group3) (if applicable)
    """
    assert mode in ('mask', 'detection'), f"Unsupported mode: {mode}"

    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        if data_mode == "dbl":  # Batch processing mode
            if idx is None:
                raise ValueError("In batch mode, 'idx' must be provided")

            # Process single batch
            results = _process_batch_core(model, data, device, mode)
            _save_batch_results(results, idx, save_dir, model.__class__.__name__, mode, save_npy, plot, **kwarg_groups)

        else:  # Dataloader processing mode
            if max_steps is None:
                raise ValueError("In dataloader mode, 'max_steps' must be provided")

            # Process entire dataloader
            for batch_idx, batch in enumerate(data):
                f_min, f_max, start_t, end_t, ascending = None, None, None, None, False
                if batch_idx >= max_steps:
                    break
                if isinstance(batch[1], list):
                    freqs = data.dataset.freqs
                    patch_t = data.dataset.patch_t
                    patch_f = data.dataset.patch_f
                    start_t = batch[1][0]
                    start_f = batch[1][1]
                    end_t = start_t + patch_t
                    end_f = start_f + patch_f
                    ascending = data.dataset.ascending
                    if freqs[0] < freqs[-1]:
                        f_min, f_max = freqs[start_f], freqs[end_f - 1]
                    else:
                        f_min, f_max = freqs[end_f - 1], freqs[start_f]

                results = _process_batch_core(model, batch, device, mode, f_min, f_max, start_t, end_t, ascending)
                _save_batch_results(results, batch_idx, save_dir, model.__class__.__name__, mode, save_npy, plot,
                                    **kwarg_groups)
