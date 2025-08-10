import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def pred_model(model, dataloader, save_dir, max_steps, device, save_npy=True, plot=False):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    plot_dir = Path(save_dir) / "plots" / f"{model.__class__.__name__}"
    if plot:
        plot_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            if idx >= max_steps:
                break

            if isinstance(batch, (list, tuple)):
                inputs = batch[0].to(device)  # Noisy input
                clean = batch[1].to(device) if len(batch) > 1 else None
                rfi_mask = batch[2].to(device) if len(batch) > 2 else None
            else:
                inputs = batch.to(device)
                clean = None
                rfi_mask = None

            outputs = model(inputs)

            # 假设模型输出：(denoised_spectrum, predicted_rfi_mask, detection_logits)
            if isinstance(outputs, (tuple, list)):
                if len(outputs) == 3:
                    denoised, pred_mask, _ = outputs
                elif len(outputs) == 2:
                    denoised, _ = outputs
                    pred_mask = None
                else:
                    raise ValueError(f"Unexpected outputs format. Got {outputs}")
            else:
                denoised = outputs
                pred_mask = None

            # 保存输出
            if save_npy:
                np.save(os.path.join(save_dir, f"pred_denoised_{idx:04d}.npy"), denoised.cpu().numpy())
                if pred_mask is not None:
                    np.save(os.path.join(save_dir, f"pred_mask_{idx:04d}.npy"), pred_mask.cpu().numpy())
                print(f"Saved denoised spectrum: {os.path.join(save_dir, f'pred_denoised_{idx:04d}.npy')}")

            # 绘图
            if plot:
                # 转 CPU numpy
                noisy_spec = inputs[0].cpu().squeeze().numpy()
                denoised_spec = denoised[0].cpu().squeeze().numpy()
                pred_mask_np = pred_mask[0].cpu().squeeze().numpy() if pred_mask is not None else None
                clean_spec = clean[0].cpu().squeeze().numpy() if clean is not None else None
                rfi_mask_np = rfi_mask[0].cpu().squeeze().numpy() if rfi_mask is not None else None

                # 假设 freq 为横轴
                freq_axis = np.arange(noisy_spec.shape[1])
                time_frames = noisy_spec.shape[0]

                fig, axs = plt.subplots(5, 1, figsize=(14, 35))

                def plot_spec(ax, data, title, cmap='viridis'):
                    im = ax.imshow(data, aspect='auto', origin='lower',
                                   extent=[freq_axis[0], freq_axis[-1], 0, time_frames],
                                   cmap=cmap)
                    ax.set_title(title)
                    ax.set_ylabel("Time Frame")
                    fig.colorbar(im, ax=ax, label="Intensity")

                plot_spec(axs[0], clean_spec if clean_spec is not None else np.zeros_like(noisy_spec), "Clean Spectrum")
                plot_spec(axs[1], noisy_spec, "Noisy Spectrum", cmap='viridis')
                plot_spec(axs[2], rfi_mask_np if rfi_mask_np is not None else np.zeros_like(noisy_spec),
                          "Ground Truth RFI Mask", cmap='Reds')
                plot_spec(axs[3], denoised_spec, "Denoised Spectrum", cmap='viridis')
                plot_spec(axs[4], pred_mask_np if pred_mask_np is not None else np.zeros_like(noisy_spec),
                          "Predicted RFI Mask", cmap='Reds')

                axs[-1].set_xlabel("Frequency Channel")

                plt.tight_layout()
                plot_path = plot_dir / f"pred_{idx:04d}.png"
                # plot_path = plot_dir / f"pred_{model.__class__.__name__}_{idx:04d}.png"
                plt.savefig(plot_path, dpi=480, bbox_inches='tight')
                plt.close()
                print(f"Saved plot: {plot_path}")


def process_batch(model, batch, idx, save_dir, device, save_npy=True, plot=False):
    """Process a single batch with the given model and save results"""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    # Prepare plot directory if needed
    plot_dir = Path(save_dir) / "plots" / f"{model.__class__.__name__}"
    if plot:
        plot_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        if isinstance(batch, (list, tuple)):
            inputs = batch[0].to(device)  # Noisy input
            clean = batch[1].to(device) if len(batch) > 1 else None
            rfi_mask = batch[2].to(device) if len(batch) > 2 else None
        else:
            inputs = batch.to(device)
            clean = None
            rfi_mask = None

        outputs = model(inputs)

        # Handle model outputs (denoised spectrum, predicted RFI mask)
        if isinstance(outputs, (tuple, list)):
            if len(outputs) == 3:
                denoised, pred_mask, _ = outputs
            elif len(outputs) == 2:
                denoised, _ = outputs
                pred_mask = None
            else:
                raise ValueError(f"Unexpected outputs format. Got {outputs}")

        # Save outputs as numpy files
        if save_npy:
            np.save(os.path.join(save_dir, f"pred_denoised_{idx:04d}.npy"), denoised.cpu().numpy())
            if pred_mask is not None:
                np.save(os.path.join(save_dir, f"pred_mask_{idx:04d}.npy"), pred_mask.cpu().numpy())
            print(f"Saved denoised spectrum: {os.path.join(save_dir, f'pred_denoised_{idx:04d}.npy')}")

        # Generate plots if requested
        if plot:
            # Convert tensors to numpy arrays
            noisy_spec = inputs[0].cpu().squeeze().numpy()
            denoised_spec = denoised[0].cpu().squeeze().numpy()
            pred_mask_np = pred_mask[0].cpu().squeeze().numpy() if pred_mask is not None else None
            clean_spec = clean[0].cpu().squeeze().numpy() if clean is not None else None
            rfi_mask_np = rfi_mask[0].cpu().squeeze().numpy() if rfi_mask is not None else None

            # Create frequency axis for plotting
            freq_axis = np.arange(noisy_spec.shape[1])
            time_frames = noisy_spec.shape[0]

            # Create figure with subplots
            fig, axs = plt.subplots(5, 1, figsize=(14, 35))

            def plot_spec(ax, data, title, cmap='viridis'):
                im = ax.imshow(data, aspect='auto', origin='lower',
                               extent=[freq_axis[0], freq_axis[-1], 0, time_frames],
                               cmap=cmap)
                ax.set_title(title)
                ax.set_ylabel("Time Frame")
                fig.colorbar(im, ax=ax, label="Intensity")

            plot_spec(axs[0], clean_spec if clean_spec is not None else np.zeros_like(noisy_spec), "Clean Spectrum")
            plot_spec(axs[1], noisy_spec, "Noisy Spectrum", cmap='viridis')
            plot_spec(axs[2], rfi_mask_np if rfi_mask_np is not None else np.zeros_like(noisy_spec),
                      "Ground Truth RFI Mask", cmap='Reds')
            plot_spec(axs[3], denoised_spec, "Denoised Spectrum", cmap='viridis')
            plot_spec(axs[4], pred_mask_np if pred_mask_np is not None else np.zeros_like(noisy_spec),
                      "Predicted RFI Mask", cmap='Reds')

            axs[-1].set_xlabel("Frequency Channel")

            plt.tight_layout()
            plot_path = plot_dir / f"pred_{idx:04d}.png"
            plt.savefig(plot_path, dpi=480, bbox_inches='tight')
            plt.close()
            print(f"Saved plot: {plot_path}")
