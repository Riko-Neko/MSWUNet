import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from SETIdataset import DynamicSpectrumDataset
from model.DWTNet import DWTNet
from model.UNet import UNet


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


# Main function
def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Set device (MPS for Apple Silicon, CUDA for NVIDIA, CPU as fallback)
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Create dataset with fixed parameters
    pred_dataset = DynamicSpectrumDataset(
        tchans=128,
        fchans=1024,
        df=7.5,
        dt=10.0,
        fch1=None,
        ascending=False,
        drift_min=-1.0,
        drift_max=1.0,
        snr_min=10.0,
        snr_max=20.0,
        width_min=5,
        width_max=7.5,
        num_signals=(1, 1),
        noise_std_min=0.05,
        noise_std_max=0.1
    )

    # Create data loader
    pred_dataloader = DataLoader(
        pred_dataset,
        batch_size=1,  # Process one sample at a time
        num_workers=1,
        pin_memory=True
    )

    def load_model(model_class, checkpoint_path, **kwargs):
        """Load trained model from checkpoint"""
        model = model_class(**kwargs).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model

    # Prediction configuration
    pred_steps = 10  # Number of samples to process
    pred_dir = "./pred_results"

    # Load both models
    dwtnet_ckpt = Path("./checkpoints/dwtnet") / "best_model.pth"
    dwtnet = load_model(DWTNet, dwtnet_ckpt, in_chans=1, dim=64, levels=[2, 4, 8, 16], wavelet_name='db4')

    unet_ckpt = Path("./checkpoints/unet") / "best_model.pth"
    unet = load_model(UNet, unet_ckpt)

    # Process the same samples with both models
    for idx, batch in enumerate(pred_dataloader):
        if idx >= pred_steps:
            break

        print(f"\nProcessing sample {idx + 1}/{pred_steps}")

        # Process with DWTNet
        print("Running DWTNet inference...")
        process_batch(
            dwtnet,
            batch,
            idx,
            pred_dir,
            device,
            save_npy=False,
            plot=True
        )

        # Process with UNet
        print("Running UNet inference...")
        process_batch(
            unet,
            batch,
            idx,
            pred_dir,
            device,
            save_npy=False,
            plot=True
        )


if __name__ == "__main__":
    main()
