import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from gen.SETIdataset import DynamicSpectrumDataset
from model import DWTNet, UNet
from utils.pred_core import pred_model, process_batch


def main(mode=None):
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Set device
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"\n[\033[32mInfo\033[0m] Using device: {device}")

    # Create datasets
    pred_dataset = DynamicSpectrumDataset(tchans=144, fchans=1024, df=7.5, dt=10.0, fch1=None, ascending=False,
                                          drift_min=-1.0, drift_max=1.0,
                                          snr_min=10.0, snr_max=20.0,
                                          width_min=5, width_max=7.5,
                                          num_signals=(1, 1),
                                          noise_std_min=0.05, noise_std_max=0.1)

    # Create data loaders
    pred_dataloader = DataLoader(
        pred_dataset,
        batch_size=1,
        num_workers=1,
        pin_memory=True
    )

    def load_model(model_class, checkpoint_path, **kwargs):
        model = model_class(**kwargs).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model

    # Prediction configuration
    pred_dir = "./pred_results"
    pred_steps = 10

    if mode == "dbl":
        print("[\033[32mInfo\033[0m] Running dual-model comparison mode")

        # Load both models
        dwtnet_ckpt = Path("./checkpoints/dwtnet") / "best_model.pth"
        dwtnet = load_model(DWTNet, dwtnet_ckpt, in_chans=1, dim=64, levels=[2, 4, 8, 16], wavelet_name='db4')

        unet_ckpt = Path("./checkpoints/unet") / "best_model.pth"
        unet = load_model(UNet, unet_ckpt)

        # Process the same samples with both models
        for idx, batch in enumerate(pred_dataloader):
            if idx >= pred_steps:
                break

            print(f"[\033[32mInfo\033[0m] Processing sample {idx + 1}/{pred_steps}")

            # Process with DWTNet
            print("[\033[32mInfo\033[0m] Running DWTNet inference...")
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
            print("[\033[32mInfo\033[0m] Running UNet inference...")
            process_batch(
                unet,
                batch,
                idx,
                pred_dir,
                device,
                save_npy=False,
                plot=True
            )
    else:
        print("[\033[32mInfo\033[0m] Running single-model mode")

        # --- 推理 DWTNet ---
        dwtnet_ckpt = Path("./checkpoints/dwtnet") / "best_model.pth"
        dwtnet = load_model(DWTNet, dwtnet_ckpt, in_chans=1, dim=64, levels=[2, 4, 8, 16], wavelet_name='db4')
        pred_model(dwtnet, pred_dataloader, pred_dir + "_dwtnet", pred_steps, device, save_npy=False, plot=True)

        # --- 推理 UNet (可选) ---
        # unet_ckpt = Path("./checkpoints/unet") / "best_model.pth"
        # unet = load_model(UNet, unet_ckpt)
        # pred_model(unet, pred_dataloader, pred_dir + "_unet", pred_steps, device, save_npy=False, plot=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select the run mode")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["dbl", "pipeline"],  # restrict allowed values
        help="Run mode: no argument for single-model pred, 'dbl' for dual-model comparison, 'pipeline' for pipeline processing"
    )
    args = parser.parse_args()

    if args.mode is None:
        main()
    elif args.mode == "dbl":
        main("dbl")
    elif args.mode == "pipeline":
        main("pipeline")
