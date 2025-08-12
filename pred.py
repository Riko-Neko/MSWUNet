import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PyQt5.QtWidgets import QApplication
from torch.utils.data import DataLoader

from gen.SETIdataset import DynamicSpectrumDataset
from model.DWTNet import DWTNet
from model.UNet import UNet
from pipeline.patch_engine import SETIWaterFullDataset
from pipeline.pipeline_processor import SETIPipelineProcessor
from pipeline.renderer import SETIWaterfallRenderer
from utils.pred_core import pred


def main(mode=None, ui=False, *args):
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
    tchans = 144
    fchans = 1024
    df = 7.5
    dt = 1.0
    drift_min = -4.0
    drift_max = 4.0
    drift_min_abs = df // (tchans * dt)
    pred_dataset = DynamicSpectrumDataset(tchans=tchans, fchans=fchans, df=df, dt=dt, fch1=None, ascending=False,
                                          drift_min=drift_min, drift_max=drift_max, drift_min_abs=0.2,
                                          snr_min=10.0, snr_max=20.0, width_min=10, width_max=15, num_signals=(1, 1),
                                          noise_std_min=0.025, noise_std_max=0.05)

    # Create data loaders
    pred_dataloader = DataLoader(pred_dataset, batch_size=1, num_workers=1, pin_memory=True)

    def load_model(model_class, checkpoint_path, **kwargs):
        model = model_class(**kwargs).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.eval()
        return model

    # Prediction configuration
    pred_dir = "./pred_results"
    pred_steps = 10
    dwtnet_ckpt = Path("./checkpoints/dwtnet") / "best_model.pth"
    unet_ckpt = Path("./checkpoints/unet") / "best_model.pth"

    if mode == "dbl":
        pred_dir = Path(pred_dir) / "dbl"
        print("[\033[32mInfo\033[0m] Running dual-model comparison mode")
        # Load both models
        dwtnet = load_model(DWTNet, dwtnet_ckpt, in_chans=1, dim=64, levels=[2, 4, 8, 16], wavelet_name='db4')
        unet = load_model(UNet, unet_ckpt)
        # Process the same samples with both models
        for idx, batch in enumerate(pred_dataloader):
            if idx >= pred_steps:
                break
            print(f"[\033[32mInfo\033[0m] Processing sample {idx + 1}/{pred_steps}")
            print("[\033[32mInfo\033[0m] Running DWTNet inference...")
            pred(dwtnet, mode='dbl', data=batch, idx=idx, save_dir=pred_dir, device=device, save_npy=False, plot=True)
            print("[\033[32mInfo\033[0m] Running UNet inference...")
            pred(unet, mode='dbl', data=batch, idx=idx, save_dir=pred_dir, device=device, save_npy=False, plot=True)

    elif mode == "pipeline":
        print("[\033[32mInfo\033[0m] Running pipeline processing mode")
        file_path = "./data/BLIS692NS/BLIS692NS_data/spliced_blc00010203040506o7o0111213141516o7o0212223242526o7o031323334353637_guppi_58060_26569_HIP17147_0021.gpuspec.0002.fil"
        # Create dataset
        dataset = SETIWaterFullDataset(
            file_path=file_path,
            patch_t=144,
            patch_f=1024,
            overlap_pct=0.02
        )
        # Load model
        model = load_model(DWTNet, dwtnet_ckpt, in_chans=1, dim=64, levels=[2, 4, 8, 16], wavelet_name='db4')
        if ui:
            # Create and show the renderer
            app = QApplication(sys.argv)
            renderer = SETIWaterfallRenderer(dataset, model, device)
            renderer.setWindowTitle("SETI Waterfall Data Processor")
            renderer.show()
            sys.exit(app.exec_())
        else:
            print("[\033[32mInfo\033[0m] Running in no-UI mode, logging only")
            processor = SETIPipelineProcessor(dataset, model, device)
            processor.process_all_patches()


    else:
        print("[\033[32mInfo\033[0m] Running single-model mode")
        execute0, execute1 = args
        # --- 推理 DWTNet ---
        if execute0:
            print("[\033[32mInfo\033[0m] Running DWTNet inference...")
            dwtnet = load_model(DWTNet, dwtnet_ckpt, in_chans=1, dim=64, levels=[2, 4, 8, 16], wavelet_name='db4')
            pred(dwtnet, data=pred_dataloader, save_dir=pred_dir, device=device, max_steps=pred_steps, save_npy=False,
                 plot=True)
        # --- 推理 UNet ---
        if execute1:
            print("[\033[32mInfo\033[0m] Running UNet inference...")
            unet = load_model(UNet, unet_ckpt)
            pred(unet, data=pred_dataloader, save_dir=pred_dir, device=device, max_steps=pred_steps, save_npy=False,
                 plot=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select the run mode")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["dbl", "pipeline"],  # restrict allowed values
        help="Run mode: no argument for single-model pred, 'dbl' for dual-model comparison, 'pipeline' for pipeline processing"
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        default=False,
        help="Run pipeline in UI mode"
    )
    args = parser.parse_args()

    if args.mode is None:
        main(None, args.ui, True, False)
    elif args.mode == "dbl":
        main("dbl", args.ui)
    elif args.mode == "pipeline":
        main("pipeline", args.ui)
