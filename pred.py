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

# Prediction modes
pmode = "detection"
# mode = "mask"

# Data config
patch_t = 116
patch_f = 1024
overlap_pct = 0.02
tchans = 116
fchans = 1024
df = 7.450580597
dt = 10.200547328
fch1 = None
ascending = True
drift_min = -4.0
drift_max = 4.0
drift_min_abs = df // (tchans * dt)
snr_min = 15.0
snr_max = 25.0
width_min = 10
width_max = 30
num_signals = (1, 1)
noise_std_min = 0.025
noise_std_max = 0.05
noise_mean_min = 2
noise_mean_max = 3
nosie_type = "chi2"
use_fil = False
background_fil = ""

# Observation data
# obs_file_path = "./data/BLIS692NS/BLIS692NS_data/spliced_blc00010203040506o7o0111213141516o7o0212223242526o7o031323334353637_guppi_58060_26569_HIP17147_0021.gpuspec.0002.fil"
# obs_file_path = "./data/BLIS692NS/BLIS692NS_data/spliced_blc00010203040506o7o0111213141516o7o0212223242526o7o031323334353637_guppi_58060_26569_HIP17147_0021.gpuspec.0000.fil"
# obs_file_path = "./data/BLIS692NS/BLIS692NS_data/spliced_blc00010203040506o7o0111213141516o7o0212223242526o7o031323334353637_guppi_58060_26569_HIP17147_0021.gpuspec.0000_chunk30720000_part0.fil"
obs_file_path = "./data/33exoplanets/Kepler-438_M01_pol2_f1120.00-1150.00.fil"

# Prediction config
batch_size = 1  # Fixed to 1 for now
num_workers = 0
pred_dir = "./pred_results"
pred_steps = 10
dwtnet_ckpt = Path("./checkpoints/dwtnet") / "best_model.pth"
unet_ckpt = Path("./checkpoints/unet") / "best_model.pth"
P = 2

# NMS config
iou_thresh = 0.8
score_thresh = 0.99
top_k = None

# hits conf info
drift = [-4.0, 4.0]
snr_threshold = 20.0

# Model config
dwtnet_args = dict(
    in_chans=1,
    dim=64,
    levels=[2, 4, 8, 16],
    wavelet_name='db4',
    extension_mode='periodization',
    P=P,
    use_spp=True,
    use_pan=True)
unet_args = dict()


def main(mode=None, ui=False, obs=False, verbose=False, device=None, *args):
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Set device
    def check_device(dev):
        try:
            if dev.type == "cuda":
                return torch.cuda.is_available()
            elif dev.type == "mps":
                return torch.backends.mps.is_available() and torch.backends.mps.is_built()
            elif dev.type == "cpu":
                return True
            else:
                return False
        except Exception:
            return False

    if device is not None:
        try:
            device = torch.device(device)
            if not check_device(device):
                print(f"[\033[33mWarn\033[0m] Device '{device}' is not available. Fallback to default...")
                device = None
        except Exception as e:
            print(f"[\033[33mWarn\033[0m] Invalid device argument ({device}): {e}. Fallback to default...")
            device = None

    if device is None:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    print(f"\n[\033[32mInfo\033[0m] Using device: {device}")

    file_stem = Path(obs_file_path).stem

    # Create datasets based on mode and obs flag
    if obs and mode != "pipeline":
        # Use pipeline dataset for obs mode
        print("[\033[32mInfo\033[0m] Using observation data from:", obs_file_path)
        dataset = SETIWaterFullDataset(file_path=obs_file_path, patch_t=patch_t, patch_f=patch_f,
                                       overlap_pct=overlap_pct)
        pred_dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    else:

        pred_dataset = DynamicSpectrumDataset(mode=pmode, tchans=tchans, fchans=fchans, df=df, dt=dt, fch1=fch1,
                                              ascending=ascending, drift_min=drift_min, drift_max=drift_max,
                                              drift_min_abs=drift_min_abs, snr_min=snr_min, snr_max=snr_max,
                                              width_min=width_min, width_max=width_max, num_signals=num_signals,
                                              noise_std_min=noise_std_min, noise_std_max=noise_std_max,
                                              noise_mean_min=noise_mean_min, noise_mean_max=noise_mean_max,
                                              noise_type=nosie_type, use_fil=use_fil, background_fil=background_fil)
        pred_dataloader = DataLoader(pred_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    def load_model(model_class, checkpoint_path, **kwargs):
        model = model_class(**kwargs).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.eval()
        return model

    if mode == "dbl":
        global pred_dir
        pred_dir = Path(pred_dir) / "dbl"
        print("[\033[32mInfo\033[0m] Running dual-model comparison mode")
        # Load both models
        dwtnet = load_model(DWTNet, dwtnet_ckpt, **dwtnet_args)
        unet = load_model(UNet, unet_ckpt)
        # Process the same samples with both models
        for idx, batch in enumerate(pred_dataloader):
            if idx >= pred_steps:
                break
            print(f"[\033[32mInfo\033[0m] Processing sample {idx + 1}/{pred_steps}")
            print("[\033[32mInfo\033[0m] Running DWTNet inference...")
            pred(dwtnet, data_mode='dbl', mode=pmode, data=batch, idx=idx, save_dir=pred_dir, device=device,
                 save_npy=False, plot=True, iou_thresh=iou_thresh, top_k=top_k, score_thresh=score_thresh, )
            print("[\033[32mInfo\033[0m] Running UNet inference...")
            pred(unet, data_mode='dbl', mode=pmode, data=batch, idx=idx, save_dir=pred_dir, device=device,
                 save_npy=False, plot=True, iou_thresh=iou_thresh, top_k=top_k, score_thresh=score_thresh, )


    elif mode == "pipeline":
        print("[\033[32mInfo\033[0m] Running pipeline processing mode")
        obs_path = Path(obs_file_path)
        if obs_path.is_dir():
            file_list = sorted([f for f in obs_path.iterdir() if f.suffix in [".fil", ".h5"]])
            if not file_list:
                print(f"[\033[31mError\033[0m] No .fil or .h5 files found in directory: {obs_path}")
            for idx, f in enumerate(file_list):
                print(f"[\033[34mFile\033[0m] Processing file: {f}")
                dataset = SETIWaterFullDataset(file_path=str(f), patch_t=patch_t, patch_f=patch_f,
                                               overlap_pct=overlap_pct, device=device)

                # Load model
                model = load_model(DWTNet, dwtnet_ckpt, **dwtnet_args)
                if ui:
                    app = QApplication(sys.argv)
                    renderer = SETIWaterfallRenderer(dataset, model, device, log_dir=f.stem, drift=drift,
                                                     snr_threshold=snr_threshold, min_abs_drift=drift_min_abs,
                                                     verbose=verbose)
                    renderer.setWindowTitle(f"SETI Waterfall Data Processor - {f.name}")
                    renderer.show()
                    if idx == len(file_list) - 1:
                        sys.exit(app.exec_())
                    else:
                        app.exec_()
                else:
                    print("[\033[32mInfo\033[0m] Running in no-UI mode, logging only")
                    processor = SETIPipelineProcessor(dataset, model, device, log_dir=f.stem, drift=drift,
                                                      snr_threshold=snr_threshold, min_abs_drift=drift_min_abs,
                                                      verbose=verbose, nms_iou_thresh=iou_thresh,
                                                      nms_score_thresh=score_thresh, nms_top_k=top_k)
                    processor.process_all_patches()

        else:
            dataset = SETIWaterFullDataset(file_path=obs_file_path, patch_t=patch_t, patch_f=patch_f,
                                           overlap_pct=overlap_pct, device=device)
            model = load_model(DWTNet, dwtnet_ckpt, **dwtnet_args)

            if ui:
                app = QApplication(sys.argv)
                renderer = SETIWaterfallRenderer(dataset, model, device, log_dir=file_stem, drift=drift,
                                                 snr_threshold=snr_threshold, min_abs_drift=drift_min_abs,
                                                 verbose=verbose)
                renderer.setWindowTitle("SETI Waterfall Data Processor")
                renderer.show()
                sys.exit(app.exec_())
            else:
                print("[\033[32mInfo\033[0m] Running in no-UI mode, logging only")
                processor = SETIPipelineProcessor(dataset, model, device, log_dir=file_stem, drift=drift,
                                                  snr_threshold=snr_threshold, min_abs_drift=drift_min_abs,
                                                  verbose=verbose, nms_iou_thresh=iou_thresh,
                                                  nms_score_thresh=score_thresh, nms_top_k=top_k)
                processor.process_all_patches()

    else:
        print("[\033[32mInfo\033[0m] Running single-model mode")
        execute0, execute1 = args
        # --- 推理 DWTNet ---
        if execute0:
            print("[\033[32mInfo\033[0m] Running DWTNet inference...")
            dwtnet = load_model(DWTNet, dwtnet_ckpt, **dwtnet_args)
            pred(dwtnet, mode=pmode, data=pred_dataloader, save_dir=pred_dir, device=device, max_steps=pred_steps,
                 save_npy=False, plot=True, iou_thresh=iou_thresh, top_k=top_k, score_thresh=score_thresh)
        # --- 推理 UNet ---
        if execute1:
            print("[\033[32mInfo\033[0m] Running UNet inference...")
            unet = load_model(UNet, unet_ckpt, **unet_args)
            pred(unet, mode=pmode, data=pred_dataloader, save_dir=pred_dir, device=device, max_steps=pred_steps,
                 save_npy=False, plot=True, iou_thresh=iou_thresh, top_k=top_k, score_thresh=score_thresh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select the run mode")
    parser.add_argument("--mode",
                        type=str,
                        choices=["dbl", "pipeline"],
                        help="Run mode: no argument for single-model pred, 'dbl' for dual-model comparison, 'pipeline' for pipeline processing")
    parser.add_argument("--ui",
                        action="store_true",
                        default=False,
                        help="Run pipeline in UI mode")
    parser.add_argument("--obs",
                        action="store_true",
                        default=False,
                        help="Use observation data file for default and dbl modes")
    parser.add_argument("--verbose",
                        action="store_true",
                        default=False,
                        help="Use verbose output for pipeline mode")
    parser.add_argument("-d", "--device",
                        type=str,
                        default=None,
                        help="Device to use for inference (e.g. 'cuda:0', 'cpu', 'mps')")
    args = parser.parse_args()

    if args.mode is None:
        main(None, args.ui, args.obs, args.verbose, args.device, True, False)
    elif args.mode == "dbl":
        main("dbl", args.ui, args.obs, args.verbose, args.device)
    elif args.mode == "pipeline":
        main("pipeline", args.ui, args.obs, args.verbose, args.device)
