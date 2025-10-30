import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PyQt5.QtWidgets import QApplication
from torch.utils.data import DataLoader

from gen.SETIdataset import DynamicSpectrumDataset
from model.DetDWTNet import DWTNet
from model.UNet import UNet
from pipeline.patch_engine import SETIWaterFullDataset
from pipeline.pipeline_processor import SETIPipelineProcessor
from pipeline.renderer import SETIWaterfallRenderer
from utils.pred_core import pred

# Prediction modes
pmode = "detection"
# pmode = "mask"
# dmode = "none"
dmode = "edge"
# dmode = "argmax"

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
use_fil = True
background_fil = ['./data/33exoplanets/Kepler-438_M01_pol2_f1120.00-1150.00.fil',
                  './data/33exoplanets/HD-180617_M04_pol1_f1400.00-1410.00.fil']

# Polarization config
ignore_polarization = False
stokes_mode = "I"
XX_dir = "./data/XX/"
YY_dir = "./data/YY/"

# Observation data
# obs_file_path = "./data/BLIS692NS/BLIS692NS_data/spliced_blc00010203040506o7o0111213141516o7o0212223242526o7o031323334353637_guppi_58060_26569_HIP17147_0021.gpuspec.0002.fil"
# obs_file_path = "./data/BLIS692NS/BLIS692NS_data/spliced_blc00010203040506o7o0111213141516o7o0212223242526o7o031323334353637_guppi_58060_26569_HIP17147_0021.gpuspec.0000.fil"
# obs_file_path = "./data/BLIS692NS/BLIS692NS_data/spliced_blc00010203040506o7o0111213141516o7o0212223242526o7o031323334353637_guppi_58060_26569_HIP17147_0021.gpuspec.0000_chunk30720000_part0.fil"
obs_file_path = "./data/33exoplanets/Kepler-438_M01_pol2_f1120.00-1150.00.fil"
# obs_file_path = "./data/33exoplanets/HD-180617_M04_pol1_f1400.00-1410.00.fil"
# obs_file_path = './data/33exoplanets/'
obs_file_path = obs_file_path if ignore_polarization else [XX_dir, YY_dir]

# Prediction config
batch_size = 1  # Fixed to 1 for now
num_workers = 0
pred_dir = "./pred_results"
pred_steps = 100
dwtnet_ckpt = Path("./checkpoints/dwtnet") / "best_model.pth"
# dwtnet_ckpt = Path("./archived/weights/20250925_33e_det_realbk_none.pth")
unet_ckpt = Path("./checkpoints/unet") / "best_model.pth"
P = 2

# NMS config
iou_thresh = 0.99
score_thresh = 0.1
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
    use_spp=False,
    use_pan=False)
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

    # Create datasets based on mode and obs flag
    if obs and mode != "pipeline":
        if isinstance(obs_file_path, list):
            raise TypeError("In non-pipeline mode, observation data path should be a file, not a list.")
        else:
            if Path(obs_file_path).is_dir():
                raise ValueError("In non-pipeline mode, observation data path should be a file, not a directory.")
            # Use pipeline dataset for obs mode
            print("[\033[32mInfo\033[0m] Using observation data from:", obs_file_path)
            dataset = SETIWaterFullDataset(file_path=obs_file_path, patch_t=patch_t, patch_f=patch_f,
                                           overlap_pct=overlap_pct, ignore_polarization=ignore_polarization,
                                           stokes_mode=stokes_mode)
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
                 deocde_mode=dmode, save_npy=False, plot=True, iou_thresh=iou_thresh, top_k=top_k,
                 score_thresh=score_thresh, )
            print("[\033[32mInfo\033[0m] Running UNet inference...")
            pred(unet, data_mode='dbl', mode=pmode, data=batch, idx=idx, save_dir=pred_dir, device=device,
                 deocde_mode=dmode, save_npy=False, plot=True, iou_thresh=iou_thresh, top_k=top_k,
                 score_thresh=score_thresh, )


    elif mode == "pipeline":
        print("[\033[32mInfo\033[0m] Running pipeline processing mode")

        def match_polarization_files(files):
            from collections import defaultdict
            groups = defaultdict(list)
            unmatched = []
            for file_path in files:
                # Extract base by removing _pol* part
                stem = file_path.stem
                if '_pol' in stem:
                    base = stem.split('_pol')[0]
                    groups[base].append(str(file_path))
                else:
                    unmatched.append(str(file_path))
            matched_groups = [group for group in groups.values() if len(group) > 1]
            for base, group in groups.items():
                if len(group) == 1:
                    unmatched.extend(group)
            return matched_groups, unmatched

        all_files = []
        if ignore_polarization:
            # When True, handle polarization matching
            if isinstance(obs_file_path, str) and Path(obs_file_path).is_file():
                print(
                    f"[\033[31mError\033[0m] When ignoring polarization, observation data cannot be a single file: {obs_file_path}")
                sys.exit(1)  # Or raise error

            print("[\033[32mInfo\033[0m] Ignoring polarization: matching files for intensity stacking")
            if isinstance(obs_file_path, list) and len(obs_file_path) == 2:

                # [XX_dir, YY_dir]
                xx_dir = Path(obs_file_path[0])
                yy_dir = Path(obs_file_path[1])
                if not (xx_dir.is_dir() and yy_dir.is_dir()):
                    print(
                        f"[\033[31mError\033[0m] Both elements in observation data path must be directories when ignoring polarization: {obs_file_path}")
                    sys.exit(1)

                xx_files = sorted([f for f in xx_dir.iterdir() if f.suffix in [".fil", ".h5"]])
                yy_files = sorted([f for f in yy_dir.iterdir() if f.suffix in [".fil", ".h5"]])

                # Match by base name, assuming xx_files have _pol1, yy have _pol2
                all_files = xx_files + yy_files

            elif isinstance(obs_file_path, str) and Path(obs_file_path).is_dir():
                # Single directory, collect all files
                obs_path = Path(obs_file_path)
                all_files = sorted([f for f in obs_path.iterdir() if f.suffix in [".fil", ".h5"]])
            else:
                print(
                    f"[\033[31mError\033[0m] Invalid observation data path format: {obs_file_path}")
                sys.exit(1)

            if not all_files:
                print(f"[\033[31mError\033[0m] No .fil or .h5 files found in provided paths: {obs_file_path}")
                sys.exit(1)

            file_groups, unmatched = match_polarization_files(all_files)
            if unmatched:
                print(
                    f"[\033[33mWarning\033[0m] Unmatched files (not paired or no _pol* pattern): {', '.join(unmatched)}")
            # file_list is now list of lists (groups)
            file_list = file_groups  # Only process matched groups

        else:
            if isinstance(obs_file_path, list):
                print(
                    f"[\033[31mError\033[0m] When ignoring polarization, observation data path must be a file or directory, not a list: {obs_file_path}")
                sys.exit(1)
            obs_path = Path(obs_file_path)
            if obs_path.is_dir():
                file_list = sorted([f for f in obs_path.iterdir() if f.suffix in [".fil", ".h5"]])
                if not file_list:
                    print(f"[\033[31mError\033[0m] No .fil or .h5 files found in directory: {obs_path}")
                    sys.exit(1)
            else:
                file_list = [obs_path]

        for idx, f in enumerate(file_list):
            # f could be Path or list[str]
            if isinstance(f, list):
                print(f"[\033[32mInfo\033[0m] Processing polarization group: {', '.join([Path(p).name for p in f])}")
                file_path_for_dataset = f  # list[str]
                f_log_dir = Path(f[0]).stem.split('_pol')[0]  # Use base name for log dir
            else:
                print(f"[\033[32mInfo\033[0m] Processing file: {f}")
                file_path_for_dataset = str(f)
                f_log_dir = f.stem
            dataset = SETIWaterFullDataset(file_path=file_path_for_dataset, patch_t=patch_t, patch_f=patch_f,
                                           overlap_pct=overlap_pct, device=device,
                                           ignore_polarization=ignore_polarization, stokes_mode=stokes_mode)

            # Load model
            model = load_model(DWTNet, dwtnet_ckpt, **dwtnet_args)

            if ui:
                app = QApplication(sys.argv)
                renderer = SETIWaterfallRenderer(dataset, model, device, mode=pmode, log_dir=f_log_dir, drift=drift,
                                                 snr_threshold=snr_threshold, min_abs_drift=drift_min_abs,
                                                 verbose=verbose, nms_iou_thresh=iou_thresh,
                                                 nms_score_thresh=score_thresh, nms_top_k=top_k)
                renderer.setWindowTitle(f"SETI Waterfall Data Processor - {f_log_dir}")
                renderer.show()
                if idx == len(file_list) - 1:
                    sys.exit(app.exec_())
                else:
                    app.exec_()

            else:
                print("[\033[32mInfo\033[0m] Running in no-UI mode, logging only")
                processor = SETIPipelineProcessor(dataset, model, device, mode=pmode, log_dir=f_log_dir, drift=drift,
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
            pred(dwtnet, mode=pmode, data=pred_dataloader, save_dir=pred_dir, device=device, deocde_mode=dmode,
                 max_steps=pred_steps, save_npy=False, plot=True, iou_thresh=iou_thresh, top_k=top_k,
                 score_thresh=score_thresh)
        # --- 推理 UNet ---
        if execute1:
            print("[\033[32mInfo\033[0m] Running UNet inference...")
            unet = load_model(UNet, unet_ckpt, **unet_args)
            pred(unet, mode=pmode, data=pred_dataloader, save_dir=pred_dir, device=device, deocde_mode=dmode,
                 max_steps=pred_steps, save_npy=False, plot=True, iou_thresh=iou_thresh, top_k=top_k,
                 score_thresh=score_thresh)


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
