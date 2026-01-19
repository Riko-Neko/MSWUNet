import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PyQt5.QtWidgets import QApplication
from torch.utils.data import DataLoader

from gen.SETIdataset import DynamicSpectrumDataset
from model.DetMSWNet import MSWNet
from model.UNet import UNet
from pipeline.patch_engine import SETIWaterFullDataset
from pipeline.pipeline_processor import SETIPipelineProcessor
from pipeline.renderer import SETIWaterfallRenderer
from utils.pred_core import pred

# Prediction modes
pmode = "detection"
# pmode = "mask"

# Data config
patch_t = 116
patch_f = 256
overlap_pct = 0.02
tchans = 116
fchans = 256
df = 7.450580597
dt = 10.200547328
fch1 = None
ascending = True
drift_min = -4.0
drift_max = 4.0
drift_min_abs = df // (tchans * dt)
snr_min = 15
snr_max = 35
width_min = 10
width_max = 30
num_signals = (1, 1)
noise_std_min = 0.025
noise_std_max = 0.05
noise_mean_min = 2
noise_mean_max = 3
nosie_type = "chi2"
rfi_enhance = False
use_fil = True
fil_folder = Path('./data/33exoplanets')
background_fil = list(fil_folder.rglob("*.fil"))

# Polarization config
ignore_polarization = True
stokes_mode = "I"
# XX_dir = "/data/Raid0/obs_data/33exoplanets/xx/"
# YY_dir = "/data/Raid0/obs_data/33exoplanets/yy/"
XX_dir = "./data/33exoplanets/xx/"
YY_dir = "./data/33exoplanets/yy/"
Beam = [1, 14, 3, 7, 15]
# Beam = [8, 16, 4, 9, 17]
# Beam = [10, 18, 5, 11, 19]
# Beam = [12, 2, 6, 13]
# Beam = [1, 10]
# Beam = [4]
# Beam = None

# Observation data
# obs_file_path = "./data/BLIS692NS/BLIS692NS_data/spliced_blc00010203040506o7o0111213141516o7o0212223242526o7o031323334353637_guppi_58060_26569_HIP17147_0021.gpuspec.0002.fil"
# obs_file_path = "./data/BLIS692NS/BLIS692NS_data/spliced_blc00010203040506o7o0111213141516o7o0212223242526o7o031323334353637_guppi_58060_26569_HIP17147_0021.gpuspec.0000.fil"
# obs_file_path = "./data/BLIS692NS/BLIS692NS_data/spliced_blc00010203040506o7o0111213141516o7o0212223242526o7o031323334353637_guppi_58060_26569_HIP17147_0021.gpuspec.0000_chunk30720000_part0.fil"
obs_file_path = "data/33exoplanets/xx/tmp/Kepler-438_M01_pol1_f1140.50-1140.70.fil"
# obs_file_path = "data/33exoplanets/yy/Kepler-438_M01_pol2_f1140.50-1140.70.fil"
# obs_file_path = "./data/33exoplanets/xx/HD-180617_M04_pol1_f1404.00-1404.10.fil"
# obs_file_path = "./data/33exoplanets/yy/HD-180617_M04_pol2_f1404.00-1404.10.fil"
# obs_file_path = './data/33exoplanets/'
obs_file_path = [XX_dir, YY_dir] if ignore_polarization else obs_file_path

# Prediction config
RAW = False
batch_size = 1  # ⚠️Fixed to 1 for now, cannot use batch_size > 1, which will break the data.
num_workers = 0
pred_dir = "./pred_results"
pred_steps = 9999999
# dwtnet_ckpt = Path("./checkpoints/mswunet/bin1024") / "best_model.pth"
dwtnet_ckpt = Path("./checkpoints/mswunet/bin256") / "final.pth"
# dwtnet_ckpt = Path("./checkpoints/mswunet/bin256") / "case_model.pth"
unet_ckpt = Path("./checkpoints/unet") / "best_model.pth"
P = 2

# NMS config
nms_kargs = dict(
    iou_thresh=0.,
    score_thresh=0.5)
if pmode == 'yolo':
    nms_kargs['top_k'] = None

# hits config
drift = [-4.0, 4.0]  # work only for mask mode
snr_threshold = 5.0
pad_fraction = 0.5
fsnr_args = dict(
    fsnr_threshold=300,
    top_fraction=0.001,
    min_pixels=50)
dedrift_args = dict(
    df_hz=df,
    dt_s=dt,
    guard_bins=3
)

# Model config
dim = 64
levels = [2, 4, 8, 16]
feat_channels = 64
dwtnet_args = dict(
    in_chans=1,
    dim=dim,
    levels=levels,
    wavelet_name='db4',
    extension_mode='periodization')
detector_args = dict(
    fchans=fchans,
    N=5,
    num_classes=2,
    feat_channels=feat_channels,
    dropout=0.005)

unet_args = dict()


def main(mode=None, ui=False, obs=False, verbose=False, device=None, *args):
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    global batch_size
    if batch_size != 1:
        print(f"[\033[31mSevere Warn\033[0m] !!!! Batch size is fixed to 1 for now, cannot use batch_size > 1 !!!!")
        batch_size = 1

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

    def match_polarization_files(files, M_list=None):
        """
        Match polarization files only for selected beams M_list.
        Example M_list = [1,2,3] -> match M01, M02, M03

        Args:
            files: list of Path objects
            M_list: list of integers for beam selection

        Returns:
            matched_groups (sorted), unmatched (sorted)
        """
        from collections import defaultdict

        groups = defaultdict(list)
        unmatched = []

        if M_list is not None:
            # Convert [8,10] -> ["M08","M10"]
            allowed_M = [f"M{m:02d}" for m in M_list]
            allowed_M_set = set(allowed_M)
            # Priority map: lower index = higher priority
            M_priority = {m: i for i, m in enumerate(allowed_M)}
            print(f"[\033[32mInfo\033[0m] Selected beams (ordered): {allowed_M}")
        else:
            allowed_M = None
            allowed_M_set = None
            M_priority = None
            print("[\033[32mInfo\033[0m] No beam filtering applied.")

        for file_path in files:
            stem = file_path.stem

            if "_pol" not in stem:
                unmatched.append(str(file_path))
                continue

            try:
                parts = stem.split("_")
                beam_name = next(p for p in parts if p.startswith("M") and len(p) == 3)
            except StopIteration:
                unmatched.append(str(file_path))
                continue

            if allowed_M_set is not None and beam_name not in allowed_M_set:
                continue

            base = stem.split("_pol")[0]
            groups[(beam_name, base)].append(str(file_path))

        matched_groups = []
        for (beam_name, base), group in groups.items():
            if len(group) > 1:
                matched_groups.append((beam_name, group))
            else:
                unmatched.extend(group)

        if M_priority is not None:
            matched_groups = sorted(matched_groups, key=lambda x: M_priority[x[0]])
        else:
            matched_groups = sorted(matched_groups, key=lambda x: int(x[0][1:]))

        matched_groups = [g for (_, g) in matched_groups]

        return matched_groups, sorted(unmatched)

    def load_model(model_class, checkpoint_path, **kwargs):
        model = model_class(**kwargs).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.eval()
        return model

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
        if ignore_polarization:
            if not isinstance(obs_file_path, list):
                raise ValueError(
                    "In observation mode ignoring polarization, observation data should be a list of [pol1_dir, pol2_dir, ...].")
            else:
                matched, _ = match_polarization_files(
                    sorted([f for f in Path(obs_file_path[0]).iterdir() if f.suffix in [".fil", ".h5"]]) + sorted(
                        [f for f in Path(obs_file_path[1]).iterdir() if f.suffix in [".fil", ".h5"]]), M_list=Beam)
                obs_file_1st = matched[0]
        else:
            if isinstance(obs_file_path, list):
                raise ValueError("In non-pipeline mode, observation data path should be a file, not a list.")
            else:
                if Path(obs_file_path).is_dir():
                    raise ValueError("In non-pipeline mode, observation data path should be a file, not a directory.")
                obs_file_1st = obs_file_path
        # Use pipeline dataset for obs mode
        print("[\033[32mInfo\033[0m] Using observation data from:", obs_file_1st)
        dataset = SETIWaterFullDataset(file_path=obs_file_1st, patch_t=patch_t, patch_f=patch_f,
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
                                              noise_type=nosie_type, rfi_enhance=rfi_enhance, use_fil=use_fil,
                                              background_fil=background_fil)
        pred_dataloader = DataLoader(pred_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    if mode == "dbl":
        global pred_dir
        pred_dir = Path(pred_dir) / "dbl"
        print("[\033[32mInfo\033[0m] Running dual-model comparison mode")
        # Load both models
        dwtnet = load_model(MSWNet, dwtnet_ckpt, **dwtnet_args, **detector_args)
        unet = load_model(UNet, unet_ckpt)
        # Process the same samples with both models
        for idx, batch in enumerate(pred_dataloader):
            if idx >= pred_steps:
                break
            print(f"[\033[32mInfo\033[0m] Processing sample {idx + 1}/{pred_steps}")
            print("[\033[32mInfo\033[0m] Running MSWNet inference...")
            pred(dwtnet, data_mode='dbl', mode=pmode, data=batch, idx=idx, save_dir=pred_dir, device=device,
                 save_npy=False, plot=True, **nms_kargs)
            print("[\033[32mInfo\033[0m] Running UNet inference...")
            pred(unet, data_mode='dbl', mode=pmode, data=batch, idx=idx, save_dir=pred_dir, device=device,
                 save_npy=False, plot=True, **nms_kargs)


    elif mode == "pipeline":
        print("[\033[32mInfo\033[0m] Running pipeline processing mode")

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

            file_groups, unmatched = match_polarization_files(all_files, M_list=Beam)
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
            model = load_model(MSWNet, dwtnet_ckpt, **dwtnet_args, **detector_args)

            if ui:
                if RAW:
                    print("[\033[33mWarn\033[0m] UI mode cannot be used with RAW output, using original config...")
                    app = QApplication(sys.argv)
                renderer = SETIWaterfallRenderer(dataset, model, device, mode=pmode, log_dir=f_log_dir, drift=drift,
                                                 snr_threshold=snr_threshold, min_abs_drift=drift_min_abs,
                                                 verbose=verbose, **nms_kargs, **fsnr_args)
                renderer.setWindowTitle(f"SETI Waterfall Data Processor - {f_log_dir}")
                renderer.show()
                if idx == len(file_list) - 1:
                    sys.exit(app.exec_())
                else:
                    app.exec_()

            else:
                print("[\033[32mInfo\033[0m] Running in no-UI mode, logging only")
                if RAW:
                    print(
                        "[\033[33mWarn\033[0m] You are logging raw data, which may be extremely large. Make sure you have enough space.")
                processor = SETIPipelineProcessor(dataset, model, device, mode=pmode, log_dir=f_log_dir,
                                                  raw_output=RAW, drift=drift, snr_threshold=snr_threshold,
                                                  pad_fraction=pad_fraction, min_abs_drift=drift_min_abs,
                                                  verbose=verbose, **nms_kargs, **fsnr_args)
                processor.process_all_patches()

    else:
        print("[\033[32mInfo\033[0m] Running single-model mode")
        execute0, execute1 = args
        # --- 推理 MSWNet ---
        if execute0:
            print("[\033[32mInfo\033[0m] Running MSWNet inference...")
            dwtnet = load_model(MSWNet, dwtnet_ckpt, **dwtnet_args, **detector_args)
            pred(dwtnet, mode=pmode, data=pred_dataloader, save_dir=pred_dir, device=device, max_steps=pred_steps,
                 save_npy=False, plot=True, group_nms=nms_kargs, group_fsnr=fsnr_args, group_dedrift=dedrift_args)
        # --- 推理 UNet ---
        if execute1:
            print("[\033[32mInfo\033[0m] Running UNet inference...")
            unet = load_model(UNet, unet_ckpt, **unet_args)
            pred(unet, mode=pmode, data=pred_dataloader, save_dir=pred_dir, device=device, max_steps=pred_steps,
                 save_npy=False, plot=True, group_nms=nms_kargs, group_fsnr=fsnr_args, group_dedrift=dedrift_args)


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
