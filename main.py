r"""
This file contains the main execution code for the SETI data.

Use "
find . -type f -name "*.png" \
  | grep -v "^./abandoned/" \
  | grep -v "^./archived/" \
  | sort \
  | awk -F/ '
    {
      dir = $(1);
      for (i = 2; i < NF; ++i) dir = dir "/" $i;
      file_map[dir]++;
      if (file_map[dir] <= 3) print $0;
    }
  ' \
  | xargs git add -f
  " to sort and add pngs.

Use "
git diff --cached --name-only | grep '\.png$' | xargs git restore --staged
  “ to delete all pngs.

Make sure you do this before committing.

"""
import argparse
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchinfo import summary

from gen.SETIdataset import DynamicSpectrumDataset
from model.DetDWTNet import DWTNet
from utils.loss_func import DetectionCombinedLoss, MaskCombinedLoss
from utils.train_core import train_model

# modes
# mode = 'yolo'
mode = "detection"
# mode = "mask"

# Data config
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
num_signals = (0, 2)
noise_std_min = 0.025
noise_std_max = 0.05
noise_mean_min = 2
noise_mean_max = 3
nosie_type = "chi2"
use_fil = True
fil_folder = Path('./data/33exoplanets/bk')
background_fil = list(fil_folder.rglob("*.fil"))

# Training config
batch_size = 16
num_workers = 0
num_epochs = 1000
steps_per_epoch = 200
valid_interval = 1
valid_steps = 50
log_interval = 50
force_save_best = True

# Optimization config
# checkpoint_dir = "./checkpoints/unet"
checkpoint_dir = "./checkpoints/dwtnet"
det_level_weights = None

# Model config
dwtnet_args = dict(
    in_chans=1,
    dim=64,
    levels=[2, 4, 8, 16],
    wavelet_name='db4',
    extension_mode='periodization',
    N=10,
    num_classes=2,
    dropout=0.05)
unet_args = dict()

regress_loss_args = dict(
    lambda_denoise=1.0,
    loss_type='mse',
    lambda_learnable=False,
    regression_loss_kwargs=dict(
        w_loc=1.0,
        w_class=1.0,
        w_conf=1.0,
        eps=1e-8)
)

mask_loss_args = dict(
    alpha=1.0,
    beta=0.,
    gamma=0.,
    delta=0.,
    momentum=0.99,
    fixed_g_d=True)


# Main function
def main():
    parser = argparse.ArgumentParser(description="Select additional options for training")
    parser.add_argument('-d', '--device',
                        type=int,
                        default=0,
                        help='CUDA device ID, default is 0')
    parser.add_argument('-l', '--load',
                        action='store_true',
                        help='Load best weights instead of checkpoint weights')
    args = parser.parse_args()
    cuda_id = args.device
    load_best = args.load

    # Set device
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{cuda_id}")
    else:
        device = torch.device("cpu")

    print(f"[\033[32mInfo\033[0m] Using device: {device}")

    # Create datasets
    train_dataset = DynamicSpectrumDataset(mode=mode, tchans=tchans, fchans=fchans, df=df, dt=dt, fch1=fch1,
                                           ascending=ascending, drift_min=drift_min, drift_max=drift_max,
                                           drift_min_abs=drift_min_abs, snr_min=snr_min, snr_max=snr_max,
                                           width_min=width_min, width_max=width_max, num_signals=num_signals,
                                           noise_std_min=noise_std_min, noise_std_max=noise_std_max,
                                           noise_mean_min=noise_mean_min, noise_mean_max=noise_mean_max,
                                           noise_type=nosie_type, use_fil=use_fil, background_fil=background_fil)

    valid_dataset = DynamicSpectrumDataset(mode=mode, tchans=tchans, fchans=fchans, df=df, dt=dt, fch1=fch1,
                                           ascending=ascending, drift_min=drift_min, drift_max=drift_max,
                                           drift_min_abs=drift_min_abs, snr_min=snr_min, snr_max=snr_max,
                                           width_min=width_min, width_max=width_max, num_signals=num_signals,
                                           noise_std_min=noise_std_min, noise_std_max=noise_std_max,
                                           noise_mean_min=noise_mean_min, noise_mean_max=noise_mean_max,
                                           noise_type=nosie_type, use_fil=use_fil, background_fil=background_fil)

    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    # Loss function and optimizer
    if mode == "detection":
        model = DWTNet(**dwtnet_args)
        # model = UNet(**unet_args)
        criterion = DetectionCombinedLoss(**regress_loss_args)
    else:  # "mask" as default
        model = DWTNet(**dwtnet_args)
        # model = UNet(**unet_args)
        criterion = MaskCombinedLoss(device, **mask_loss_args)

    summary(model, input_size=(1, 1, 116, 1024))

    optimizer = optim.Adam(list(model.parameters()) + list(criterion.parameters()), lr=5e-4, weight_decay=1e-7)

    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True,
    #                                                  min_lr=1e-9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1.0e-12)

    # Check for latest checkpoint to resume from
    checkpoint_files = list(Path(checkpoint_dir).glob("model_epoch_*.pth"))

    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.stem.split('_')[-1]))
        if latest_checkpoint.exists():
            resume_from = latest_checkpoint
        else:
            resume_from = None
    else:
        resume_from = None

    # Moving to device
    model = model.to(device)
    criterion = criterion.to(device)

    # Start training
    train_model(model=model, train_dataloader=train_dataloader, valid_dataloader=valid_dataloader, criterion=criterion,
                optimizer=optimizer, scheduler=scheduler, device=device, mode=mode, num_epochs=num_epochs,
                steps_per_epoch=steps_per_epoch, valid_interval=valid_interval, valid_steps=valid_steps,
                checkpoint_dir=checkpoint_dir, log_interval=log_interval, det_level_weights=det_level_weights,
                resume_from=resume_from, use_best_weights=load_best, force_save_best=force_save_best)


if __name__ == "__main__":
    main()
