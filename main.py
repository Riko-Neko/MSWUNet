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
from model.DWTNet import DWTNet
from utils.loss_func import CombinedLoss
from utils.train_core import train_model


# Main function
def main():
    parser = argparse.ArgumentParser(description="Select CUDA device")
    parser.add_argument('-d', '--device', type=int, default=0,
                        help='CUDA device ID, default is 0')
    args = parser.parse_args()
    cuda_id = args.device

    # Set device
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{cuda_id}")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Create datasets
    tchans = 144
    fchans = 1024
    df = 7.5
    dt = 1.0
    drift_min = -4.0
    drift_max = 4.0
    drift_min_abs = df // (tchans * dt)
    train_dataset = DynamicSpectrumDataset(tchans=tchans, fchans=fchans, df=df, dt=dt, fch1=None, ascending=False,
                                           drift_min=drift_min, drift_max=drift_max, drift_min_abs=0.2,
                                           snr_min=10.0, snr_max=20.0, width_min=10, width_max=15, num_signals=(1, 1),
                                           noise_std_min=0.025, noise_std_max=0.05)
    valid_dataset = DynamicSpectrumDataset(tchans=tchans, fchans=fchans, df=df, dt=dt, fch1=None, ascending=False,
                                           drift_min=drift_min, drift_max=drift_max, drift_min_abs=0.2,
                                           snr_min=10.0, snr_max=20.0, width_min=10, width_max=15, num_signals=(1, 1),
                                           noise_std_min=0.025, noise_std_max=0.05)

    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=16,
        num_workers=0,
        pin_memory=True
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=16,
        num_workers=0,
        pin_memory=True
    )

    # Initialize model (assuming DWTNet outputs two tensors)
    model = DWTNet(in_chans=1, dim=64, levels=[2, 4, 8, 16], wavelet_name='db4').to(device)
    # model = UNet().to(device)
    summary(model, input_size=(1, 1, 144, 1024))

    # Training configuration
    num_epochs = 1000
    steps_per_epoch = 200
    valid_interval = 1
    valid_steps = 30

    # Loss function and optimizer
    criterion = CombinedLoss(device, alpha=0.5, beta=0.5, gamma=0., delta=0.1, momentum=0.99, fixed_g_d=True)
    optimizer = optim.Adam(list(model.parameters()) + list(criterion.parameters()), lr=5e-4, weight_decay=1e-7)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer,
    #     mode='min',
    #     factor=0.5,
    #     patience=5,
    #     verbose=True,
    #     min_lr=1e-9
    # )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1.0e-18)

    # Check for latest checkpoint to resume from
    # checkpoint_dir = "./checkpoints/unet"
    checkpoint_dir = "./checkpoints/dwtnet"
    checkpoint_files = list(Path(checkpoint_dir).glob("model_epoch_*.pth"))

    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.stem.split('_')[-1]))
        if latest_checkpoint.exists():
            resume_from = latest_checkpoint
        else:
            resume_from = None
    else:
        resume_from = None

    # Start training
    train_model(
        model=model,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        valid_interval=valid_interval,
        valid_steps=valid_steps,
        checkpoint_dir=checkpoint_dir,
        resume_from=resume_from
    )


if __name__ == "__main__":
    main()
