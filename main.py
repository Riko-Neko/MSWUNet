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
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchinfo import summary
from torchmetrics.image import StructuralSimilarityIndexMeasure
from tqdm import tqdm

from SETIdataset import DynamicSpectrumDataset
from model.DWTNet import DWTNet


class CombinedLoss(nn.Module):
    def __init__(self, device, alpha=0.7, beta=0.3, gamma=0., delta=0.1, adjust_threshold=20, momentum=0.9):
        super(CombinedLoss, self).__init__()
        self.alpha_init = alpha  # initial spectrum detection weight
        self.beta_init = beta  # initial RFI detection weight
        self.gamma = nn.Parameter(
            torch.tensor(gamma, device=device)) if gamma != 0. else gamma  # ssim loss weight (0 for observation)
        self.delta = nn.Parameter(torch.tensor(delta, device=device))  # BCE loss for physical detection
        self.adjust_threshold = adjust_threshold  # window size for adjusting weights
        self.momentum = momentum
        self.jitter_period = 30  # cosine jitter period
        self.jitter_amplitude = 0.1  # jitter amplitude
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

        self.mse = nn.MSELoss(reduction='none')
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0], device=device))
        self.dice = DiceLoss()

        # register buffers for moving average and weights
        self.register_buffer('mse_moving_avg', torch.tensor(1.0))
        self.register_buffer('alpha', torch.tensor(alpha))
        self.register_buffer('beta', torch.tensor(beta))
        self.step = 0

        # history for trend detection
        self.spec_history = []
        self.rfi_history = []

        self.trend_threshold = 0.001  # threshold for trend detection

    def forward(self, denoised, rfi_pred, plogits, clean, mask, pprob):
        # mse loss for spectrum detection
        signal_weight = 10.0
        background_weight = 1.
        weight_map = torch.where(clean > 0.,
                                 torch.full_like(clean, signal_weight),
                                 torch.full_like(clean, background_weight))
        spectrum_loss = (self.mse(denoised, clean) * weight_map)
        ssim_loss = 1 - self.ssim(denoised, clean)

        # initialize moving average if it's the first step
        if self.step == 0:
            self.mse_moving_avg.fill_(spectrum_loss.detach().mean())

        # apply moving average normalization to spectrum loss
        normalized_spectrum_loss = spectrum_loss / torch.clamp(self.mse_moving_avg, min=1e-6)
        spectrum_loss_scalar = normalized_spectrum_loss.mean() * (1 - self.gamma) + ssim_loss * self.gamma

        # dice loss for RFI detection
        rfi_loss = self.dice(rfi_pred, mask)

        # update history
        self.spec_history.append(spectrum_loss_scalar.detach().item())
        self.rfi_history.append(rfi_loss.detach().item())

        # ensure history size is within reasonable range
        if len(self.spec_history) > self.adjust_threshold:
            self.spec_history.pop(0)
            self.rfi_history.pop(0)

        # dynamically adjust weights based on trend detection
        if len(self.spec_history) >= 2:  # at least two history entries required
            spec_trend = self._compute_trend(self.spec_history)
            rfi_trend = self._compute_trend(self.rfi_history)

            # case1: both loss decrease - do nothing
            if spec_trend < -self.trend_threshold and rfi_trend < -self.trend_threshold:
                pass

            # case2: spectrum loss decrease but RFI loss increase - decrease spectrum weight
            elif spec_trend < -self.trend_threshold <= rfi_trend:
                new_alpha = max(0.0, self.alpha.item() - 0.01)
                self.alpha.fill_(new_alpha)
                self.beta.fill_(1.0 - new_alpha)

            # case3: rfi loss decrease but spectrum loss increase - decrease RFI weight
            elif rfi_trend < -self.trend_threshold <= spec_trend:
                new_alpha = min(1.0, self.alpha.item() + 0.01)
                self.alpha.fill_(new_alpha)
                self.beta.fill_(1.0 - new_alpha)

            # case4: both loss not decrease - apply jitter
            else:
                # cosine jitter to adjust weights
                jitter = self.jitter_amplitude * np.cos(2 * np.pi * self.step / self.jitter_period)
                new_alpha = np.clip(self.alpha_init + jitter, 0.0, 1.0)
                self.alpha.fill_(new_alpha)
                self.beta.fill_(1.0 - new_alpha)

        # detection loss for physical detection
        detection_loss = self.bce(plogits, pprob)

        # compute total loss
        total_loss = + (self.alpha * spectrum_loss_scalar + self.beta * rfi_loss) * (
                1 - self.delta) + self.delta * detection_loss

        # update moving average
        current_mse_avg = spectrum_loss_scalar.detach()
        self.mse_moving_avg.data = (
                self.momentum * self.mse_moving_avg +
                (1 - self.momentum) * current_mse_avg
        )

        self.step += 1
        return total_loss, spectrum_loss_scalar, ssim_loss, rfi_loss, detection_loss, self.alpha.item(), self.beta.item()

    def _compute_trend(self, history):
        n = len(history)
        if n < 2:
            return 0.0

        # compute trend using linear regression
        x = np.arange(n)
        y = np.array(history)
        x_mean = np.mean(x)
        y_mean = np.mean(y)

        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)

        if abs(denominator) < 1e-8:
            return 0.0

        slope = numerator / denominator
        return slope


class DiceLoss(nn.Module):
    def forward(self, pred, target):
        pred = pred.sigmoid()
        intersection = (pred * target).sum()
        return 1.0 - (2.0 * intersection) / (pred.sum() + target.sum() + 1e-6)


# Training function with validation, best model saving, and checkpoint loading
def train_model(model, train_dataloader, valid_dataloader, criterion, optimizer, scheduler, device,
                num_epochs=100, steps_per_epoch=1000, valid_interval=1, valid_steps=50,
                checkpoint_dir='./checkpoints', log_interval=50, resume_from=None):
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training log files
    step_log_file = Path(checkpoint_dir) / "training_log.csv"
    epoch_log_file = Path(checkpoint_dir) / "epoch_log.csv"

    # Initialize logs
    if resume_from and os.path.exists(step_log_file):
        # If resuming, append to existing step log
        pass
    else:
        # Otherwise, create new step log with header
        with open(step_log_file, 'w') as f:
            f.write("epoch,global_step,total_loss,spectrum_loss,ssim_loss,rfi_loss,alpha,beta\n")

    if resume_from and os.path.exists(epoch_log_file):
        # Load existing epoch log
        epoch_log = pd.read_csv(epoch_log_file).to_dict('records')
    else:
        epoch_log = []
        with open(epoch_log_file, 'w') as f:
            f.write("epoch,train_loss,valid_loss,epoch_time\n")

    # Determine the best validation loss and epoch from epoch log
    if epoch_log:
        valid_epochs = [log for log in epoch_log if log['valid_loss'] is not None]
        if valid_epochs:
            best_log = min(valid_epochs, key=lambda x: x['valid_loss'])
            best_valid_loss = best_log['valid_loss']
        else:
            best_valid_loss = float('inf')
    else:
        best_valid_loss = float('inf')

    # Load checkpoint if resuming
    start_epoch = 0
    if resume_from and os.path.exists(resume_from):
        print(f"[\033[32mInfo\033[0m] Resuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        start_epoch = checkpoint['epoch'] + 1  # Start from the next epoch
        criterion.step = checkpoint['criterion_step']
        criterion.mse_moving_avg = checkpoint['mse_moving_avg']
        print(f"[\033[32mInfo\033[0m] Resumed at epoch {start_epoch}")
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("[\033[32mInfo\033[0m] Optimizer state loaded.")
        except Exception as e:
            print(f"[\033[33mWarning\033[0m]: failed to load optimizer state: {e}")
    else:
        print("[\033[32mInfo\033[0m] Starting training from scratch.")

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()
        epoch_losses = []

        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Training phase
        model.train()
        train_progress = tqdm(total=steps_per_epoch, desc=f"Training Epoch {epoch + 1}")

        for step in range(steps_per_epoch):
            try:
                noisy, clean, mask, pprob = next(iter(train_dataloader))
            except StopIteration:
                train_dataloader = iter(train_dataloader)
                noisy, clean, mask, pprob = next(train_dataloader)

            noisy = noisy.to(device)
            clean = clean.to(device)
            mask = mask.to(device)
            pprob = pprob.to(device)

            # Forward pass
            denoised, rfi_mask, plogits = model(noisy)

            # Calculate loss
            total_loss, spectrum_loss, simm_loss, rfi_loss, detection_loss, current_alpha, current_beta = criterion(
                denoised, rfi_mask, plogits, clean, mask, pprob)
            epoch_losses.append(total_loss.item())

            # Backpropagation and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Update progress bar
            train_progress.set_postfix({
                'loss': total_loss.item(),
                'spec': spectrum_loss.item(),
                'ssim': simm_loss.item(),
                'rfi': rfi_loss.item(),
                'det': detection_loss.item(),
                'α': current_alpha,
                'β': current_beta
            })
            train_progress.update(1)

            # Log step data
            if step % log_interval == 0:
                with open(step_log_file, 'a') as f:
                    f.write(f"{epoch},{criterion.step},{total_loss.item():.6f},"
                            f"{spectrum_loss.item():.6f},{simm_loss.item():.6f},{rfi_loss.item():.6f},{detection_loss.item():.6f},"
                            f"{current_alpha:.4f},{current_beta:.4f}\n")

        train_progress.close()
        avg_train_loss = np.mean(epoch_losses)

        # Validation phase
        valid_loss = None
        if valid_dataloader and (epoch + 1) % valid_interval == 0:
            model.eval()
            valid_losses = []
            valid_progress = tqdm(total=valid_steps, desc=f"Validation Epoch {epoch + 1}")

            for i in range(valid_steps):
                try:
                    noisy, clean, mask = next(iter(valid_dataloader))
                except StopIteration:
                    valid_dataloader = iter(valid_dataloader)
                    noisy, clean, mask, pprob = next(valid_dataloader)

                noisy = noisy.to(device)
                clean = clean.to(device)
                mask = mask.to(device)
                pprob = pprob.to(device)

                with torch.no_grad():
                    denoised, rfi_mask, plogits = model(noisy)
                    total_loss, _, _, _, _, _ = criterion(denoised, rfi_mask, plogits, clean, mask, pprob)
                    valid_losses.append(total_loss.item())

                valid_progress.set_postfix({'loss': total_loss.item()})
                valid_progress.update(1)

            valid_progress.close()
            valid_loss = np.mean(valid_losses)
            print(f"Validation Loss: {valid_loss:.4f}")

            if valid_loss is not None:
                scheduler.step()

            # Save best model
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_epoch = epoch + 1
                best_model_path = Path(checkpoint_dir) / "best_model.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': valid_loss,
                    'criterion_step': criterion.step,
                    'mse_moving_avg': criterion.mse_moving_avg,
                }, best_model_path)
                print(f"Saved best model (epoch {best_epoch}) with validation loss: {best_valid_loss:.6f}")

        # Log epoch data
        epoch_time = time.time() - epoch_start
        epoch_log.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'valid_loss': valid_loss if valid_loss is not None else None,
            'epoch_time': epoch_time
        })
        pd.DataFrame(epoch_log).to_csv(epoch_log_file, index=False)

        # Save checkpoint
        checkpoint_path = Path(checkpoint_dir) / f"model_epoch_{epoch + 1}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_train_loss,
            'criterion_step': criterion.step,
            'mse_moving_avg': criterion.mse_moving_avg,
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

        # Remove old checkpoints (keep last 3)
        if epoch > 2:
            old_checkpoint = Path(checkpoint_dir) / f"model_epoch_{epoch - 2}.pth"
            if old_checkpoint.exists():
                os.remove(old_checkpoint)
                print(f"Removed old checkpoint: {old_checkpoint}")


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
    train_dataset = DynamicSpectrumDataset(tchans=144, fchans=1024, df=7.5, dt=1.0, fch1=None, ascending=False,
                                           drift_min=-1.0, drift_max=1.0,
                                           snr_min=10.0, snr_max=20.0,
                                           width_min=5, width_max=7.5,
                                           num_signals=(0, 1),
                                           noise_std_min=0.05, noise_std_max=0.1)
    valid_dataset = DynamicSpectrumDataset(tchans=144, fchans=1024, df=7.5, dt=1.0, fch1=None, ascending=False,
                                           drift_min=-1.0, drift_max=1.0,
                                           snr_min=10.0, snr_max=20.0,
                                           width_min=5, width_max=7.5,
                                           num_signals=(0, 1),
                                           noise_std_min=0.05, noise_std_max=0.1)

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
    criterion = CombinedLoss(device, alpha=0.5, beta=0.5, adjust_threshold=20, momentum=0.99)
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-7)
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
