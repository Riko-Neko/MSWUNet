import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


# Training function with validation, best model saving, checkpoint loading, and force_save_best switch
def train_model(model, train_dataloader, valid_dataloader, criterion, optimizer, scheduler, device, mode='detection',
                num_epochs=100, steps_per_epoch=1000, valid_interval=1, valid_steps=50, checkpoint_dir='./checkpoints',
                log_interval=50, det_level_weights=None, resume_from=None, use_best_weights=False,
                force_save_best=False):
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training log files
    step_log_file = Path(checkpoint_dir) / "training_log.csv"
    epoch_log_file = Path(checkpoint_dir) / "epoch_log.csv"
    best_weights_file = Path(checkpoint_dir) / "best_model.pth"

    # Initialize logs
    if resume_from and os.path.exists(step_log_file):
        # If resuming, append to existing step log
        pass
    else:
        # Otherwise, create new step log with header
        with open(step_log_file, 'w') as f:
            if mode == 'mask':
                f.write(
                    "epoch,global_step,total_loss,spectrum_loss,ssim_loss,rfi_loss,detection_loss,alpha,beta,gamma,delta\n")
            elif mode == 'detection':
                f.write(
                    "epoch,global_step,total_loss,detection_loss,denoise_loss,loc_loss,class_loss,conf_loss,n_matched,lambda_denoise\n")

    if resume_from and os.path.exists(epoch_log_file):
        # Load existing epoch log
        epoch_log = pd.read_csv(epoch_log_file).to_dict('records')
    else:
        epoch_log = []
        with open(epoch_log_file, 'w') as f:
            f.write("epoch,train_loss,valid_loss,epoch_time\n")

    # Determine the best validation loss and epoch from epoch log
    best_valid_loss = float('inf')
    if epoch_log and not (resume_from and force_save_best):
        valid_epochs = [log for log in epoch_log if log['valid_loss'] is not None]
        if valid_epochs:
            best_log = min(valid_epochs, key=lambda x: x['valid_loss'])
            best_valid_loss = best_log['valid_loss']

    # Load checkpoint if resuming
    start_epoch = 0
    if resume_from and os.path.exists(resume_from):
        print(f"[\033[32mInfo\033[0m] Resuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        if use_best_weights:
            best_weights = torch.load(best_weights_file, map_location=device)
            model.load_state_dict(best_weights['model_state_dict'], strict=False)
            print(f"[\033[32mInfo\033[0m] Loaded best weights from {best_weights_file}")
        else:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print(f"[\033[32mInfo\033[0m] Loaded model state from {resume_from}")
        start_epoch = checkpoint['epoch'] + 1  # Start from the next epoch

        # Load criterion state for mask mode
        if mode == 'mask':
            criterion.step = checkpoint['criterion_step']
            criterion.mse_moving_avg = checkpoint['mse_moving_avg']

        print(f"[\033[32mInfo\033[0m] Resumed at epoch {start_epoch}")
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("[\033[32mInfo\033[0m] Optimizer state loaded.")
        except Exception as e:
            print(f"[\033[33mWarn\033[0m]: failed to load optimizer state: {e}")
        if force_save_best:
            print("[\033[32mInfo\033[0m] Forcing best model save with reset validation loss.")
            best_valid_loss = float('inf')  # Reset best validation loss when resuming with force_save_best
    else:
        print("[\033[32mInfo\033[0m] Starting training from scratch.")

    total_loss = 0.
    mask = None
    pprob = None
    spectrum_loss = None
    ssim_loss = None
    rfi_loss = None
    pdet_loss = None
    current_alpha = None
    current_beta = None
    current_gamma = None
    current_delta = None
    gt_boxes = None
    raw_preds = None
    denoise_loss = None
    detection_loss = None
    det_loss_loc = None
    det_loss_class = None
    det_loss_conf = None
    det_n_matched = None
    lambda_denoise = None

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()
        epoch_losses = []

        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Training phase
        model.train()
        train_progress = tqdm(total=steps_per_epoch, desc=f"Training Epoch {epoch + 1}", colour='green')

        for step in range(steps_per_epoch):
            try:
                if mode == 'detection':
                    noisy, clean, gt_boxes = next(iter(train_dataloader))
                else:  # "mask" as default
                    noisy, clean, mask, pprob = next(iter(train_dataloader))
            except StopIteration:
                train_dataloader = iter(train_dataloader)
                if mode == 'detection':
                    noisy, clean, gt_boxes = next(train_dataloader)
                else:  # "mask" as default
                    noisy, clean, mask, pprob = next(train_dataloader)

            noisy = noisy.to(device)
            clean = clean.to(device)

            if mode == 'detection':
                gt_boxes = gt_boxes.to(device)
            else:
                mask = mask.to(device)
                pprob = pprob.to(device)

            # Forward pass
            if mode == 'detection':
                denoised, raw_preds = model(noisy)
            else:
                denoised, rfi_mask, plogits = model(noisy)

            # Calculate loss

            if mode == 'detection':
                total_loss, metrics = criterion(raw_preds=raw_preds, denoised=denoised, clean=clean, gt_boxes=gt_boxes)
                detection_loss = metrics["detection_loss"]
                denoise_loss = metrics["denoise_loss"]
                det_loss_loc = metrics["loc_loss"]
                det_loss_class = metrics["class_loss"]
                det_loss_conf = metrics["conf_loss"]
                det_n_matched = metrics["matched_count"]
                lambda_denoise = metrics["lambda_denoise"]
            else:  # "mask" as default
                total_loss, metrics = criterion(denoised, rfi_mask, plogits, clean, mask, pprob)
                spectrum_loss = metrics["spectrum_loss"]
                ssim_loss = metrics["ssim_loss"]
                rfi_loss = metrics["rfi_loss"]
                pdet_loss = metrics["detection_loss"]
                current_alpha = metrics["alpha"]
                current_beta = metrics["beta"]
                current_gamma = metrics["gamma"]
                current_delta = metrics["delta"]

            epoch_losses.append(total_loss.item())

            # Backpropagation and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Update progress bar
            if mode == 'detection':
                train_progress.set_postfix({
                    'loss': total_loss.item(),
                    'deno': denoise_loss,
                    'det': detection_loss,
                    'loc': det_loss_loc.item(),
                    'class': det_loss_class.item(),
                    'conf': det_loss_conf.item(),
                    'n_matched': det_n_matched.item(),
                    'λ': lambda_denoise.item() if hasattr(lambda_denoise, 'item') else lambda_denoise
                })
            else:
                train_progress.set_postfix({
                    'loss': total_loss.item(),
                    'spec': spectrum_loss.item(),
                    'ssim': ssim_loss.item(),
                    'rfi': rfi_loss.item(),
                    'det': pdet_loss.item(),
                    'α': current_alpha,
                    'β': current_beta,
                    'γ': current_gamma,
                    'δ': current_delta
                })

            train_progress.update(1)

            # Log step data
            if step % log_interval == 0:
                with open(step_log_file, 'a') as f:
                    if mode == 'detection':
                        f.write(
                            f"{epoch},{step},{total_loss.item():.6f},"f"{detection_loss:.6f},{denoise_loss:.6f},"f"{det_loss_loc:.6f},{det_loss_class:.6f},{det_loss_conf:.6f},{det_n_matched},"f"{lambda_denoise.item() if hasattr(lambda_denoise, 'item') else lambda_denoise:.6f}\n")
                    else:  # "mask" as default
                        f.write(
                            f"{epoch},{criterion.step},{total_loss.item():.6f},"f"{spectrum_loss.item():.6f},{ssim_loss.item():.6f},{rfi_loss.item():.6f},{pdet_loss.item():.6f},"f"{current_alpha:.4f},{current_beta:.4f},{current_gamma:.4f},{current_delta:.4f}\n")

        train_progress.close()
        avg_train_loss = np.mean(epoch_losses)

        # Validation phase
        valid_loss = None
        if valid_dataloader and (epoch + 1) % valid_interval == 0:
            model.eval()
            valid_losses = []
            valid_progress = tqdm(total=valid_steps, desc=f"Validation Epoch {epoch + 1}", colour='yellow')

            for i in range(valid_steps):
                try:
                    if mode == 'detection':
                        noisy, clean, gt_boxes = next(iter(valid_dataloader))
                    else:  # "mask" as default
                        noisy, clean, mask, pprob = next(iter(valid_dataloader))
                except StopIteration:
                    valid_dataloader = iter(valid_dataloader)
                    if mode == 'detection':
                        noisy, clean, gt_boxes = next(valid_dataloader)
                    else:  # "mask" as default
                        noisy, clean, mask, pprob = next(valid_dataloader)

                noisy = noisy.to(device)
                clean = clean.to(device)

                if mode == 'detection':
                    gt_boxes = gt_boxes.to(device)
                else:
                    mask = mask.to(device)
                    pprob = pprob.to(device)

                with torch.no_grad():
                    if mode == 'detection':
                        denoised, raw_preds = model(noisy)
                        total_loss, metrics = criterion(raw_preds, denoised, clean, gt_boxes, det_level_weights)
                    else:
                        denoised, rfi_mask, plogits = model(noisy)
                        total_loss, metrics = criterion(denoised, rfi_mask, plogits, clean, mask, pprob)

                    valid_losses.append(total_loss.item())

                valid_progress.set_postfix({'loss': total_loss.item()})
                valid_progress.update(1)

            valid_progress.close()
            valid_loss = np.mean(valid_losses)
            print(f"Validation Loss: {valid_loss:.4f}")

            if valid_loss is not None:
                scheduler.step()

            # Save best model
            if valid_loss is not None and valid_loss < best_valid_loss:
                if valid_loss > 0.:
                    best_valid_loss = valid_loss
                    best_epoch = epoch + 1
                    best_model_path = Path(checkpoint_dir) / "best_model.pth"
                    save_dict = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': valid_loss,
                    }

                    # Add mode-specific items
                    if mode == 'mask':
                        save_dict['criterion_step'] = criterion.step
                        save_dict['mse_moving_avg'] = criterion.mse_moving_avg

                    torch.save(save_dict, best_model_path)
                    print(
                        f"\033[32mSaved best model (epoch {best_epoch}) with validation loss: {best_valid_loss:.6f}\033[0m")
                else:
                    print(
                        f"\033[31mOpps! invalid loss(<0), check if there is gradient explosion.\033[0m")

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
        save_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_train_loss,
        }

        # Add mode-specific items
        if mode == 'mask':
            save_dict['criterion_step'] = criterion.step
            save_dict['mse_moving_avg'] = criterion.mse_moving_avg

        torch.save(save_dict, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

        # Remove old checkpoints (keep last 3)
        if epoch > 2:
            old_checkpoint = Path(checkpoint_dir) / f"model_epoch_{epoch - 2}.pth"
            if old_checkpoint.exists():
                os.remove(old_checkpoint)
                print(f"Removed old checkpoint: {old_checkpoint}")
