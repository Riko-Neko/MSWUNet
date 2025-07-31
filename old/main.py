import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
from datetime import datetime
from pathlib import Path
import pandas as pd
import time
from tqdm import tqdm

from DRUNet import UNet
from DWTNet import DWTNet
from HIdataset import DynamicSpectrumDataset

# 新loss计算公式
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, normalize=True, threshold=1601):
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.normalize = normalize
        self.threshold = threshold  # 切换归一化方法的阈值

        # 用于存储历史损失值
        self.spectrum_loss_history = []
        self.rfi_loss_history = []

        # 用于移动平均计算
        self.spectrum_loss_avg = 0.
        self.rfi_loss_avg = 0.
        self.count = 0

    def forward(self, denoised, rfi_pred, clean, mask):
        # 计算原始损失
        spectrum_loss = self.mse(denoised, clean)
        rfi_loss = self.bce(rfi_pred, mask)

        if self.normalize:
            # 更新历史损失
            self.spectrum_loss_history.append(spectrum_loss.item())
            self.rfi_loss_history.append(rfi_loss.item())
            self.count += 1

            # 确保历史记录不超过阈值
            if len(self.spectrum_loss_history) > self.threshold:
                self.spectrum_loss_history.pop(0)
                self.rfi_loss_history.pop(0)

            # 根据样本数量选择归一化方法
            if self.count <= self.threshold:
                # 前threshold个样本使用最大值归一化
                if len(self.spectrum_loss_history) > 1:
                    spec_max = max(self.spectrum_loss_history)
                    rfi_max = max(self.rfi_loss_history)
                    spectrum_loss = spectrum_loss / (spec_max + 1e-8)
                    rfi_loss = rfi_loss / (rfi_max + 1e-8)
            else:
                # 超过threshold个样本后使用移动平均归一化
                if self.spectrum_loss_avg is None:
                    self.spectrum_loss_avg = np.mean(self.spectrum_loss_history)
                    self.rfi_loss_avg = np.mean(self.rfi_loss_history)
                else:
                    # 使用指数移动平均
                    self.spectrum_loss_avg = 0.9 * self.spectrum_loss_avg + 0.1 * spectrum_loss.item()
                    self.rfi_loss_avg = 0.9 * self.rfi_loss_avg + 0.1 * rfi_loss.item()

                # 归一化
                spectrum_loss = spectrum_loss / (self.spectrum_loss_avg + 1e-8)
                rfi_loss = rfi_loss / (self.rfi_loss_avg + 1e-8)

        # 组合损失
        total_loss = self.alpha * spectrum_loss + self.beta * rfi_loss
        return total_loss, spectrum_loss, rfi_loss

# 动态权重
class AdaptiveWeightCombinedLoss(nn.Module):
    def __init__(self,
                 init_alpha=1.0,  # 初始的alpha值
                 final_alpha=0.7,  # 最终的alpha值
                 transition_start=0.3,  # 开始调整alpha的训练进度点
                 transition_end=0.7,  # 完成调整alpha的训练进度点
                 normalize=True,
                 threshold=1601):
        super(AdaptiveWeightCombinedLoss, self).__init__()
        self.init_alpha = init_alpha
        self.final_alpha = final_alpha
        self.transition_start = transition_start
        self.transition_end = transition_end
        self.beta = 1.0  # beta会根据alpha动态计算
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.normalize = normalize
        self.threshold = threshold

        # 用于存储历史损失值
        self.spectrum_loss_history = []
        self.rfi_loss_history = []

        # 用于移动平均计算
        self.spectrum_loss_avg = 0.
        self.rfi_loss_avg = 0.
        self.count = 0

        # 训练总步数（将在训练开始时设置）
        self.total_steps = 0.
        self.current_step = 0

    def set_total_steps(self, total_steps):
        """设置训练的总步数，用于计算当前进度"""
        self.total_steps = total_steps

    def update_step(self):
        """更新当前训练步数"""
        self.current_step += 1

    def calculate_current_alpha(self):
        """根据训练进度计算当前的alpha值"""
        if self.total_steps is None:
            return self.init_alpha  # 如果未设置总步数，使用初始alpha

        progress = self.current_step / self.total_steps

        # 如果进度小于开始调整的点，使用初始alpha
        if progress <= self.transition_start:
            return self.init_alpha

        # 如果进度大于结束调整的点，使用最终alpha
        if progress >= self.transition_end:
            return self.final_alpha

        # 线性插值计算当前alpha
        factor = (progress - self.transition_start) / (self.transition_end - self.transition_start)
        return self.init_alpha + factor * (self.final_alpha - self.init_alpha)

    def forward(self, denoised, rfi_pred, clean, mask):
        # 计算当前的alpha值
        self.alpha = self.calculate_current_alpha()
        self.beta = 1.0 - self.alpha  # 确保alpha + beta = 1

        # 计算原始损失
        spectrum_loss = self.mse(denoised, clean)
        rfi_loss = self.bce(rfi_pred, mask)

        if self.normalize:
            # 更新历史损失
            self.spectrum_loss_history.append(spectrum_loss.item())
            self.rfi_loss_history.append(rfi_loss.item())
            self.count += 1

            # 确保历史记录不超过阈值
            if len(self.spectrum_loss_history) > self.threshold:
                self.spectrum_loss_history.pop(0)
                self.rfi_loss_history.pop(0)

            # 根据样本数量选择归一化方法
            if self.count <= self.threshold:
                # 前threshold个样本使用最大值归一化
                if len(self.spectrum_loss_history) > 1:
                    spec_max = max(self.spectrum_loss_history)
                    rfi_max = max(self.rfi_loss_history)
                    spectrum_loss = spectrum_loss / (spec_max + 1e-8)
                    rfi_loss = rfi_loss / (rfi_max + 1e-8)
            else:
                # 超过threshold个样本后使用移动平均归一化
                if self.spectrum_loss_avg is None:
                    self.spectrum_loss_avg = np.mean(self.spectrum_loss_history)
                    self.rfi_loss_avg = np.mean(self.rfi_loss_history)
                else:
                    # 使用指数移动平均
                    self.spectrum_loss_avg = 0.9 * self.spectrum_loss_avg + 0.1 * spectrum_loss.item()
                    self.rfi_loss_avg = 0.9 * self.rfi_loss_avg + 0.1 * rfi_loss.item()

                # 归一化
                spectrum_loss = spectrum_loss / (self.spectrum_loss_avg + 1e-8)
                rfi_loss = rfi_loss / (self.rfi_loss_avg + 1e-8)

        # 组合损失
        total_loss = self.alpha * spectrum_loss + self.beta * rfi_loss
        return total_loss, spectrum_loss, rfi_loss


# Combined loss function
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha  # Weight for spectrum restoration
        self.beta = beta    # Weight for RFI detection
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, denoised, rfi_pred, clean, mask):
        # Calculate spectrum restoration loss
        spectrum_loss = self.mse(denoised, clean)
        # Calculate RFI detection loss
        rfi_loss = self.bce(rfi_pred, mask)
        # Combined loss
        total_loss = self.alpha * spectrum_loss + self.beta * rfi_loss
        return total_loss, spectrum_loss, rfi_loss

# Training function with validation and best model saving
def train_model(model, train_dataloader, valid_dataloader, criterion, optimizer, device,
                num_epochs=100, steps_per_epoch=1000, valid_interval=1, valid_steps=50,
                checkpoint_dir='./checkpoints', log_interval=50):

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training log files
    step_log_file = Path(checkpoint_dir) / "training_log.csv"
    epoch_log_file = Path(checkpoint_dir) / "epoch_log.csv"

    # Initialize logs
    with open(step_log_file, 'w') as f:
        f.write("epoch,step,total_loss,spectrum_loss,rfi_loss\n")

    epoch_log = []
    best_valid_loss = float('inf')
    best_epoch = 0

    # Training loop
    for epoch in range(num_epochs):
        epoch_start = time.time()
        epoch_losses = []

        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Training phase
        model.train()
        train_progress = tqdm(total=steps_per_epoch, desc=f"Training Epoch {epoch + 1}")

        for step in range(steps_per_epoch):
            try:
                noisy, clean, mask = next(iter(train_dataloader))
            except StopIteration:
                train_dataloader = iter(train_dataloader)
                noisy, clean, mask = next(train_dataloader)

            noisy = noisy.to(device)
            clean = clean.to(device)
            mask = mask.to(device)

            # Forward pass
            denoised, rfi_mask = model(noisy)

            # Calculate loss
            total_loss, spectrum_loss, rfi_loss = criterion(denoised, rfi_mask, clean, mask)
            epoch_losses.append(total_loss.item())

            # Backpropagation and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Update progress bar
            train_progress.set_postfix({
                'loss': total_loss.item(),
                'spec': spectrum_loss.item(),
                'rfi': rfi_loss.item()
            })
            train_progress.update(1)

            # Log step data
            if step % log_interval == 0:
                with open(step_log_file, 'a') as f:
                    f.write(f"{epoch},{step},{total_loss.item():.6f},"
                            f"{spectrum_loss.item():.6f},{rfi_loss.item():.6f}\n")

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
                    noisy, clean, mask = next(train_dataloader)

                noisy = noisy.to(device)
                clean = clean.to(device)
                mask = mask.to(device)

                with torch.no_grad():
                    denoised, rfi_mask = model(noisy)
                    total_loss, _, _ = criterion(denoised, rfi_mask, clean, mask)
                    valid_losses.append(total_loss.item())

                valid_progress.set_postfix({'loss': total_loss.item()})
                valid_progress.update(1)

            valid_progress.close()
            valid_loss = np.mean(valid_losses)
            print(f"Validation Loss: {valid_loss:.4f}")

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

    print(f"Using device: {device}")

    # Create datasets
    train_dataset = DynamicSpectrumDataset(data_dir="./alfalfa_spectra/train")
    valid_dataset = DynamicSpectrumDataset(data_dir="./alfalfa_spectra/val")

    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=8,
        num_workers=0,
        pin_memory=True
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=8,
        num_workers=0,
        pin_memory=True
    )

    # Initialize model (assuming DWTNet outputs two tensors)
    model = DWTNet(use_multibranch=True).to(device)

    # Loss function and optimizer
    criterion = CombinedLoss(alpha=0.7, beta=0.3)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # Training configuration
    num_epochs = 100
    steps_per_epoch = 1000
    valid_interval = 1
    valid_steps = 50

    # Create checkpoint directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_dir = f"./checkpoints"

    # Start training
    train_model(
        model=model,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        valid_interval=valid_interval,
        valid_steps=valid_steps,
        checkpoint_dir=checkpoint_dir
    )

if __name__ == "__main__":
    main()
