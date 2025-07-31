import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm

import ezCol
from dataset import generate_synthetic_dataset, SpectraDataset, InfiniteSpectraDataset
from model.net import FreqNet
from model.vae import DenoisingVAE

freqBinQty = getattr(ezCol, 'ezColFreqBinQty', 1024)  # Frequency resolution
feature_length = freqBinQty  # Feature length
freq_num = feature_length
batch_size = 128  # Batch size
num_workers = 0   # Number of loader threads
num_epochs = 3000
recompute = False  # Whether to recompute statistics
load_model = False  # Whether to load a pre-trained model
model_type = 'resnet'  # Model type: 'vae' or 'resnet'
device_id = 0  # GPU device ID

if torch.cuda.is_available():
    device = torch.device(f'cuda:{device_id}')  # 优先使用 NVIDIA GPU（适用于 Linux/Windows）
    device_name = torch.cuda.get_device_name(device_id)
elif torch.backends.mps.is_available():
    device = torch.device("mps")   # Apple M1/M2/M3 GPU 支持
    device_name = "Apple MPS"
else:
    device = torch.device("cpu")   # 回退到 CPU
    device_name = "CPU"

print(f"Using device: {device} ({device_name})")

MODEL_MAP = {
    'vae': [
        DenoisingVAE,
        {'latent_dim': 128, 'feature_length': feature_length},
        'best_vae_model.pth',
        1
    ],
    'resnet': [
        FreqNet,
        {'in_chans': 1, 'base_channels': 32},
        'best_net_model.pth',
        0
    ]
}

if __name__ == '__main__':
    # Generate frequency grid
    freqs = np.linspace(1419.205, 1421.605, freq_num)

    # Compute or load statistics
    stats_path = 'dataset_stats.json'
    if os.path.exists(stats_path) and not recompute:
        with open(stats_path) as f:
            stats = json.load(f)
        print(f"Loaded Clean Mean: {stats['clean']['mean']:.4f} ± {stats['clean']['std']:.4f}")
        print(f"Loaded Noisy Mean: {stats['noisy']['mean']:.4f} ± {stats['noisy']['std']:.4f}")
    else:
        print("Computing statistics from a sample batch...")
        clean_spectra, noisy_spectra, prob_tar = generate_synthetic_dataset(freqs, num_samples = 100000)
        temp_dataset = SpectraDataset(clean_spectra, noisy_spectra, prob_tar)
        temp_dataset.compute_stats(sample_type = 'both', exclude_zero = True)
        stats = temp_dataset.get_stats()
        print(f"Computed Clean Mean: {stats['clean']['mean']:.4f} ± {stats['clean']['std']:.4f}")
        print(f"Computed Noisy Mean: {stats['noisy']['mean']:.4f} ± {stats['noisy']['std']:.4f}")
        with open(stats_path, 'w') as f:
            json.dump(stats, f)

    # Create infinite data generator
    train_dataset = InfiniteSpectraDataset(freqs, batch_size, stats)
    train_loader = DataLoader(train_dataset, batch_size = None, num_workers = num_workers)

    # Initialize model
    model = MODEL_MAP[model_type][0](**MODEL_MAP[model_type][1]).to(device)
    weights_path = os.path.join('weights', MODEL_MAP[model_type][2])
    if load_model:
        print(f"Loading pre-trained model {weights_path}")
        if os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path, map_location = device))
        else:
            raise FileNotFoundError("Pre-trained model not found.")


    def vae_loss(recon_x, x, mu, logvar, delta = 0.5):
        # --------------------------
        # 1. 重构损失改用Huber Loss
        # --------------------------
        recon_loss = F.huber_loss(
            recon_x,
            x,
            reduction = 'mean',  # 使用均值而非总和，避免尺度问题
            delta = delta  # 控制二次/线性区域的阈值，可调节超参
        )

        # --------------------------
        # 2. KL散度项（保持原结构）
        # --------------------------
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())  # 使用均值

        # --------------------------
        # 3. 多分辨率STFT损失（保持归一化）
        # --------------------------
        stft_loss = 0
        for n_fft in [64, 128, 256]:
            stft_loss += multi_resolution_stft(recon_x, x, n_fft)

        # --------------------------
        # 4. 平衡损失项权重
        # --------------------------
        return recon_loss + 0.5 * kl_loss + 0.1 * stft_loss  # 权重可调


    def multi_resolution_stft(x, y, n_fft):
        window = torch.hann_window(n_fft, device = x.device)
        X = torch.stft(
            x.squeeze(1),
            n_fft = n_fft,
            hop_length = n_fft // 4,
            win_length = n_fft,
            window = window,
            return_complex = True
        )
        Y = torch.stft(
            y.squeeze(1),
            n_fft = n_fft,
            hop_length = n_fft // 4,
            win_length = n_fft,
            window = window,
            return_complex = True
        )
        loss = F.l1_loss(torch.abs(X), torch.abs(Y), reduction = 'sum')
        return loss / (X.numel() + 1e-8)  # 归一化损失


    lossF = nn.HuberLoss(delta = 0.1)
    loss_prob = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr = 0.0001, weight_decay = 1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 20, eta_min = 1e-11, verbose = True)

    # Print model summary
    summary(model, input_size = (batch_size, 1, feature_length))

    # Training loop
    best_loss = float('inf')
    steps_per_epoch = 256  # Number of steps per epoch

    if MODEL_MAP[model_type][3] == 1:
        for epoch in range(num_epochs):
            model.train().to(device)
            total_loss = 0.0
            recon_loss = 0.0
            kl_loss = 0.0

            bar = tqdm(range(steps_per_epoch), desc = f'Epoch {epoch + 1}/{num_epochs}')
            for step in bar:
                try:
                    noisy, clean, prob_tar = next(iter(train_loader))
                    noisy = noisy.unsqueeze(1).float().to(device)
                    clean = clean.unsqueeze(1).float().to(device)
                    prob_tar = prob_tar.unsqueeze(1).float().to(device)

                    optimizer.zero_grad()
                    recon_batch, mu, logvar = model(noisy)
                    loss = vae_loss(recon_batch, clean, mu, logvar)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    total_loss += loss.item()
                    recon_loss += F.mse_loss(recon_batch, clean, reduction = 'sum').item()
                    kl_loss += 0.1 * (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())).item()

                    bar.set_postfix({
                        'Loss': f"{loss.item() / len(noisy):.4e}",
                        'Recon': f"{recon_loss / ((step + 1) * batch_size):.4e}",
                        'KL': f"{kl_loss / ((step + 1) * batch_size):.4e}"
                    })

                except StopIteration:
                    break

            avg_loss = total_loss / (steps_per_epoch * batch_size)
            scheduler.step()

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), weights_path)

            print(
                f"Epoch {epoch + 1} | Avg Loss: {avg_loss:.4e} | Recon: {recon_loss / (steps_per_epoch * batch_size):.4e} | KL: {kl_loss / (steps_per_epoch * batch_size):.4e}")

    if MODEL_MAP[model_type][3] == 0:
        for epoch in range(num_epochs):
            model.train().to(device)
            total_loss = 0.0
            total_recon_loss = 0.0
            total_prob_loss = 0.0

            bar = tqdm(range(steps_per_epoch), desc = f'Epoch {epoch + 1}/{num_epochs}')
            for step in bar:
                try:
                    noisy, clean, prob_tar = next(iter(train_loader))
                    noisy = noisy.unsqueeze(1).float().to(device)
                    clean = clean.unsqueeze(1).float().to(device)
                    prob_tar = prob_tar.unsqueeze(1).float().to(device)

                    optimizer.zero_grad()
                    denoised, prob = model(noisy)

                    recon_loss = lossF(denoised, clean)
                    prob_loss = loss_prob(prob, prob_tar)

                    # # --- 加权和反向传播 ---
                    # loss = recon_loss + prob_loss * 0.01
                    # loss.backward()

                    # --- 分步反向传播 ---
                    loss = recon_loss + prob_loss
                    recon_loss.backward(retain_graph = True)
                    (prob_loss * 0.01).backward()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    total_loss += loss.item()

                    total_recon_loss += recon_loss.item()
                    total_prob_loss += prob_loss.item()

                    bar.set_postfix({
                        'Loss': f"{loss.item():.4e}",
                        'Recon': f"{recon_loss.item():.4e}",
                        'Prob': f"{prob_loss.item():.4e}"
                    })

                except StopIteration:
                    break

            avg_loss = total_loss / steps_per_epoch
            avg_recon = total_recon_loss / steps_per_epoch
            avg_prob = total_prob_loss / steps_per_epoch

            scheduler.step()

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), weights_path)

            # 使用累计的平均值进行输出
            print(f"Epoch {epoch + 1} | Avg Loss: {avg_loss:.4e} | Recon: {avg_recon:.4e} | Prob: {avg_prob:.4e}")

    # Stop data generation thread
    train_dataset.stop()
    torch.save(model.state_dict(), weights_path)
