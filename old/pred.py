import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from dataset import generate_dataset, SpectraDataset
from model.net import FreqNet
from model.vae import DenoisingVAE
from train import feature_length

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


# 定义推理函数
def denoise_spectrum(noisy_spectrum):
    """对输入光谱进行降噪并返回降噪结果和概率"""
    noisy_tensor = noisy_spectrum.clone().detach().float().to(device)
    with torch.no_grad():
        if MODEL_MAP[model_type][3] == 1:
            denoised_tensor, _, _ = model(noisy_tensor)
            prob = None
        elif MODEL_MAP[model_type][3] == 0:
            denoised_tensor, prob = model(noisy_tensor)
    return denoised_tensor, prob


# 可视化与保存
def result_plot(clean, noisy, denoised, index):
    """分别保存clean、noisy和denoised光谱图"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    prefix = f'pred_plot/sample_{index}_{timestamp}'

    # 保存数据
    np.save(f'{prefix}_clean.npy', clean)
    np.save(f'{prefix}_noisy.npy', noisy)
    np.save(f'{prefix}_denoised.npy', denoised)

    # 分别绘制三张图
    for data, name in zip([clean, noisy, denoised], ['Clean', 'Noisy', 'Denoised']):
        plt.figure(figsize = (10, 5))
        plt.plot(data, label = name, linewidth = 2)
        plt.title(f'Sample {index} - {name} Spectrum')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{prefix}_{name.lower()}_plot.png')
        plt.close()


def comparison_plot(clean, noisy, denoised, prob_tar, prob, index):
    """绘制四线二区对比图，包含prob_tar和prob的颜色渐变和高概率点标记"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fig, axs = plt.subplots(2, 1, figsize = (10, 15), sharex = True)

    # 第一个子图：noisy + denoised + prob
    axs[0].plot(noisy, label = 'Noisy', linestyle = '--', color = 'orange')
    if prob is not None:
        prob = prob.squeeze()
        # 用颜色渐变表示概率
        axs[0].scatter(range(len(prob)), denoised, c = prob, cmap = 'hot', s = 10, alpha = 0.5)
        # 标记高概率点（>=0.9）
        high_prob_indices = np.where(prob >= 0.95)[0]
        axs[0].scatter(high_prob_indices, denoised[high_prob_indices], color = 'red', s = 20, label = 'High Prob')
    axs[0].plot(denoised, label = 'Denoised', linewidth = 2, color = 'green')
    axs[0].set_title(f'Sample {index} - Noisy vs Denoised')
    axs[0].legend()
    axs[0].grid(True)

    # 第二个子图：clean + noisy + prob_tar
    axs[1].plot(noisy, label = 'Noisy', linestyle = '--', color = 'orange')
    if prob_tar is not None:
        prob_tar = prob_tar.squeeze().cpu().numpy()
        # 用颜色渐变表示概率
        axs[1].scatter(range(len(prob_tar)), clean, c = prob_tar, cmap = 'hot', s = 10, alpha = 0.5)
        # 标记高概率点（>=0.9）
        high_prob_indices = np.where(prob_tar >= 0.9)[0]
        axs[1].scatter(high_prob_indices, clean[high_prob_indices], color = 'red', s = 20, label = 'High Prob Tar')
    axs[1].plot(clean, label = 'Clean', linewidth = 2, color = 'blue')
    axs[1].set_title(f'Sample {index} - Clean vs Noisy')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(f'pred_plot/sample_{index}_{timestamp}_comparison.png')
    plt.close()


if __name__ == '__main__':
    # 设备设置
    device_id = 0
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model_type = 'resnet'
    model = MODEL_MAP[model_type][0](**MODEL_MAP[model_type][1]).to(device)
    model.load_state_dict(
        torch.load(os.path.join('weights', MODEL_MAP[model_type][2]), weights_only = True, map_location = device))
    model.eval()

    # 创建输出文件夹
    os.makedirs('pred_plot', exist_ok = True)

    # 生成数据集
    freqs = np.linspace(1419.205, 1421.605, 1024)
    print("Preparing data...")
    clean_spectra, noisy_spectra, prob_tar = generate_dataset(data_type = 'synthetic',
                                                              freqs = freqs,
                                                              num_samples = 100,
                                                              fits_dir = None)

    # 创建数据集和加载器
    train_dataset = SpectraDataset(clean_spectra, noisy_spectra, prob_tar)

    # 计算统计量
    stats_path = 'dataset_stats.json'
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        train_dataset.clean_mean = stats['clean']['mean']
        train_dataset.clean_std = stats['clean']['std']
        train_dataset.noisy_mean = stats['noisy']['mean']
        train_dataset.noisy_std = stats['noisy']['std']
        print(f"Loaded Clean Mean: {stats['clean']['mean']:.4f} ± {stats['clean']['std']:.4f}")
        print(f"Loaded Noisy Mean: {stats['noisy']['mean']:.4f} ± {stats['noisy']['std']:.4f}")
    else:
        raise ValueError("Dataset stats not found.")

    N = 50
    for i in tqdm(range(N), desc = 'Inference'):
        noisy_np, clean_np, prob_tar_np = train_dataset[i]

        noisy = noisy_np.clone().detach()
        noisy = noisy.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, 1024]

        with torch.no_grad():
            denoised, prob = denoise_spectrum(noisy)
            denoised = denoised.squeeze().cpu().numpy()
            prob = prob.squeeze().cpu().numpy() if prob is not None else None

        # result_plot(clean_np, noisy_np, denoised, i)
        comparison_plot(clean_np, noisy_np, denoised, prob_tar_np, prob, i)
