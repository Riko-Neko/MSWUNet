import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import random

from alfalfa import load_alfalfa_spectrum
from dynamic_spectrum_gen import simulate_HI_dynamic_spectrum


class DynamicSpectrumDataset(Dataset):
    def __init__(self, data_dir="./fits", time_frames=224, resample_n=224, noise_std=0.15):
        """
        动态HI频谱数据集

        参数:
        data_dir: 包含FITS文件的目录
        time_frames: 时间帧数
        noise_std: 噪声标准差
        """
        self.data_dir = Path(data_dir)
        self.time_frames = time_frames
        self.resample_n = resample_n
        self.noise_std = noise_std

        # 获取所有FITS文件
        self.fits_files = list(self.data_dir.glob("*.fits"))
        if not self.fits_files:
            raise FileNotFoundError(f"No FITS files found in {data_dir}")

        # 加载第一个文件以获取频率轴信息
        freq_MHz, _ = load_alfalfa_spectrum(str(self.fits_files[0]))
        self.freq_axis = freq_MHz
        self.n_chan = len(freq_MHz)
        self.current_index = 0

    def __len__(self):
        """返回一个较大的数表示无限数据集"""
        return 99999999  # 任意大的数

    def __getitem__(self, idx):
        """
        生成一个随机的动态频谱样本

        返回:
        noisy_spec: 含噪频谱 (1, time_frames, n_chan)
        clean_spec: 干净频谱 (1, time_frames, n_chan)
        rfi_mask: RFI掩膜 (1, time_frames, n_chan)
        """
        try:
            # 随机选择一个FITS文件
            # fname = str(random.choice(self.fits_files))
            # freq_MHz, flux = load_alfalfa_spectrum(fname, verbose=False)

            # 按顺序选择FITS文件，并循环
            fname = str(self.fits_files[self.current_index])
            self.current_index = (self.current_index + 1) % len(self.fits_files)  # 更新索引，达到末尾时重置为0
            freq_MHz, flux = load_alfalfa_spectrum(fname, verbose=False)
        except Exception as e:
            print(f"[Warning] Failed to load {fname}: {e}")
            return self.__getitem__(idx)  # 递归调用，尝试下一个文件

        # 随机生成多分量配置 (1-3个分量)
        num_components = random.randint(1, 3)
        components = []
        for _ in range(num_components):
            offset = random.uniform(-0.8, 0.8)  # 随机频率偏移
            amplitude = random.uniform(0.2, 0.9)  # 随机幅度
            components.append({'offset': offset, 'amplitude': amplitude})

        # 随机生成RFI配置
        rfi_conf = {
            'NBC': random.randint(5, 15),  # 窄带连续RFI数量
            'NBC_amp': random.uniform(15, 30),
            'NBT': random.randint(10, 30),  # 窄带瞬态RFI数量
            'NBT_amp': random.uniform(15, 30),
            'BBT': random.randint(5, 15),  # 宽带瞬态RFI数量
            'BBT_amp': random.uniform(15, 30)
        }

        # 随机生成其他参数
        width_scale = random.uniform(0.95, 1.05)  # 宽度缩放因子
        drift_rate = random.uniform(0., 0.)  # 漂移率d

        # 生成动态频谱
        clean_spec, noisy_spec, rfi_mask = simulate_HI_dynamic_spectrum(
            freq_MHz, flux,
            time_frames=self.time_frames,
            resample_n=self.resample_n,
            noise_std=self.noise_std,
            width_scale=width_scale,
            drift_rate=drift_rate,
            multi_components=components,
            rfi_params=rfi_conf,
            plot=False
        )

        # === 添加归一化操作 ===
        # 1. 计算干净频谱的均值和标准差（用于归一化）
        clean_mean = np.mean(clean_spec)
        clean_std = np.std(clean_spec)

        # 2. 避免除零错误（如果标准差为0则设为1）
        if clean_std < 1e-10:
            clean_std = 1.0

        # 3. 归一化处理
        clean_spec = (clean_spec - clean_mean) / clean_std
        noisy_spec = (noisy_spec - clean_mean) / clean_std
        # =====================

        # 添加通道维度并转换为float32
        noisy_spec = noisy_spec[np.newaxis, :, :].astype(np.float32)
        clean_spec = clean_spec[np.newaxis, :, :].astype(np.float32)
        rfi_mask = rfi_mask[np.newaxis, :, :].astype(np.float32)

        return noisy_spec, clean_spec, rfi_mask
