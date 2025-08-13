import os
import random

import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader

from gen.SETIgen import sim_dynamic_spec_seti


class DynamicSpectrumDataset(Dataset):
    def __init__(self,
                 tchans=224, fchans=224, df=1.0, dt=1.0, fch1=None, ascending=False,
                 drift_min=-2.0, drift_max=2.0, drift_min_abs=0.2,
                 snr_min=10.0, snr_max=30.0,
                 width_min=1.0, width_max=5.0,
                 num_signals=(0, 1),
                 noise_std_min=0.05, noise_std_max=0.2):
        """
        动态生成式数据集构造函数，参数动态适应频率和时间通道数。

        参数:
            tchans, fchans, df, dt, fch1, ascending: 与动态频谱函数对应的频谱尺寸参数
            drift_min, drift_max: 漂移率范围 (Hz/s)
            drift_min_abs: 最小绝对漂移率 (Hz/s)，确保信号漂移率绝对值不低于此值
            snr_min, snr_max: 信噪比范围
            width_min, width_max: 频谱宽度范围 (Hz)
            num_signals: 信号个数范围 (min, max)
            noise_std_min, noise_std_max: 噪声标准差范围
        """
        self.tchans = tchans
        self.fchans = fchans
        self.df = df
        self.dt = dt
        self.fch1 = fch1 if fch1 is not None else 1.42e9  # 默认 1.42 GHz
        self.ascending = ascending
        self.drift_min = drift_min
        self.drift_max = drift_max
        self.drift_min_abs = drift_min_abs
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.width_min = width_min
        self.width_max = width_max
        self.num_signals = num_signals
        self.noise_std_min = noise_std_min
        self.noise_std_max = noise_std_max

        # 动态计算总带宽和总时间
        self.total_bandwidth = self.fchans * self.df
        self.total_time = self.tchans * self.dt

    def __len__(self):
        return 10 ** 9  # 虚拟一个很大的长度

    def __getitem__(self, idx):
        # 随机生成信号列表
        n_signals = random.randint(self.num_signals[0], self.num_signals[1])
        if np.random.random() < 0.01:
            n_signals += 1  # 1% 的概率增加一个SETI信号

        if np.random.random() < 0.1:  # 10% 的概率不生成任何信号
            n_signals = 0

        # 生成判据
        if n_signals == 0:
            phy_prob = 0.
        else:
            phy_prob = 1.

        signals = []
        for i in range(n_signals):
            # 随机路径类型
            path_type = random.choices(['constant', 'sine', 'squared', 'rfi'],
                                       weights=[0.4, 0.2, 0.4, 0.])[0]
            # 默认信号参数
            margin = int(0.2 * self.fchans)
            # 随机漂移率，确保绝对值不低于 drift_min_abs
            while True:
                drift_rate = random.uniform(self.drift_min, self.drift_max)
                if abs(drift_rate) >= self.drift_min_abs:
                    break
            if drift_rate < 0:
                f_index = self.fchans // 2 + np.random.randint(0, self.fchans // 2 - margin)
            else:
                f_index = self.fchans // 2 - np.random.randint(0, self.fchans // 2 - margin)

            # 随机信噪比和宽度
            snr = random.uniform(self.snr_min, self.snr_max)
            width = random.uniform(self.width_min, self.width_max)

            # 信号参数字典
            sig = {
                'f_index': f_index,
                'drift_rate': drift_rate,
                'snr': snr,
                'width': width,
                'path': path_type,
                't_profile': random.choices(
                    ['pulse', 'sine', 'constant'], weights=[0.25, 0.25, 0.5], k=1)[0],
                'f_profile': random.choices(
                    ['gaussian', 'box', 'sinc', 'lorentzian', 'voigt'],
                    weights=[0.3, 0.2, 0.2, 0.15, 0.15],
                    k=1)[0],
                # rfi 相关参数
                'rfi_type': random.choice(['stationary', 'random_walk']),
                'spread_type': random.choice(['uniform', 'normal']),
                'spread': random.choices(
                    [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 7.5, 10.0, 15.0, 20.0, 25.0]),
                # sine 相关参数
                'squared_drift': drift_rate * 5.e-4
            }

            # 路径类型特定参数
            if path_type == 'sine':
                sig['period'] = random.uniform(0.1 * self.total_time, self.total_time)
                sig['amplitude'] = random.uniform(-0.1 * self.total_bandwidth, 0.1 * self.total_bandwidth)
            elif path_type == 'rfi':
                sig['spread'] = random.uniform(0.005 * self.total_bandwidth, 0.05 * self.total_bandwidth)
                sig['spread_type'] = random.choice(['uniform', 'normal'])
                sig['rfi_type'] = random.choice(['stationary', 'random_walk'])
            elif path_type == 'squared':
                sig['drift_rate'] = random.uniform(0.1 * drift_rate / self.total_bandwidth,
                                                   1 * drift_rate / self.total_bandwidth)

            signals.append(sig)

        # 随机噪声标准差
        noise_std = random.uniform(self.noise_std_min, self.noise_std_max)

        # 随机 RFI 配置
        rfi_params = {
            'NBC': np.random.randint(1, self.fchans // 128),
            'NBC_amp': np.random.uniform(1, 25),
            'NBT': np.random.randint(1, self.tchans // 16),
            'NBT_amp': np.random.uniform(1, 50),
            'BBT': np.random.randint(0, self.fchans // 512),
            'BBT_amp': np.random.uniform(1, 25),
            'LowDrift': np.random.randint(1, 5),
            'LowDrift_amp': np.random.uniform(1, 25),
            'LowDrift_width': np.random.uniform(7.5, 15),
            'NegBand': np.random.randint(0, 2),
            'NegBand_amp': np.random.uniform(0.5, 5),
            'NegBand_width': np.random.uniform(0.3e6, 0.7e6)
        }

        # 生成动态频谱样本
        signal_spec, clean_spec, noisy_spec, rfi_mask = sim_dynamic_spec_seti(
            fchans=self.fchans,
            tchans=self.tchans,
            df=self.df,
            dt=self.dt,
            fch1=self.fch1,
            ascending=self.ascending,
            signals=signals,
            noise_x_mean=0.0,
            noise_x_std=noise_std,
            noise_type='normal',
            rfi_params=rfi_params,
            seed=None,
            plot=False
        )

        # 归一化处理
        mean = np.mean(signal_spec)
        std = np.std(signal_spec)
        if std < 1e-10:
            std = 1.0
        clean_spec = (clean_spec - mean) / std
        noisy_spec = (noisy_spec - mean) / std

        # 添加通道维度并转换为 float32
        clean_spec = clean_spec.astype(np.float32)[np.newaxis, :, :]
        noisy_spec = noisy_spec.astype(np.float32)[np.newaxis, :, :]
        rfi_mask = rfi_mask.astype(np.float32)[np.newaxis, :, :]

        return noisy_spec, clean_spec, rfi_mask, phy_prob


def plot_samples(dataset, kind='clean', num=10, out_dir=None):
    """
    Plot and save specific type of spectrograms from a dynamic dataset.

    Parameters:
    - dataset: instance of DynamicSpectrumDataset or similar
    - kind: 'clean' | 'noisy' | 'mask'
    - num: number of samples to plot
    - out_dir: output directory to save images (default depends on kind)
    """
    assert kind in ['clean', 'noisy', 'mask'], f"Invalid kind: {kind}"

    if out_dir is None:
        out_dir = {
            'clean': '../plot/sim',
            'noisy': '../plot/no',
            'mask': '../plot/rfi'
        }[kind]
    os.makedirs(out_dir, exist_ok=True)

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    iterator = iter(loader)

    for i in range(num):
        try:
            sample = next(iterator)
        except StopIteration:
            break

        if isinstance(sample, (list, tuple)):
            noisy_spec, clean_spec, rfi_mask, _ = sample
        else:
            raise TypeError("Dataset must return a tuple (clean, noisy, mask)")

        if kind == 'clean':
            spec = clean_spec.squeeze().numpy()
        elif kind == 'noisy':
            spec = noisy_spec.squeeze().numpy()
        elif kind == 'mask':
            spec = rfi_mask.squeeze().float().numpy()

        plt.figure(figsize=(15, 3))
        plt.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
        plt.title(f"{kind} spectrogram #{i}")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{kind}_{i:03d}.png"))
        print(f"Saved to {os.path.join(out_dir, f"{kind}_{i:03d}.png")}")
        plt.close()


if __name__ == "__main__":
    tchans = 144
    fchans = 1024
    df = 7.5
    dt = 1.0
    drift_min = -4.0
    drift_max = 4.0
    drift_min_abs = df // (tchans * dt)
    dataset = DynamicSpectrumDataset(tchans=tchans, fchans=fchans, df=df, dt=dt, fch1=None, ascending=False,
                                     drift_min=drift_min, drift_max=drift_max, drift_min_abs=0.2,
                                     snr_min=10.0, snr_max=20.0, width_min=10, width_max=15, num_signals=(1, 1),
                                     noise_std_min=0.025, noise_std_max=0.05)

    """
    参数生成 Refs:
    tchans, fchans, df, dt: 128, 1024, 7.5, 10.0, experimental values 
        from arXiv:2502.20419v1 [astro-ph.IM] 27 Feb 2025 
    drift_rate: -1.0-1.0 Hz/s, the signal drift rate is generally small
        from arXiv:2208.02511v4 [astro-ph.IM] 13 Oct 2022
    snr: 10-20 dB, Referring to previous SETI studies using TurboSETI, we set the S/N threshold to 10.
        from arXiv:2502.20419v1 [astro-ph.IM] 27 Feb 2025;Enriquez et al. 2017; Price et al. 2020; 
        Sheikh et al. 2020; Gajjaret al. 2021; Smith et al. 2021; Traas et al. 2021
    width: 5-7.5 Hz, the signal bandwidth is generally narrower than the frequency resolution
        from arXiv:2502.20419v1 [astro-ph.IM] 27 Feb 2025
    """

    plot_samples(dataset, kind='clean', num=30)
    plot_samples(dataset, kind='noisy', num=30)
    plot_samples(dataset, kind='mask', num=30)
