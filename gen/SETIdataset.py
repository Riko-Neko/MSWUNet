import os
import random

import numpy as np
import torch
from blimpy import Waterfall
from matplotlib import pyplot as plt
from scipy.stats import truncnorm
from torch.utils.data import Dataset

from gen.SETIgen import sim_dynamic_spec_seti
from utils.det_utils import plot_F_lines

DEBUG = False


class DynamicSpectrumDataset(Dataset):
    def __init__(self, mode='test', tchans=224, fchans=224, df=1.0, dt=1.0, fch1=None, ascending=True, drift_min=-2.0,
                 drift_max=2.0, drift_min_abs=0.2, snr_min=10.0, snr_max=30.0, width_min=1.0, width_max=5.0,
                 num_signals=(0, 1), noise_std_min=0.05, noise_std_max=0.2, noise_mean_min=0.0, noise_mean_max=0.05,
                 noise_type='normal', rfi_enhance=False, use_fil=False, background_fil=None):
        """
        动态生成式数据集构造函数，参数动态适应频率和时间通道数。

        Args:
            mode: 模式，'train' 或 'test'
            tchans: 时间通道数
            fchans: 频率通道数
            df: 频率分辨率
            dt: 时间分辨率
            fch1: 起始频率，默认 1.42 GHz
            ascending: 升序还是降序
            drift_min: 最小漂移率
            drift_max: 最大漂移率
            drift_min_abs: 最小漂移率绝对值
            snr_min: 最小信噪比
            snr_max: 最大信噪比
            width_min: 最小宽度
            width_max: 最大宽度
            num_signals: 信号数量范围，元组 (min, max)
            noise_std_min: 噪声标准差最小值
            noise_std_max: 噪声标准差最大值
            noise_mean_min: 噪声均值最小值
            noise_mean_max: 噪声均值最大值
            noise_type: Distribution to use for synthetic noise, {"chi2", "gaussian", "normal"}, default: "chi2"
            use_fil: Weather to use FILTERBANK file as background noise
            background_fil: Path to background noise file
        """
        self.mode = mode
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
        self.max_num_signals = num_signals[1] + 1
        self.noise_std_min = noise_std_min
        self.noise_std_max = noise_std_max
        self.noise_mean_min = noise_mean_min
        self.noise_mean_max = noise_mean_max
        self.noise_type = noise_type
        self.rfi_enhance = rfi_enhance
        self.waterfall_itr = None if not use_fil else split_waterfall_generator(background_fil, fchans, tchans=tchans,
                                                                                f_shift=[fchans, 8 * fchans])
        # 动态计算总带宽和总时间
        self.total_bandwidth = self.fchans * self.df
        self.total_time = self.tchans * self.dt
        self.t_center = torch.tensor((self.tchans - 1) / 2 / (self.tchans - 1))
        self.t_width = torch.tensor((self.tchans - 1) / (self.tchans - 1))

    def __len__(self):
        return 10 ** 9  # 虚拟一个很大的长度

    def __getitem__(self, idx):
        # 随机生成信号列表
        n_signals = random.randint(self.num_signals[0], self.num_signals[1])
        if np.random.random() < 0.3:
            n_signals += 1  # 30% 的概率增加一个SETI信号

        if np.random.random() < 0.1:  # 10% 的概率不生成任何信号
            n_signals = 0

        # 生成判据
        if self.mode == 'test' or self.mode == 'mask':
            if n_signals == 0:
                phy_prob = 0.
            else:
                phy_prob = 1.
        else:
            phy_prob = None

        signals = []
        for i in range(n_signals):
            # 随机路径类型
            path_type = random.choices(['constant', 'sine', 'squared', 'rfi'],
                                       weights=[0.6, 0.25, 0.35, 0.])[0]
            # 默认信号参数
            margin = int(0.2 * self.fchans)

            def _truncated_normal(a, b, mean=0.0, std=1.2):
                lower = (a - mean) / std
                upper = (b - mean) / std
                return truncnorm.rvs(lower, upper, loc=mean, scale=std)

            while True:
                # ✔ 使用截断正态分布：中心概率最高，两端概率极低，靠近两端也不会显著升高
                x = _truncated_normal(0.0, 1.0, mean=0.5, std=0.05)
                drift_rate = self.drift_min + x * (self.drift_max - self.drift_min)
                # !!⚠️ 较大的 drift rate 在轨迹为抛物线时可能出现类似直线但是无法标记为 candidate 的情况
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
                    ['pulse', 'sine', 'constant'], weights=[0.3, 0.2, 0.5], k=1)[0],
                'f_profile': random.choices(
                    ['gaussian', 'box', 'sinc', 'lorentzian', 'voigt'],
                    weights=[0.3, 0.2, 0.2, 0.15, 0.15],
                    k=1)[0],
                # rfi 相关参数
                'rfi_type': random.choice(['stationary', 'random_walk']),
                'spread_type': random.choice(['uniform', 'normal']),
                'spread': random.choices(
                    [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 7.5, 10.0, 15.0, 20.0, 25.0])
            }

            # 路径类型特定参数
            if path_type == 'sine':
                sig['period'] = random.uniform(1.25 * self.total_time, 2.5 * self.total_time)
                sig['amplitude'] = random.uniform(0.01 * self.total_bandwidth,
                                                  0.03 * self.total_bandwidth) * random.choice([1, -1])
                sig['drift_rate'] = random.uniform(0.01, 0.25) * drift_rate
            elif path_type == 'rfi':
                sig['spread'] = random.uniform(0.005 * self.total_bandwidth, 0.05 * self.total_bandwidth)
                sig['spread_type'] = random.choice(['uniform', 'normal'])
                sig['rfi_type'] = random.choice(['stationary', 'random_walk'])
            elif path_type == 'squared':
                if abs(drift_rate) < 0.1:
                    sig['path'] = 'constant'
                else:
                    sig['drift_rate'] = drift_rate * 5.e-4

            # 时间调制类型参数
            if sig['t_profile'] == 'pulse':
                sig['p_width'] = random.uniform(self.total_time, 100 * self.total_time)
                sig['p_amplitude_factor'] = random.uniform(0.1, 1.0)
                sig['p_num'] = random.randint(1, 5)
                sig['p_min_level_factor'] = random.uniform(0.1, 1.0)
            elif sig['t_profile'] == 'sine':
                sig['s_period'] = random.uniform(0.1 * self.total_time, 1 * self.total_time)
                sig['s_amplitude_factor'] = random.uniform(0.01, 1.0)

            signals.append(sig)

        # 随机噪声标准差
        noise_std = random.uniform(self.noise_std_min, self.noise_std_max)
        # 随机噪声均值
        noise_mean = random.uniform(self.noise_mean_min, self.noise_mean_max)

        # 随机 RFI 配置
        rfi_params = {
            'NBC': np.random.randint(1, self.fchans // 128),
            'NBC_amp': np.random.uniform(1, 25),
            'NBT': np.random.randint(1, self.tchans // 16 + 1),
            'NBT_amp': np.random.uniform(1, 50),
            'BBT': np.random.randint(1, self.fchans // 256),
            'BBT_amp': np.random.uniform(1, 25),
            'LowDrift': np.random.randint(1, 10),
            'LowDrift_amp_factor': np.random.uniform(0.1, 1.0),
            'LowDrift_width': np.random.uniform(7.5, 15)
        }

        args = dict(fchans=self.fchans,
                    tchans=self.tchans,
                    df=self.df,
                    dt=self.dt,
                    fch1=self.fch1,
                    ascending=self.ascending,
                    signals=signals,
                    noise_x_mean=noise_mean,
                    noise_x_std=noise_std,
                    mode=self.mode,
                    noise_type=self.noise_type,
                    rfi_params=rfi_params,
                    seed=None,
                    plot=False,
                    waterfall_itr=self.waterfall_itr,
                    rfi_enhance=self.rfi_enhance)

        # 生成动态频谱样本
        freq_info = None
        if self.mode == 'detection':
            signal_spec, clean_spec, noisy_spec, freq_info = sim_dynamic_spec_seti(**args)
            rfi_mask = None
        elif self.mode == 'mask':
            signal_spec, clean_spec, noisy_spec, rfi_mask = sim_dynamic_spec_seti(**args)
        else:
            signal_spec, clean_spec, noisy_spec, rfi_mask, freq_info = sim_dynamic_spec_seti(**args)

        if DEBUG:
            print(f"[\033[36mDebug\033[0m] Ground truth boxes to generate: {freq_info}, "
                  f"normalized ({np.array(freq_info[2]) / self.fchans}, {np.array(freq_info[3]) / self.fchans})")

        # 归一化处理
        mean = np.mean(signal_spec)
        std = np.std(signal_spec)
        if std < 1e-10:
            std = 1.0

        clean_mean = np.mean(clean_spec)
        clean_std = np.std(clean_spec)
        if clean_std < 1e-10:
            clean_std = 1.0

        # noisy_mean = np.mean(noisy_spec)
        # noisy_std = np.std(noisy_spec)
        # if noisy_std < 1e-10:
        #     noisy_std = 1.0

        clean_spec = (clean_spec - clean_mean) / clean_std
        noisy_spec = (noisy_spec - mean) / std

        # 添加通道维度并转换为 float32
        clean_spec = clean_spec.astype(np.float32)[np.newaxis, :, :]
        noisy_spec = noisy_spec.astype(np.float32)[np.newaxis, :, :]
        if self.mode == 'mask' or self.mode == 'test':
            rfi_mask = rfi_mask.astype(np.float32)[np.newaxis, :, :]

        if self.mode == 'yolo':
            N, classes, f_starts, f_stops = freq_info if freq_info else (0, [], [], [])
            gt_boxes = torch.full((self.max_num_signals, 5), float('nan'), dtype=torch.float32)
            if N > 0:
                t_start, t_stop = 0.0, float(self.tchans - 1)
                t_center = (t_start + t_stop) / 2.0
                t_width = t_stop - t_start
                f_starts = torch.tensor(f_starts, dtype=torch.float32)
                f_stops = torch.tensor(f_stops, dtype=torch.float32)
                f_center = (f_starts + f_stops) / 2.0
                f_width = f_stops - f_starts
                t_center /= (self.tchans - 1)
                t_width /= (self.tchans - 1)
                f_center /= (self.fchans - 1)
                f_width /= (self.fchans - 1)
                classes = torch.tensor(classes, dtype=torch.float32)
                # YOLO format: [class_id, x_center, y_center, width, height]
                gt_boxes[:N, 0] = torch.clamp(classes, 0.0, 1.0)
                gt_boxes[:N, 1] = f_center
                gt_boxes[:N, 2] = self.t_center
                gt_boxes[:N, 3] = torch.clamp(f_width, 0.0, 1.0)
                gt_boxes[:N, 4] = torch.clamp(self.t_width, 0.0, 1.0)
                if DEBUG:
                    print(
                        f"[\033[36mDebug\033[0m] Generated {N} boxes, format [class_id, x_center, y_center, width, height]:\n{gt_boxes}")
            return noisy_spec, clean_spec, gt_boxes
        elif self.mode == 'detection':
            N, classes, f_starts, f_stops = freq_info if freq_info else (0, [], [])
            gt_boxes = torch.full((self.max_num_signals, 3), float('nan'), dtype=torch.float32)
            if N > 0:
                starts_norm = torch.tensor(f_starts, dtype=torch.float64) / (self.fchans - 1)
                stops_norm = torch.tensor(f_stops, dtype=torch.float64) / (self.fchans - 1)
                gt_boxes[:N, 0] = torch.clamp(starts_norm, 0.0, 1.0)
                gt_boxes[:N, 1] = torch.clamp(stops_norm, 0.0, 1.0)
                classes = torch.tensor(classes, dtype=torch.float32)
                gt_boxes[:N, 2] = torch.clamp(classes, 0.0, 1.0)
            return noisy_spec, clean_spec, gt_boxes
        elif self.mode == 'mask':
            return noisy_spec, clean_spec, rfi_mask, phy_prob
        else:
            return noisy_spec, clean_spec, rfi_mask, freq_info, phy_prob


def split_waterfall_generator(waterfall_fn, fchans, tchans=None, f_shift=None):
    """
    Generator that yields smaller Waterfall objects split by frequency.

    Parameters
    ----------
    waterfall_fn : str or list of str
        Single filterbank filename or list of filenames.
    fchans : int
        Number of frequency channels per split.
    tchans : int, optional
        Number of time samples to keep (default = all).
    f_shift : int or (int,int), optional
        If int -> fixed shift. If tuple/list -> random shift in [low, high].
        Default = fchans (no overlap).
    """
    if isinstance(waterfall_fn, str):
        waterfall_fn = [waterfall_fn]

    while True:
        for fn in waterfall_fn:
            info = Waterfall(fn, load_data=False)
            fch1 = info.header['fch1']
            nchans = info.header['nchans']
            df = info.header['foff']
            total_t = info.container.selection_shape[0]

            if tchans is None:
                t_keep = total_t
            elif tchans > total_t:
                print(
                    f"[\033[33mWarn\033[0m] tchans ({tchans}) larger than observation length ({total_t}) for file: {fn}, skipping")
                continue
            else:
                t_keep = tchans

            # 初始窗口
            f_start, f_stop = fch1, fch1 + fchans * df

            # 遍历直到剩余不够 fchans
            while np.abs(f_stop - fch1) <= np.abs(nchans * df):
                fmin, fmax = np.sort([f_start, f_stop])
                wf = Waterfall(fn, f_start=fmin, f_stop=fmax,
                               t_start=0, t_stop=t_keep)
                yield wf

                # 计算 shift
                if f_shift is None:
                    step = fchans
                elif isinstance(f_shift, (tuple, list)) and len(f_shift) == 2:
                    step = random.randint(f_shift[0], f_shift[1])
                else:
                    step = int(f_shift)

                f_start += step * df
                f_stop += step * df


def plot_samples(dataset, kind='clean', num=10, out_dir=None, with_spectrum=False, spectrum_type='mean'):
    """
    Plot and save specific type of spectrograms from a dynamic dataset.

    Parameters:
    - dataset: instance of DynamicSpectrumDataset or similar
    - kind: 'clean' | 'noisy' | 'mask'
    - num: number of samples to plot
    - out_dir: output directory to save images (default depends on kind)
    - with_spectrum: whether to also plot frequency spectrum
    - spectrum_type: 'mean' | 'middle' | 'peak' | 'fft2d'
    """
    assert kind in ['clean', 'noisy', 'mask'], f"Invalid kind: {kind}"
    assert dataset.mode == 'test', "Dataset mode must be 'test' for plotting"
    assert spectrum_type in ['mean', 'middle', 'peak', 'fft2d'], f"Invalid spectrum type: {kind}"

    if out_dir is None:
        out_dir = {
            'clean': '../plot/sim',
            'noisy': '../plot/no',
            'mask': '../plot/rfi'
        }[kind]
    os.makedirs(out_dir, exist_ok=True)

    for i in range(num):
        if i >= len(dataset):
            break
        sample = dataset[i]

        if isinstance(sample, (list, tuple)):
            noisy_spec, clean_spec, rfi_mask, freq_info, _ = sample
        else:
            raise TypeError("Dataset must return a tuple (noisy, clean, mask, freq_info)")

        if kind == 'clean':
            spec = clean_spec.squeeze()
        elif kind == 'noisy':
            spec = noisy_spec.squeeze()
        elif kind == 'mask':
            if with_spectrum:
                print("[\033[33mWarn\033[0m] Cannot plot frequency spectrum with mask, ignoring.")
                with_spectrum = False
            spec = rfi_mask.squeeze().float()

        # 计算频率轴
        fch1 = dataset.fch1
        df = dataset.df
        fchans = dataset.fchans
        if dataset.ascending:
            freqs = fch1 + np.arange(fchans) * df
        else:
            freqs = fch1 - np.arange(fchans) * df

        if with_spectrum:
            if spectrum_type == "fft2d":
                fig, axs = plt.subplots(2, 1, figsize=(15, 6), sharex=False)
            else:
                fig, axs = plt.subplots(2, 1, figsize=(15, 6), sharex=True)

            if spectrum_type == "fft2d":
                # 原始动态频谱
                im0 = axs[0].imshow(spec, aspect='auto', origin='lower', cmap='viridis',
                                    extent=[freqs[0], freqs[-1], 0, dataset.tchans])
                axs[0].set_title(f"{kind} spectrogram #{i}")
                fig.colorbar(im0, ax=axs[0])

                if kind in ['noisy', 'clean']:
                    plot_F_lines(axs[0], freqs, freq_info, normalized=False)

                # 2D FFT 幅度谱
                fft2d = np.fft.fftshift(np.fft.fft2(spec))
                fft_mag = np.log1p(np.abs(fft2d))  # log 缩放便于可视化

                im1 = axs[1].imshow(fft_mag, aspect='auto', origin='lower', cmap='inferno')
                axs[1].set_title("2D FFT magnitude spectrum")
                fig.colorbar(im1, ax=axs[1])

            else:
                # 原始动态频谱
                axs[0].imshow(spec, aspect='auto', origin='lower', cmap='viridis',
                              extent=[freqs[0], freqs[-1], 0, dataset.tchans])
                axs[0].set_title(f"{kind} spectrogram #{i}")

                if kind in ['noisy', 'clean']:
                    plot_F_lines(axs[0], freqs, freq_info, normalized=False)

                # 1D 频谱
                if spectrum_type == "mean":
                    spectrum = spec.mean(axis=0)
                elif spectrum_type == "middle":
                    T = dataset.tchans // 2
                    spectrum = spec[T, :]
                elif spectrum_type == "peak":
                    spectrum = spec.max(axis=0)
                else:
                    raise ValueError(f"Invalid spectrum_type: {spectrum_type}")

                axs[1].plot(freqs, spectrum, color='blue')
                axs[1].set_xlabel("Frequency")
                axs[1].set_ylabel("Power")
                axs[1].set_title(f"Spectrum ({spectrum_type})")

        else:
            fig, ax = plt.subplots(figsize=(15, 3))
            im = ax.imshow(spec, aspect='auto', origin='lower', cmap='viridis',
                           extent=[freqs[0], freqs[-1], 0, dataset.tchans])
            ax.set_title(f"{kind} spectrogram #{i}")
            fig.colorbar(im, ax=ax)

            if kind in ['noisy', 'clean']:
                plot_F_lines(ax, freqs, freq_info, normalized=False)

        plt.tight_layout()
        save_path = os.path.join(out_dir, f"{kind}_{i:03d}.png")
        plt.savefig(save_path)
        print(f"Saved to {save_path}")
        plt.close()


if __name__ == "__main__":
    tchans = 116
    fchans = 1024
    df = 7.450580597
    dt = 10.200547328
    drift_min = -4.0
    drift_max = 4.0
    drift_min_abs = df // (tchans * dt)
    dataset = DynamicSpectrumDataset(mode='test', tchans=tchans, fchans=fchans, df=df, dt=dt, fch1=None, ascending=True,
                                     drift_min=drift_min, drift_max=drift_max, drift_min_abs=drift_min_abs,
                                     snr_min=25.0, snr_max=45.0, width_min=7.5, width_max=30, num_signals=(1, 1),
                                     noise_std_min=0.025, noise_std_max=0.05, noise_mean_min=2, noise_mean_max=3,
                                     noise_type='chi2', use_fil=True,
                                     background_fil="../data/33exoplanets/yy/Kepler-438_M01_pol2_f1139.58-1142.31.fil")

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

    plot_samples(dataset, kind='clean', num=100, with_spectrum=True, spectrum_type='mean')
    # plot_samples(dataset, kind='noisy', num=30, with_spectrum=True, spectrum_type='fft2d')
    # plot_samples(dataset, kind='mask', num=30, with_spectrum=False)
