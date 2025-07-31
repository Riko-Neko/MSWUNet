import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import resample

from alfalfa import load_alfalfa_spectrum  # 确保dataset.py在相同目录


def simulate_HI_dynamic_spectrum(freq_MHz, template_flux,
                                 time_frames=100,
                                 resample_n=128,
                                 noise_std=0.0, noise_dist='gaussian',
                                 width_scale=1.0, drift_rate=0.0,
                                 multi_components=None,
                                 rfi_params=None,
                                 plot=False,
                                 plot_filename=None):
    """
    生成模拟HI动态频谱。

    参数：
    freq_MHz, template_flux : 模板频率轴(MHz)和对应的HI谱线强度 (1D数组)。
    time_frames : 动态频谱帧数（时间方向长度）。
    resample_n: 重采样点数 (None表示不重采样)
    noise_std, noise_dist : 噪声标准差及分布类型（默认高斯白噪声）。
    width_scale : 谱线宽度缩放因子 (>1展宽，<1变窄)。
    drift_rate : 每帧的频率漂移量 (MHz/帧)，正值表示向高频方向漂移。
    multi_components : 多分量信号配置列表，每个分量包含:
        'offset': 频率偏移量(MHz)
        'amplitude': 相对幅度因子
    rfi_params : RFI模拟参数字典，包含各类型RFI数量和强度等配置。
    plot: 是否绘制可视化结果
    plot_filename: 图像保存路径

    返回：
    clean_spec (2D array), noisy_spec (2D array), rfi_mask (2D array)
    """

    if resample_n is not None:
        original_len = len(freq_MHz)

        if resample_n < original_len:
            # 计算原始频率间隔
            original_spacing = np.mean(np.diff(freq_MHz))

            # 重采样（下采样）
            template_flux, freq_MHz = resample(
                template_flux,
                resample_n,
                t=freq_MHz
            )

            # 计算新采样率
            new_spacing = np.mean(np.diff(freq_MHz))

        elif resample_n > original_len:
            pad_len = resample_n - original_len

            if original_len > 1:
                last_spacing = freq_MHz[-1] - freq_MHz[-2]
            else:
                last_spacing = 1.0

            # 创建额外的频率点
            extra_freqs = freq_MHz[-1] + last_spacing * np.arange(1, pad_len + 1)
            freq_MHz = np.concatenate([freq_MHz, extra_freqs])

            # 对通量进行填充
            template_flux = np.pad(template_flux, (0, pad_len), mode='constant')

    # 直接使用输入的频率轴
    freq_axis = np.array(freq_MHz)
    n_chan = freq_axis.size
    base_flux = template_flux.copy()  # 使用原始信号

    # 调整谱线宽度（使用高斯平滑）
    if width_scale != 1.0:
        # 计算平滑的sigma值（基于宽度缩放因子）
        smooth_sigma = max(0.1, abs(width_scale - 1.0) * 5.0)
        if width_scale > 1.0:
            base_flux = gaussian_filter1d(base_flux, smooth_sigma)
        else:
            # 对于变窄的情况，使用反卷积技术（这里简化为锐化）
            smoothed = gaussian_filter1d(base_flux, smooth_sigma)
            base_flux = np.clip(2 * base_flux - smoothed, 0, None)

    # 处理多分量信号（对真实信号进行平移和叠加）
    if multi_components:
        total_flux = np.zeros_like(base_flux)
        for comp in multi_components:
            offset = comp.get('offset', 0.0)  # 频率偏移量(MHz)
            amplitude = comp.get('amplitude', 1.0)  # 幅度缩放因子

            # 创建偏移后的信号（使用线性插值）
            shifted_flux = np.interp(
                freq_axis,
                freq_axis + offset,
                base_flux,
                left=0,
                right=0
            )
            total_flux += amplitude * shifted_flux
        base_flux = total_flux  # 更新基础信号

    # 生成干净动态频谱
    clean_spec = np.zeros((time_frames, n_chan))
    for t in range(time_frames):
        freq_offset = drift_rate * t
        if abs(freq_offset) > 1e-6:
            # 应用频率漂移（插值实现）
            shifted_flux = np.interp(
                freq_axis,
                freq_axis + freq_offset,
                base_flux,
                left=0,
                right=0
            )
            clean_spec[t, :] = shifted_flux
        else:
            clean_spec[t, :] = base_flux

    # 添加噪声
    noisy_spec = clean_spec.copy()
    if noise_std > 0:
        if noise_dist == 'gaussian':
            noise = np.random.normal(0, noise_std, noisy_spec.shape)
        elif noise_dist == 'uniform':
            noise = np.random.uniform(-noise_std, noise_std, noisy_spec.shape)
        else:
            noise = np.zeros_like(noisy_spec)
        noisy_spec += noise

    # 添加RFI并创建掩膜
    rfi_mask = np.zeros_like(noisy_spec, dtype=bool)
    if rfi_params:
        n_chan = noisy_spec.shape[1]

        # 窄带连续RFI
        for _ in range(rfi_params.get('NBC', 0)):
            chan_idx = np.random.randint(0, n_chan)
            rfi_strength = rfi_params.get('NBC_amp', 5.0) * noise_std
            noisy_spec[:, chan_idx] += rfi_strength
            rfi_mask[:, chan_idx] = True

        # 窄带瞬态RFI
        for _ in range(rfi_params.get('NBT', 0)):
            chan_idx = np.random.randint(0, n_chan)
            frame_start = np.random.randint(0, time_frames)
            duration_max = min(5, time_frames - frame_start)
            if duration_max <= 1:  # 无法生成有效持续时间时跳过
                continue
            duration = np.random.randint(1, duration_max)
            rfi_strength = rfi_params.get('NBT_amp', 8.0) * noise_std * np.random.randn()
            noisy_spec[frame_start:frame_start + duration, chan_idx] += rfi_strength
            rfi_mask[frame_start:frame_start + duration, chan_idx] = True

        # 宽带瞬态RFI
        for _ in range(rfi_params.get('BBT', 0)):
            frame_idx = np.random.randint(0, time_frames)
            chan_start = np.random.randint(0, n_chan // 2)
            chan_width = np.random.randint(n_chan // 10, n_chan // 2)
            rfi_strength = rfi_params.get('BBT_amp', 10.0) * noise_std
            affected_chans = slice(chan_start, min(n_chan, chan_start + chan_width))
            noisy_spec[frame_idx, affected_chans] += rfi_strength
            rfi_mask[frame_idx, affected_chans] = True

    # 可视化
    if plot:
        plt.figure(figsize=(14, 30))

        # 干净频谱
        plt.subplot(311)
        plt.imshow(clean_spec, aspect='auto', origin='lower',
                   extent=[freq_axis[0], freq_axis[-1], 0, time_frames],
                   cmap='viridis')
        plt.title('Clean Spectrum')
        plt.ylabel('Time Frame')
        plt.colorbar(label='Intensity')

        # 含噪频谱
        plt.subplot(312)
        plt.imshow(noisy_spec, aspect='auto', origin='lower',
                   extent=[freq_axis[0], freq_axis[-1], 0, time_frames],
                   cmap='viridis')
        plt.title('Noisy Spectrum with RFI')
        plt.ylabel('Time Frame')
        plt.colorbar(label='Intensity')

        # RFI掩膜
        plt.subplot(313)
        plt.imshow(rfi_mask, aspect='auto', origin='lower',
                   extent=[freq_axis[0], freq_axis[-1], 0, time_frames],
                   cmap='Reds')
        plt.title('RFI Mask')
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Time Frame')
        plt.colorbar(label='RFI Presence')

        plt.tight_layout()

        # 保存图像
        if plot_filename:
            plot_o_path = Path('./plot') / Path(plot_filename).with_suffix('.png').name
            plt.savefig(plot_o_path, dpi=480, bbox_inches='tight')
            print(f"Plot saved to {plot_o_path}")

        plt.show()

    return clean_spec, noisy_spec, rfi_mask


if __name__ == '__main__':
    # 示例用法
    fname = "./fits/A331060.fits"
    freq_MHz, flux = load_alfalfa_spectrum(fname)

    # 多分量配置示例
    components = [
        {'offset': 0.0, 'amplitude': 1.0},  # 原始信号
        {'offset': 0.5, 'amplitude': 0.7},  # 偏移0.5MHz，强度70%
        {'offset': -0.3, 'amplitude': 0.4}  # 偏移-0.3MHz，强度40%
    ]

    # RFI配置
    rfi_conf = {
        'NBC': np.random.randint(10, 30),  # 窄带连续RFI
        'NBC_amp': np.random.uniform(20, 30),
        'NBT': np.random.randint(50, 100),  # 窄带瞬态RFI
        'NBT_amp': np.random.uniform(20, 30),
        'BBT': np.random.randint(10, 30),  # 宽带瞬态RFI
        'BBT_amp': np.random.uniform(20, 30)
    }

    # 生成模拟动态频谱
    clean_spec, noisy_spec, rfi_mask = simulate_HI_dynamic_spectrum(
        freq_MHz, flux,
        time_frames=224,
        resample_n=224,
        noise_std=0.15,
        width_scale=1.1,  # 轻微展宽
        drift_rate=np.random.uniform(0., 0.),  # 缓慢漂移
        multi_components=components,
        rfi_params=rfi_conf,
        plot=True,
        plot_filename=fname
    )

    # 可选的保存功能
    # fits.writeto("clean_spec.fits", clean_spec, overwrite=True)
    # fits.writeto("noisy_spec.fits", noisy_spec, overwrite=True)
    # fits.writeto("rfi_mask.fits", rfi_mask.astype(int), overwrite=True)
