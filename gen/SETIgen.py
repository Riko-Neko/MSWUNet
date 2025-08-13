from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import setigen as stg
from astropy import units as u

try:
    matplotlib.use('MacOSX')
except:
    pass
import matplotlib.pyplot as plt
from gen.FRIgen import add_rfi


def sim_dynamic_spec_seti(fchans, tchans, df, dt, fch1=None, ascending=False, signals=None, noise_x_mean=0.0,
                          noise_x_std=1.0, noise_type='normal', rfi_params=None, seed=None, plot=False,
                          plot_filename=None, background_fil=None):
    """
        使用 SetiGen 库合成动态频谱并注入射频干扰（RFI）。

        参数:
            fchans (int): 频率通道数量（Y轴维度）
            tchans (int): 时间通道数量（X轴维度）
            df (float or astropy.Quantity): 频率分辨率（单位Hz）
            dt (float or astropy.Quantity): 时间分辨率（单位秒）
            fch1 (float or astropy.Quantity, optional): 第一个频率通道的中心频率（单位Hz）。默认为1.42GHz（氢线）
            ascending (bool): 频率轴是否升序排列。False表示频率递减（天文常用），True表示递增

            signals (list of dict, optional): 注入信号参数列表，每个字典定义一种信号，包含以下键：
                - 'f_index'或'f_start': [必需] 起始位置。整数表示频率通道索引，浮点数表示绝对频率(Hz)
                - 'drift_rate': [必需] 频率漂移率(Hz/s)，正数表示向高频漂移
                - 'snr'或'level': [必需] 信号强度。'snr'表示信噪比，'level'表示绝对强度值
                - 'width': [必需] 信号频谱宽度(Hz)
                - 'path': 频率漂移路径类型。可选值：
                    * 'constant' - 恒定漂移（默认）
                    * 'sine' - 正弦振荡漂移（需额外参数：period, amplitude）
                    * 'squared' - 抛物线漂移
                    * 'rfi' - RFI特征漂移（需额外参数：spread, spread_type, rfi_type）
                - 't_profile': 时间方向强度调制类型。可选值：
                    * 'constant' - 恒定强度（默认）
                    * 'sine' - 正弦调制（需额外参数：t_period, t_amplitude）
                    * 'pulse' - 周期性脉冲（需额外参数：pulse_width, t_period, pnum, min_level）
                - 'f_profile': 频谱轮廓类型。可选值：
                    * 'gaussian' - 高斯轮廓（默认）
                    * 'box' - 矩形轮廓
                    * 'sinc' - sinc函数轮廓
                    * 'lorentzian' - 洛伦兹轮廓
                    * 'voigt' - 沃伊特轮廓（需额外参数：g_width, l_width）
                # 路径类型额外参数
                - 'period': (sine路径) 振荡周期(秒)
                - 'amplitude': (sine路径) 振荡幅度(Hz)
                - 'spread': (rfi路径) 频率扩散范围(Hz)
                - 'spread_type': (rfi路径) 扩散类型（'uniform'均匀分布, 'normal'正态分布）
                - 'rfi_type': (rfi路径) RFI类型（'stationary'静止, 'drifting'漂移）
                # 时间调制额外参数
                - 't_period': (sine/pulse) 调制周期(秒)
                - 't_amplitude': (sine) 调制幅度(强度单位)
                - 'pulse_width': (pulse) 脉冲宽度(时间通道数)
                - 'pnum': (pulse) 脉冲数量
                - 'min_level': (pulse) 脉冲间最小强度
                # 频谱轮廓额外参数
                - 'g_width': (voigt) 高斯分量宽度(Hz)
                - 'l_width': (voigt) 洛伦兹分量宽度(Hz)

            noise_x_mean (float): 背景噪声均值（默认0.0）
            noise_x_std (float): 背景噪声标准差（默认1.0）
            noise_type (str): 噪声统计分布类型。可选值：
                * 'normal' - 高斯分布（默认）
                * 'chi2' - 卡方分布（忽略noise_x_std）

            rfi_params (dict, optional): RFI注入参数字典，包含以下键：
                - 'NBC': 窄带连续RFI数量
                - 'NBC_amp': 窄带连续RFI幅度倍数
                - 'NBT': 窄带瞬态RFI数量
                - 'NBT_amp': 窄带瞬态RFI幅度倍数
                - 'BBT': 宽带瞬态RFI数量
                - 'BBT_amp': 宽带瞬态RFI幅度倍数
                - 'LowDrift': 低漂移率RFI数量
                - 'LowDrift_amp': 低漂移率RFI幅度倍数
                - 'LowDrift_width': 低漂移率RFI宽度(Hz)
                - 'NegBand': 负高斯轮廓数量
                - 'NegBand_amp': 负高斯轮廓幅度倍数
                - 'NegBand_width': 负高斯轮廓宽度(Hz)

            seed (int, optional): 随机数生成器种子（保证结果可重现）
            plot (bool): 是否生成可视化图像（默认False）
            plot_filename (str, optional): 图像保存路径（若提供则保存）

        返回:
            signal_spec (numpy.ndarray): 含信号和噪声的动态频谱，形状为(tchans, fchans)
            clean_spec (numpy.ndarray): 纯净动态频谱（不含RFI和噪声），形状为(tchans, fchans)
            noisy_spec (numpy.ndarray): 含RFI和噪声的动态频谱，形状为(tchans, fchans)
            rfi_mask (numpy.ndarray): RFI位置布尔掩码，形状为(tchans, fchans)

        示例信号配置:
            signals = [
                {
                    'f_start': 1420.3e6,  # 1420.3 MHz
                    'drift_rate': -0.5,    # 漂移率 -0.5 Hz/s
                    'snr': 20,             # 信噪比20
                    'width': 5.0,          # 5Hz带宽
                    'path': 'sine',
                    'period': 120,         # 120秒周期
                    'amplitude': 50e3,     # 50kHz振荡幅度
                    't_profile': 'pulse',
                    'pulse_width': 8,
                    'f_profile': 'voigt',
                    'g_width': 3.0,
                    'l_width': 2.0
                }
            ]
    """
    # 固定随机种子
    if seed is not None:
        np.random.seed(seed)
    # 转换单位
    if not isinstance(df, u.Quantity):
        df = df * u.Hz
    if not isinstance(dt, u.Quantity):
        dt = dt * u.s
    if fch1 is None:
        fch1 = 1.42e9 * u.Hz  # 默认 1420 MHz
        ascending = False
    elif not isinstance(fch1, u.Quantity):
        fch1 = fch1 * u.Hz

    use_fil = False
    if background_fil and np.random.random() < 0.0:
        waterfall_itr = stg.split_waterfall_generator(background_fil, fchans, tchans=tchans, f_shift=None)
        waterfall = next(waterfall_itr)
        # 创建 Frame (使用背景噪声)
        frame = stg.Frame(waterfall)
        clean_spec = np.zeros_like(frame.data)
        use_fil = True

    if not use_fil:
        # 创建 Frame
        frame = stg.Frame(fchans=fchans, tchans=tchans, df=df, dt=dt, fch1=fch1, ascending=ascending)
        clean_spec = np.zeros_like(frame.data)

        # 添加背景噪声
        if noise_type in ['normal', 'gaussian']:
            frame.add_noise(x_mean=noise_x_mean, x_std=noise_x_std, noise_type='normal')
        else:
            frame.add_noise(x_mean=noise_x_mean, noise_type='chi2')

    # 注入信号
    if signals:
        for sig in signals:
            # 计算起始频率
            if 'f_index' in sig:
                f_start = frame.get_frequency(sig['f_index'])
            elif 'f_start' in sig:
                fs = sig['f_start']
                f_start = fs * u.Hz if not isinstance(fs, u.Quantity) else fs
            else:
                f_start = frame.get_frequency(fchans // 2)
            # 漂移率
            drift = sig.get('drift_rate', 0.0)
            drift = (drift * u.Hz / u.s) if not isinstance(drift, u.Quantity) else drift
            # 强度
            if 'snr' in sig:
                level = frame.get_intensity(sig['snr'])
            elif 'level' in sig:
                level = sig['level']
            else:
                level = frame.get_intensity(10)
            # 宽度
            width = sig.get('width', 1.0)
            width = width * u.Hz if not isinstance(width, u.Quantity) else width
            # 选择路径
            path_type = sig.get('path', 'constant')
            if path_type == 'constant':
                path = stg.constant_path(f_start=f_start, drift_rate=drift)
            elif path_type == 'sine':
                period = sig.get('period', tchans / 2) * u.s
                amplitude = sig.get('amplitude', 0.0) * u.Hz
                path = stg.sine_path(f_start=f_start, drift_rate=drift, period=period, amplitude=amplitude)
            elif path_type == 'squared':
                squared_drift = sig.get('squared_drift', drift)
                path = stg.squared_path(f_start=f_start, drift_rate=squared_drift)
            elif path_type == 'rfi':
                spread = sig.get('spread', 0.0) * u.Hz
                spread_type = sig.get('spread_type', 'uniform')
                rfi_type = sig.get('rfi_type', 'stationary')
                path = stg.simple_rfi_path(f_start=f_start, drift_rate=drift,
                                           spread=spread, spread_type=spread_type, rfi_type=rfi_type)
            else:
                path = stg.constant_path(f_start=f_start, drift_rate=drift)
            # 时间调制
            t_type = sig.get('t_profile', 'constant')
            if t_type == 'constant':
                t_profile = stg.constant_t_profile(level=level)
            elif t_type == 'sine':
                t_period = sig.get('t_period', tchans / 2) * u.s
                t_amplitude = sig.get('t_amplitude', level)
                t_profile = stg.sine_t_profile(period=t_period, amplitude=t_amplitude, level=level)
            elif t_type == 'pulse':
                pulse_width = sig.get('pulse_width', tchans / 10)
                p_period = sig.get('t_period', tchans / 5)
                p_amplitude = sig.get('t_amplitude', 1.0)
                pnum = sig.get('pnum', 3)
                min_level = sig.get('min_level', 0.0)
                t_profile = stg.periodic_gaussian_t_profile(pulse_width=pulse_width, period=p_period,
                                                            amplitude=p_amplitude, level=level, min_level=min_level)
            else:
                t_profile = stg.constant_t_profile(level=level)
            # 频谱轮廓
            f_type = sig.get('f_profile', 'gaussian')
            if f_type == 'gaussian':
                f_profile = stg.gaussian_f_profile(width=width)
            elif f_type == 'box':
                f_profile = stg.box_f_profile(width=width)
            elif f_type == 'sinc':
                f_profile = stg.sinc2_f_profile(width=width)
            elif f_type == 'lorentzian':
                f_profile = stg.lorentzian_f_profile(width=width)
            elif f_type == 'voigt':
                g_width = sig.get('g_width', width / 2)
                l_width = sig.get('l_width', width / 2)
                f_profile = stg.voigt_f_profile(g_width=g_width, l_width=l_width)
            else:
                f_profile = stg.gaussian_f_profile(width=width)
            bp_profile = stg.constant_bp_profile(level=1.0)
            clean_spec += frame.add_signal(path, t_profile, f_profile, bp_profile)

    # 初始化 RFI 掩码
    rfi_mask = np.zeros_like(frame.data, dtype=bool)

    # 添加低漂移率 RFI（使用 setigen，constant 类型）
    if rfi_params:
        for _ in range(rfi_params.get('LowDrift', 0)):
            f_index = np.random.randint(0, fchans)
            f_start = frame.get_frequency(f_index)
            drift_rate = np.random.uniform(-0.0001, 0.0001) * u.Hz / u.s
            path = stg.constant_path(f_start=f_start, drift_rate=drift_rate)
            level = rfi_params.get('LowDrift_amp', 5.0) * noise_x_std
            t_profile = stg.constant_t_profile(level=level)
            width = rfi_params.get('LowDrift_width', 2.0) * u.Hz
            f_profile = stg.gaussian_f_profile(width=width)
            bp_profile = stg.constant_bp_profile(1.0)
            added_signal = frame.add_signal(path, t_profile, f_profile, bp_profile)
            rfi_mask |= (added_signal > 0.1 * level)

        # 添加负噪声带（负高斯轮廓）
        for _ in range(rfi_params.get('NegBand', 0)):
            f_index = np.random.randint(0, fchans)
            f_start = frame.get_frequency(f_index)
            drift_rate = 0 * u.Hz / u.s
            path = stg.constant_path(f_start=f_start, drift_rate=drift_rate)
            amp = rfi_params.get('NegBand_amp', 1)
            level = -amp * noise_x_std
            t_profile = stg.constant_t_profile(level=level)
            width_hz = rfi_params.get('NegBand_width', 0.5e3)
            width = min(width_hz, fchans * df.to(u.Hz).value / 2) * u.Hz  # 限制不超过总带宽的一半
            f_profile = stg.gaussian_f_profile(width=width)
            bp_profile = stg.constant_bp_profile(1.0)
            added_signal = frame.add_signal(path, t_profile, f_profile, bp_profile)

    signal_spec = frame.get_data(db=False)

    # 注入传统 RFI
    if rfi_params and np.random.random() < 0.5:
        noisy_spec, traditional_rfi_mask = add_rfi(signal_spec, rfi_params)
        rfi_mask |= traditional_rfi_mask
    else:
        noisy_spec = signal_spec.copy()

    # 可视化（可选）
    if plot:
        plt.figure(figsize=(36, 36))
        if ascending:
            freqs = fch1.to(u.Hz).value + np.arange(fchans) * df.to(u.Hz).value
        else:
            freqs = fch1.to(u.Hz).value - np.arange(fchans) * df.to(u.Hz).value
        plt.subplot(221)
        plt.imshow(clean_spec, aspect='auto', origin='lower',
                   extent=[freqs[0], freqs[-1], 0, tchans], cmap='viridis')
        plt.title('Clean Signal Spectrum')
        plt.colorbar(label='Intensity')
        plt.subplot(222)
        plt.imshow(signal_spec, aspect='auto', origin='lower',
                   extent=[freqs[0], freqs[-1], 0, tchans], cmap='viridis')
        plt.title('Signal Spectrum with Noise')
        plt.colorbar(label='Intensity')
        plt.subplot(223)
        plt.imshow(noisy_spec, aspect='auto', origin='lower',
                   extent=[freqs[0], freqs[-1], 0, tchans], cmap='viridis')
        plt.title('Noisy Spectrum with RFI')
        plt.colorbar(label='Intensity')
        plt.subplot(224)
        plt.imshow(rfi_mask, aspect='auto', origin='lower',
                   extent=[freqs[0], freqs[-1], 0, tchans], cmap='Reds')
        plt.title('RFI Mask')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Time')
        plt.colorbar(label='RFI presence')
        plt.tight_layout()
        if plot_filename:
            out_path = Path(plot_filename)
            plt.savefig(out_path, dpi=480)
            print(f"Plot saved to {out_path}")
    return signal_spec, clean_spec, noisy_spec, rfi_mask


if __name__ == "__main__":
    import os

    out_dir = "../plot/sim_raw/"
    os.makedirs(out_dir, exist_ok=True)

    signals = [{
        'f_index': 1024,
        'drift_rate': 0.1,  # Hz/s
        'snr': 10,
        'width': 20,  # Hz
        'path': 'squared',
        'period': 10,
        'amplitude': 50,
        't_profile': 'constant',
        'f_profile': 'gaussian'
    }, {
        'f_index': 2048,
        'drift_rate': -1,  # Hz/s
        'snr': 10,
        'width': 20,  # Hz
        'path': 'sine',
        'period': 10,
        'amplitude': 50,
        't_profile': 'constant',
        'f_profile': 'gaussian'
    }, {
        'f_index': 3072,
        'drift_rate': 2,  # Hz/s
        'snr': 10,
        'width': 20,  # Hz
        'path': 'constant',
        'period': 10,
        'amplitude': 50,
        't_profile': 'constant',
        'f_profile': 'gaussian'
    }]

    # RFI配置
    rfi_conf = {
        'NBC': np.random.randint(5, 15),  # 窄带连续RFI
        'NBC_amp': np.random.uniform(1, 5),
        'NBT': np.random.randint(10, 30),  # 窄带瞬态RFI
        'NBT_amp': np.random.uniform(1, 5),
        'BBT': np.random.randint(5, 15),  # 宽带瞬态RFI
        'BBT_amp': np.random.uniform(1, 5),
        'LowDrift': np.random.randint(1, 6),
        'LowDrift_amp': np.random.uniform(1.25, 5),
        'LowDrift_width': np.random.uniform(7.5, 15),
        'NegBand': np.random.randint(1, 2),
        'NegBand_amp': np.random.uniform(0.25, 0.75),
        'NegBand_width': np.random.uniform(0.3e6, 0.7e6)
    }

    sim_dynamic_spec_seti(
        fchans=4096,
        tchans=512,
        df=7.5,
        dt=1.0,
        fch1=1.42e9,
        ascending=True,
        signals=signals,
        noise_x_mean=0.0,
        noise_x_std=0.1,
        noise_type='normal',
        rfi_params=rfi_conf,
        seed=42,
        plot=True,
        plot_filename=f"{out_dir}/sim_raw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    )
