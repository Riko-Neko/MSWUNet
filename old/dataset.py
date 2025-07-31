import os
import queue
import threading

import torch
from astropy.io import fits
from torch.utils.data import Dataset, IterableDataset
from tqdm import tqdm


def generate_carrier_wave(freqs, magnitude):
    """
    生成频域中的载波信号包络，形状随机选择，并添加随机斜线增益。

    参数：
    - freqs: 频率数组，例如 np.linspace(1419.205, 1421.605, 256)
    - noise_level: 噪声水平

    返回：
    - wave: 信号包络，形状与 freqs 一致（256个采样点）
    """
    wave_types = ['flat', 'window', 'double_peak', 'triple_peak']
    probabilities = [0.1, 0.3, 0.3, 0.3]

    wave_type = np.random.choice(wave_types, p = probabilities)

    base_signal = np.ones_like(freqs)

    if wave_type == 'flat':
        wave = base_signal * magnitude
        curvature = np.random.uniform(-0.02, 0.02)  # 随机曲率
        x = np.linspace(-1, 1, len(freqs))
        wave *= (1 + curvature * x ** 2)
        wave = np.clip(wave, 0, None)
    elif wave_type == 'window':
        center = (freqs[0] + freqs[-1]) / 2
        width = freqs[-1] - freqs[0]
        # 控制矩形窗的宽度比例和边缘平滑度
        rect_width_ratio = np.random.uniform(0.7, 0.9)  # 矩形主瓣占总频宽的比例
        transition_ratio = np.random.uniform(0.03, 0.08)  # 平滑边缘宽度比例
        rect_width = width * rect_width_ratio
        transition_width = width * transition_ratio
        left_edge = center - rect_width / 2
        right_edge = center + rect_width / 2
        # 使用双sigmoid近似矩形窗：值在中间为1，两边缓慢下降到0
        sigmoid_left = 1 / (1 + np.exp(-(freqs - left_edge) / transition_width))
        sigmoid_right = 1 / (1 + np.exp((freqs - right_edge) / transition_width))
        rect_window = sigmoid_left * sigmoid_right
        wave = base_signal * rect_window * magnitude
    elif wave_type == 'double_peak':
        center = (freqs[0] + freqs[-1]) / 2
        sep = (freqs[-1] - freqs[0]) * np.random.uniform(0.24, 0.36)
        left = center - sep / 2
        right = center + sep / 2
        sigma = 0.2 * (freqs[-1] - freqs[0])
        peak1 = np.exp(-((freqs - left) / sigma) ** 2)
        peak2 = np.exp(-((freqs - right) / sigma) ** 2)
        wave = base_signal * (peak1 + peak2) * magnitude
    elif wave_type == 'triple_peak':
        center = (freqs[0] + freqs[-1]) / 2
        sep = (freqs[-1] - freqs[0]) * np.random.uniform(0.18, 0.28)
        left = center - sep
        right = center + sep
        sigma = 0.15 * (freqs[-1] - freqs[0])
        peak1 = np.exp(-((freqs - left) / sigma) ** 2)
        peak2 = np.exp(-((freqs - center) / sigma) ** 2)
        peak3 = np.exp(-((freqs - right) / sigma) ** 2)
        wave = base_signal * (peak1 + peak2 + peak3) * magnitude
    else:
        raise ValueError("不支持的 wave_type")

    # 添加随机斜线增益
    angle = np.random.normal(loc = 0, scale = 5)  # 平均值0，标准差5
    angle = np.clip(angle, -15, 15)  # 保证范围在[-15, 15]
    slope = np.tan(np.deg2rad(angle))
    linear_gain = 1 + slope * (freqs - freqs[0]) / (freqs[-1] - freqs[0])
    wave *= linear_gain

    # 添加噪声
    # noise = np.random.normal(0, np.random.uniform(0, 0.001), len(freqs))
    # wave += noise

    # 确保幅度非负
    wave = np.clip(wave, 0, None)

    return wave


import numpy as np


def generate_astrophysical_signal(freqs, carrier_amplitude, v = None, v_rot = 220, use_rotation_curve = True,
                                  signal_type = None):
    """
    生成天体物理信号，支持指定速度或通过旋转曲线生成

    参数:
    - freqs: 频率数组 (MHz)
    - carrier_amplitude: 载波幅度
    - v: 指定速度 (km/s)，若未提供则随机生成
    - v_rot: 旋转速度 (km/s)，默认为220 km/s
    - use_rotation_curve: 是否使用银河系旋转曲线
    - signal_type: 信号类型，支持 'HI' 和 'HI_absorption'，默认为随机选择

    返回:
    - signal: 生成的信号数组
    - v: 使用的速度 (km/s)
    """
    # 计算频率步长
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 0.001  # 默认步长 0.001 MHz
    empirical_v_scale = 0.5  # 经验速度缩放因子
    """
    基于先验知识对该因子的估计，采用了缩小速度搜索范围的方法。
    虽然理论计算表明速度的取值范围应更为宽广，但实际观测数据中达到极端速度值的概率较低。
    经验表明，适当收紧速度范围不仅能够更准确地模拟观测结果，还能有效稳定信号的匹配过程，
    确保信号峰值位于载波信号的合理增益区间，从而避免信号峰值出现在不合理的陡峭区段，提升整体系统的鲁棒性和匹配精度。
    """

    if v is None:
        if use_rotation_curve:
            l = np.random.uniform(0, 2 * np.pi)
            v = v_rot * np.sin(l)
        else:
            v = np.random.uniform(-150, 150)

    v = v * empirical_v_scale  # 经验速度缩放因子

    # 多普勒频移计算，c = 299792.458 km/s
    doppler_factor = 1 + v / 299792.458
    signal_reverse = False
    if signal_type is None:
        signal_type = np.random.choice(['HI', 'HI_absorption'], p = [0.7, 0.3])
    else:
        if np.random.rand() < 0.01:
            signal_type = 'HI_absorption' if signal_type == 'HI' else 'HI'
            signal_reverse = True

    signal = np.zeros_like(freqs)

    # 速度分散范围（km/s）
    c = 299792.458  # 光速 (km/s)
    f0 = 1420.405751768  # 21cm线静止频率 (MHz)
    if signal_type == 'HI':  # HI发射线
        sigma_v = np.random.uniform(2, 5)  # 速度分散 2-5 km/s
        sigma = (sigma_v / c) * f0  # 转换为频率单位
        sigma = max(sigma, 2 * df)  # 确保至少2倍频率步长
        center = f0 * doppler_factor
        amp = carrier_amplitude * np.random.uniform(0.02, 0.08)
        signal = amp * np.exp(-((freqs - center) / sigma) ** 2)
    elif signal_type == 'HI_absorption':  # HI吸收线
        sigma_v = np.random.uniform(2, 5)  # 速度分散 2-5 km/s
        sigma = (sigma_v / c) * f0
        sigma = max(sigma, 2 * df)
        center = f0 * doppler_factor
        depth = carrier_amplitude * np.random.uniform(0.02, 0.08)
        signal = -depth * np.exp(-((freqs - center) / (0.5 * sigma)) ** 4)

        """
        References:
        Analysis of the Absorption Line Profile at 21 cm for the Hydrogen Atom in Interstellar Medium
        A High Galactic Latitude HI 21cm-line Absorption Survey using the GMRT: I. Observations and Spectra
        temperature of the diffuse H i in the Milky Way - II. Gaussian decomposition of the H i-21 cm absorption spectra
        Key: 采用超高斯分布拟合吸收线可能是合理的选择，
        在 Analysis of the Absorption Line Profile at 21 cm for the Hydrogen Atom in Interstellar Medium 中，
        研究了HI 21cm吸收线受Lyα跃迁的影响，发现线形可能因非多普勒展宽和频率偏移而偏离标准高斯分布。
        超高斯分布可能是一种近似这些复杂效应的方式，尤其是当吸收线的翼部比高斯分布更陡峭时。
        """

    else:
        raise ValueError("Unsupported signal_type")

    if signal_reverse:
        signal_type = 'HI_absorption' if signal_type == 'HI' else 'HI'

    return signal, v, signal_type, center


def add_gaussian_noise(spectrum, noise_level = 0.1):
    """添加高斯噪声"""
    return spectrum + np.random.normal(0, noise_level, spectrum.shape)


def add_rfi(spectrum, freqs, num_rfi = 5, rfi_amplitude = 1.0):
    """添加射频干扰（窄带高斯尖峰）"""
    rfi_freqs = np.random.choice(freqs, num_rfi, replace = False)
    for f in rfi_freqs:
        rfi_amplitude = rfi_amplitude * np.random.uniform(0.1, 1.0)
        spectrum += rfi_amplitude * np.exp(-((freqs - f) / 0.001) ** 2)
    return spectrum


def generate_synthetic_dataset(freqs, num_samples = 10000, v_rot = 220, b = 10, use_rotation_curve = True,
                               verbose = True):
    """
    生成合成数据集，信号速度具有指数关联性

    参数:
    - freqs: 频率数组 (MHz)
    - num_samples: 样本数量
    - v_rot: 旋转速度 (km/s)，默认为220 km/s
    - b: Laplace分布的尺度参数 (km/s)，默认为10 km/s
    - use_rotation_curve: 是否使用旋转曲线生成基速度
    - verbose: 是否显示进度条
    """
    clean_spectra = []
    noisy_spectra = []
    prob_vectors = []
    num_signals_probs = [0.1, 0.635, 0.190, 0.057, 0.018]  # 0到4个信号的概率
    signal_type = None

    iterator = range(num_samples)
    if verbose:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc = "Generating")

    for _ in iterator:
        random_magnitude = np.clip(np.random.normal(loc = 10, scale = 8), 1, 30)
        carrier = generate_carrier_wave(freqs, random_magnitude)
        noise_level = random_magnitude * np.random.uniform(0.001, 0.01)
        rfi_amplitude = random_magnitude * np.random.uniform(0.01, 0.3)
        num_rfi = np.random.randint(1, 5)

        clean = carrier.copy()
        num_signals = np.random.choice([0, 1, 2, 3, 4], p = num_signals_probs)
        centers = []

        if num_signals >= 1:
            if use_rotation_curve:
                l = np.random.uniform(0, 2 * np.pi)
                v_base = v_rot * np.sin(l)
            else:
                v_base = np.random.uniform(-150, 150)

            for _ in range(num_signals):
                delta_v = np.random.laplace(0, b)
                v = v_base + delta_v
                signal, _, signal_type, center = generate_astrophysical_signal(freqs, random_magnitude, v = v,
                                                                               signal_type = signal_type)
                clean += signal
                centers.append(center)

        noisy = add_gaussian_noise(clean, noise_level)
        noisy = add_rfi(noisy, freqs, num_rfi, rfi_amplitude)

        prob_vector = np.zeros_like(freqs)
        for center in centers:
            idx = np.argmin(np.abs(freqs - center))
            prob_vector[idx] = 1.0

        clean_spectra.append(clean)
        noisy_spectra.append(noisy)
        prob_vectors.append(prob_vector)

    return np.array(clean_spectra), np.array(noisy_spectra), np.array(prob_vectors)


def load_alfalfa_spectrum(fits_file, freqs_target = None, verbose = True):
    """
    从ALFALFA FITS文件加载光谱
    如果提供freqs_target，则重采样到目标频率网格
    """
    try:
        with fits.open(fits_file) as hdul:
            if len(hdul) < 2:
                print("未找到光谱HDU")
                return None
            table_hdu = hdul[1]  # 光谱数据在第二个HDU（BinTableHDU）
            data = table_hdu.data
            header = table_hdu.header

            # 获取列名
            column_names = [header.get(f'TTYPE{i + 1}', f'Column{i + 1}') for i in range(len(data.columns))]
            if verbose:
                print("列名:", column_names)

            # 查找频率/速度列（假设列名包含 "VEL" 或 "FREQ"）
            freq_index = next(
                (i for i, name in enumerate(column_names) if 'FREQ' in name.upper() or 'FREQUENCY' in name.upper()), None)
            if freq_index is None:
                print("未找到频率或速度列")
                return None

            # 查找通量列（假设列名包含 "FLUX"）
            flux_index = next((i for i, name in enumerate(column_names) if 'FLUX' in name.upper()), None)
            if flux_index is None:
                print("未找到通量列")
                return None

            freqs = data.field(freq_index)
            flux = data.field(flux_index)

            if freqs_target is not None:
                # 重采样到目标频率网格
                flux_interp = np.interp(freqs_target, freqs, flux, left = 0, right = 0)
                return flux_interp

            flux = np.nan_to_num(flux, nan=0.0)  # 处理NaN值

            # mean_flux = np.nanmean(flux)
            # flux = np.nan_to_num(flux, nan=mean_flux)

            return freqs, flux
    except Exception as e:
        print(f"加载FITS文件 {fits_file} 失败: {e}")
        return None


def generate_alfalfa_dataset(fits_dir, freqs_target, num_samples = 10000):
    """从指定目录加载ALFALFA光谱，生成训练数据集"""
    noise_level = 0.1
    num_rfi = 5
    rfi_amplitude = 1.0
    clean_spectra = []
    noisy_spectra = []
    fits_files = [os.path.join(fits_dir, f) for f in os.listdir(fits_dir) if f.endswith('.fits')]
    if not fits_files:
        raise ValueError(f"目录 {fits_dir} 中未找到FITS文件")
    bar = tqdm(range(num_samples), total = num_samples, desc = 'Loading ALFALFA')
    for _ in bar:
        # 随机选择一个FITS文件
        fits_file = np.random.choice(fits_files)
        flux = load_alfalfa_spectrum(fits_file, freqs_target)
        if flux is None:
            continue
        clean = flux
        noisy = add_gaussian_noise(clean, noise_level)
        noisy = add_rfi(noisy, freqs_target, num_rfi, rfi_amplitude)
        clean_spectra.append(clean)
        noisy_spectra.append(noisy)
        if len(clean_spectra) >= num_samples:
            break
    if len(clean_spectra) < num_samples:
        print(f"警告：仅加载了 {len(clean_spectra)} 个样本，少于请求的 {num_samples}")
    return np.array(clean_spectra), np.array(noisy_spectra)


def generate_dataset(data_type = 'synthetic', freqs = None, num_samples = 10000, fits_dir = None):
    """
    生成训练数据集，支持合成数据或ALFALFA真实数据
    参数：
        data_type: 'synthetic' 或 'alfalfa'
        freqs: 频率网格（MHz）
        num_samples: 样本数
        fits_dir: ALFALFA FITS文件目录（data_type='alfalfa'时需要）
        noise_level: 高斯噪声水平
        num_rfi: RFI尖峰数量
        rfi_amplitude: RFI幅度
    """
    if freqs is None:
        freqs = np.linspace(1419.205, 1421.605, 1024)  # 默认频率网格
    if data_type == 'synthetic':
        return generate_synthetic_dataset(freqs, num_samples)
    elif data_type == 'alfalfa':
        if fits_dir is None:
            raise ValueError("ALFALFA数据集需要指定fits_dir")
        return generate_alfalfa_dataset(fits_dir, freqs, num_samples)
    else:
        raise ValueError("data_type 必须是 'synthetic' 或 'alfalfa'")


class SpectraDataset(Dataset):
    def __init__(self, clean, noisy, prob):
        self.clean = clean
        self.noisy = noisy
        self.prob = prob
        self.clean_mean = None
        self.clean_std = None
        self.noisy_mean = None
        self.noisy_std = None

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, idx):
        noisy = torch.tensor(((self.noisy[idx] - self.noisy_mean) / self.noisy_std), dtype = torch.float32)
        clean = torch.tensor(((self.clean[idx] - self.clean_mean) / self.clean_std), dtype = torch.float32)
        prob = torch.tensor(self.prob[idx], dtype = torch.float32)
        return noisy, clean, prob

    def compute_stats(self, sample_type = 'both', exclude_zero = True):
        """
        计算数据集的统计特征
        参数：
            sample_type: 'clean'/'noisy'/'both' 指定计算哪些统计量
            exclude_zero: 是否排除全零样本（针对clean数据中的无信号情况）
        """
        print("Computing statistics...")

        def _calculate_stats(data, exclude_zero):
            if exclude_zero:
                # 过滤全零样本
                non_zero_mask = ~np.all(np.isclose(data, 0), axis = 1)
                filtered_data = data[non_zero_mask]
                if len(filtered_data) == 0:
                    return 0.0, 1.0  # 防止除零
                return filtered_data.mean(), filtered_data.std()
            else:
                return data.mean(), data.std()

        if sample_type in ['both', 'clean']:
            self.clean_mean, self.clean_std = _calculate_stats(
                self.clean, exclude_zero
            )
        if sample_type in ['both', 'noisy']:
            self.noisy_mean, self.noisy_std = _calculate_stats(
                self.noisy, exclude_zero = False  # 噪声数据通常不需要排除零值
            )

    def get_stats(self):
        """返回统计量字典"""
        return {
            'clean': {'mean': self.clean_mean, 'std': self.clean_std},
            'noisy': {'mean': self.noisy_mean, 'std': self.noisy_std}
        }


class InfiniteSpectraDataset(IterableDataset):
    def __init__(self, freqs, batch_size, stats):
        self.freqs = freqs
        self.batch_size = batch_size
        self.stats = stats
        self.data_queue = queue.Queue(maxsize = 10)
        self.stop_event = threading.Event()
        self.generator_thread = threading.Thread(target = self._generate_data)
        self.generator_thread.start()

    def _generate_data(self):
        while not self.stop_event.is_set():
            clean_spectra, noisy_spectra, prob_vectors = generate_synthetic_dataset(self.freqs, self.batch_size,
                                                                                    verbose = False)
            clean = torch.tensor(((clean_spectra - self.stats['clean']['mean']) / self.stats['clean']['std']),
                                 dtype = torch.float32)
            noisy = torch.tensor(((noisy_spectra - self.stats['noisy']['mean']) / self.stats['noisy']['std']),
                                 dtype = torch.float32)
            self.data_queue.put((noisy, clean, prob_vectors))

    def __iter__(self):
        return self

    def __next__(self):
        if self.stop_event.is_set():
            raise StopIteration
        return self.data_queue.get()

    def stop(self):
        self.stop_event.set()
        self.generator_thread.join()


if __name__ == '__main__':
    import json

    freqs = np.linspace(1419.205, 1421.605, 1024)
    clean, noisy, prob = generate_synthetic_dataset(freqs, num_samples = 10)
    dataset = SpectraDataset(clean, noisy, prob)
    stats_path = 'dataset_stats.json'
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        dataset.clean_mean = stats['clean']['mean']
        dataset.clean_std = stats['clean']['std']
        dataset.noisy_mean = stats['noisy']['mean']
        dataset.noisy_std = stats['noisy']['std']
        print(f"Loaded Clean Mean: {stats['clean']['mean']:.4f} ± {stats['clean']['std']:.4f}")
        print(f"Loaded Noisy Mean: {stats['noisy']['mean']:.4f} ± {stats['noisy']['std']:.4f}")
    else:
        raise ValueError("Dataset stats not found.")
    noisy, clean, prob_target = dataset[0]
    print(noisy.shape)  # torch.Size([1024])
    print(clean.shape)  # torch.Size([1024])
    print(prob_target.shape)  # torch.Size([1024])
    print("Unique values in prob_target:", prob_target.unique())  # Should be [0, 1]
    print("Sample:")
    print(noisy)
    print(clean)
    print(prob_target)
