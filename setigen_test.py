import numpy as np
import setigen as stg
from astropy import units as u
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------- 可调参数 ----------------
fchans = 224           # 频率通道数
tchans = 224             # 时间通道数
df = 2.7939677238464355 * u.Hz   # 频率分辨率 (Hz)
dt = 18.253611008 * u.s         # 时间分辨率 (s)
fch1 = 6095.214842353016 * u.MHz  # 首通道频率 (MHz)，若 ascending=False 则为最高频
ascending = True     # 频率是否升序排列

noise_x_mean = 1     # 添加噪声的均值
noise_type = 'chi2'     # 噪声类型，可选 'chi2' 或 'gaussian'

# 信号漂移和起始位置
drift_rate = 2 * u.Hz / u.s       # 漂移速率 (Hz/s)
f_start_idx = 1                 # 信号起始的频率通道索引
f_start_freq = None               # 可直接指定起始频率（MHz），若为 None 则使用索引计算

# 强度轮廓参数
t_profile_level = 1     # 时间强度剖面的基准水平
f_profile_width = 20 * u.Hz  # 频谱剖面宽度 (Hz)
bp_profile_level = 1    # 带通强度剖面（常数）

# 是否绘图测试
plot = True
plot_filename = "seti_test"  # 若不为空则保存图像

# ------------------------------------------

# 创建 Frame 对象（初始化数据全为 0）
frame = stg.Frame(fchans=fchans,
                  tchans=tchans,
                  df=df,
                  dt=dt,
                  fch1=fch1,
                  ascending=ascending)

# 在 Frame 中添加噪声
noise = frame.add_noise(x_mean=noise_x_mean, noise_type=noise_type)  # 返回噪声数组
clean_spec = np.copy(frame.data)  # 干净的频谱（只有噪声，无信号）

# 确定信号起始频率
if f_start_freq is not None:
    f_start = f_start_freq * u.MHz
else:
    f_start = frame.get_frequency(index=f_start_idx)  # 由通道索引计算起始频率

# 添加合成信号（例如恒定漂移率信号）
signal = frame.add_signal(
    stg.constant_path(f_start=f_start, drift_rate=drift_rate),        # 漂移路径（常速漂移）
    stg.constant_t_profile(level=t_profile_level),                    # 时间强度剖面（恒定强度）
    stg.box_f_profile(width=f_profile_width),                         # 频谱剖面（矩形宽度）
    stg.constant_bp_profile(level=bp_profile_level)                   # 带通剖面（恒定）
)
# 截至此时 frame.data 包含噪声 + 信号

noisy_spec = frame.data  # 含信号的频谱
rfi_mask = (signal != 0)  # 简单地将信号数组非零值视为RFI存在掩膜

# 频率和时间轴
freq_axis = frame.fs / 1e6  # 将频率数组转换为 MHz
time_frames = frame.tchans

# 绘图测试（仅当 plot=True 时执行）
if plot:
    plt.figure(figsize=(7, 15))

    # 干净频谱
    plt.subplot(311)
    plt.imshow(clean_spec, aspect='auto', origin='lower',
               extent=[freq_axis[0], freq_axis[-1], 0, time_frames],
               cmap='viridis')
    plt.title('Background Noise')
    plt.ylabel('Time Frame')
    plt.colorbar(label='Intensity')

    # 含噪频谱（含信号/RFI）
    plt.subplot(312)
    plt.imshow(noisy_spec, aspect='auto', origin='lower',
               extent=[freq_axis[0], freq_axis[-1], 0, time_frames],
               cmap='viridis')
    plt.title('SETI Signal')
    plt.ylabel('Time Frame')
    plt.colorbar(label='Intensity')

    # RFI掩膜
    plt.subplot(313)
    plt.imshow(rfi_mask, aspect='auto', origin='lower',
               extent=[freq_axis[0], freq_axis[-1], 0, time_frames],
               cmap='Reds')
    plt.title('SETI Signal Mask')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Time Frame')
    plt.colorbar(label='RFI Presence')

    plt.tight_layout()

    if plot_filename:
        plot_o_path = Path('plot/tmp/') / Path(plot_filename).with_suffix('.png')
        plt.savefig(plot_o_path, dpi=480, bbox_inches='tight')
        print(f"Plot saved to {plot_o_path}")

    plt.show()