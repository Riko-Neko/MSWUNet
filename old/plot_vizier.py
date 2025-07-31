import os

import matplotlib.pyplot as plt
import numpy as np

from dataset import load_alfalfa_spectrum


def plot_alfalfa_spectra(fits_dir = './fits', num_plots = None, freqs_target = None):
    """
    读取FITS频谱文件并绘制图像
    参数:
        fits_dir: FITS文件目录，默认为 './fits'
        num_plots: 绘制图片数量，默认为 None（绘制所有）
        freqs_target: 可选的目标频率网格，用于重采样
    """
    # 获取FITS文件列表
    fits_files = [f for f in os.listdir(fits_dir) if f.endswith('.fits')]

    # 根据num_plots控制绘制数量
    if num_plots is not None:
        fits_files = fits_files[:num_plots]

    # 确保输出目录存在
    output_dir = './plot_alfalfa_spectra'
    os.makedirs(output_dir, exist_ok = True)

    for fits_file in fits_files:
        file_path = os.path.join(fits_dir, fits_file)
        data = load_alfalfa_spectrum(file_path, freqs_target)

        if data is None:
            continue

        # 根据是否提供freqs_target处理数据
        if freqs_target is not None:
            flux = data
            freqs = freqs_target
        else:
            freqs, flux = data

        # 绘制频谱图
        plt.figure()
        plt.plot(freqs, flux)
        plt.xlabel('频率 (MHz)')
        plt.ylabel('流量')
        plt.title(f'光谱: {fits_file}')

        # 保存图像，文件名与FITS文件相同（扩展名为.png）
        plot_name = os.path.splitext(fits_file)[0] + '.png'
        plot_path = os.path.join(output_dir, plot_name)
        plt.savefig(plot_path)
        plt.close()

        print(f"已保存图像: {plot_path}")


# 示例调用
if __name__ == "__main__":
    freqs = np.linspace(1419.205, 1421.605, 1024)
    # plot_alfalfa_spectra(num_plots = 5, freqs_target = freqs)
    plot_alfalfa_spectra(num_plots = 5)
