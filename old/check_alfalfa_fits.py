import numpy as np
from astropy.io import fits

from alfalfa import load_alfalfa_spectrum


def check_alfalfa_fits(type = '1'):
    if type == '1':
        hdul = fits.open(fitsf)
        print(hdul.info())
        # 访问并打印主 HDU 的头信息
        primary_header = hdul[0].header
        print("主头信息:")
        print(primary_header)
        # 如果有多个 HDU，逐个检查
        if len(hdul) > 1:
            for i, hdu in enumerate(hdul[1:], start = 1):
                print(f"\nHDU {i} - 名称: {hdu.name}, 类型: {hdu.__class__.__name__}")
                print(hdu.header)
        hdul.close()
    elif type == '2':
        # 加载光谱数据
        freqs, flux = load_alfalfa_spectrum(fitsf)
        if freqs is not None:
            print("频率范围:", freqs.min(), "到", freqs.max())
            print("通量形状:", flux.shape)
    elif type == '3':
        # 重采样到目标频率
        freqs_target = np.linspace(1400, 1440, 256)
        flux_resampled = load_alfalfa_spectrum(fitsf, freqs_target = freqs_target)
        if flux_resampled is not None:
            print("重采样通量形状:", flux_resampled.shape)
    else:
        print("Type not supported.")

def f(path):
    return path


if __name__ == '__main__':
    fitsf = f('fits/AGC_100202.fits')
    check_alfalfa_fits('1')
    check_alfalfa_fits('2')
    check_alfalfa_fits('3')
