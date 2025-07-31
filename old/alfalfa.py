import numpy as np
from astropy.io import fits


def load_alfalfa_spectrum(fits_file, freqs_target=None, verbose=True):
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
                (i for i, name in enumerate(column_names) if 'FREQ' in name.upper() or 'FREQUENCY' in name.upper()),
                None)
            if freq_index is None:
                print("未找到频率或速度列")
                return None

            # 查找通量列（假设列名包含 "FLUX"）
            flux_index = next((i for i, name in enumerate(column_names) if 'FLUX' in name.upper()), None)
            if flux_index is None:
                print("未找到通量列")
                return None

            freqs = data.field(freq_index).flatten()  # 展平数组
            flux = data.field(flux_index).flatten()

            if freqs_target is not None:
                # 重采样到目标频率网格
                flux_interp = np.interp(freqs_target, freqs, flux, left=0, right=0)
                return flux_interp

            flux = np.nan_to_num(flux, nan=0.0)  # 处理NaN值

            # mean_flux = np.nanmean(flux)
            # flux = np.nan_to_num(flux, nan=mean_flux)

            return freqs, flux
    except Exception as e:
        print(f"加载FITS文件 {fits_file} 失败: {e}")
        return None
