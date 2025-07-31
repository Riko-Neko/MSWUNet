import os
import shutil
import pandas as pd
from astropy.io import fits

def classify_fits_file(filepath):
    """
    判断 FITS 文件格式类型：
    - 返回 'first' / 'second' / 'third'
    """
    try:
        with fits.open(filepath) as hdul:
            if len(hdul) <= 1 or not isinstance(hdul[1], fits.BinTableHDU):
                return 'third'

            # 取出表头字段
            colnames = set(hdul[1].columns.names)

            second_keys = {'wave', 'waveErr', 'waveMin', 'waveMax', 'sigma', 'sigmaErr',
                           'sigmaMin', 'sigmaMax', 'height', 'heightErr', 'continuum',
                           'ew', 'ewErr', 'ewMin', 'specIndex', 'nsigma', 'chisq', 'nu',
                           'restWave', 'weight', 'z', 'zErr', 'lineMask'}

            first_keys = {'VHELIO', 'FREQUENCY', 'FLUX', 'BASELINE', 'WEIGHT'}

            if second_keys.issubset(colnames):
                return 'second'
            elif first_keys.issubset(colnames):
                return 'first'
            else:
                return 'unknown'  # 可选处理

    except Exception as e:
        print(f"[错误] 无法读取 {filepath}: {e}")
        return 'error'


def organize_spectra_files(source_dir='./alfalfa_spectra/train'):
    others_dir = os.path.join(os.path.dirname(source_dir), 'others')
    t_dir = os.path.join(others_dir, 'T')
    u_dir = os.path.join(others_dir, 'U')

    os.makedirs(others_dir, exist_ok=True)
    os.makedirs(t_dir, exist_ok=True)

    for fname in os.listdir(source_dir):
        fpath = os.path.join(source_dir, fname)
        if not os.path.isfile(fpath):
            continue

        # 仅处理 FITS 文件
        if not fname.lower().endswith(('.fits', '.fit')):
            continue

        category = classify_fits_file(fpath)

        if category == 'second':
            print(f"[SECOND] 移动到 others/: {fname}")
            shutil.move(fpath, os.path.join(others_dir, fname))
        elif category == 'third':
            print(f"[THIRD] 移动到 others/T/: {fname}")
            shutil.move(fpath, os.path.join(t_dir, fname))
        elif category == 'unknown':
            print(f"[UNKNOWN] 未知文件: {fname} ({category})")
            shutil.move(fpath, os.path.join(u_dir, fname))
        else:
            print(f"[SKIP] 保留文件: {fname} ({category})")

# 运行
organize_spectra_files()
organize_spectra_files(source_dir='./alfalfa_spectra/val')