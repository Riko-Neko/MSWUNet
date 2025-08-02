"""
This is a test script for the turbo_seti package. It searches for Doppler shifts in BLIS692NS_EV data.

      属性                 描述                 示例值
    DIMENSION_LABELS    数据维度的标签         "['frequency', 'feed_id', 'time']"
    fch1                起始频率（MHz）        8421.386717353016
    foff                频率分辨率（MHz）      -2.7939677238464355e-06
    nchans              频率通道数             1048576
    tsamp               时间采样间隔（秒）      18.25361100
    tstart              观测开始时间（MJD）     57650.78209490741
    source_name         源名称                Voyager1
    telescope_id        望远镜ID              6

"""
import glob
import os
import time

import h5py
from blimpy import Waterfall
from turbo_seti.find_doppler.find_doppler import FindDoppler


def inspect_h5_structure(h5_file):
    with h5py.File(h5_file, 'r') as f:
        def printname(name):
            print(name)

        print("[Info] All group/dataset paths in file:")
        f.visit(printname)

        print("\n[Info] Top-level keys:")
        print(list(f.keys()))

        if 'data' in f:
            obj = f['data']
            if isinstance(obj, h5py.Dataset):
                print("`data` is a Dataset.")
            elif isinstance(obj, h5py.Group):
                print("`data` is a Group. It contains:")
                print(list(obj.keys()))

                print("\n[Info] Recursively listing contents under 'data':")

                def visit_data(name):
                    print(name)

                obj.visit(visit_data)


def search_doppler(h5_path_or_dir, output_dir='./test_out/truboseti_blis692ns'):
    os.makedirs(output_dir, exist_ok=True)

    # 判断是文件还是目录
    if os.path.isfile(h5_path_or_dir) and h5_path_or_dir.endswith('.h5'):
        h5_files = [h5_path_or_dir]
    elif os.path.isdir(h5_path_or_dir):
        h5_files = glob.glob(os.path.join(h5_path_or_dir, '**', '*.h5'), recursive=True)
    else:
        print(f"[Error] Invalid path: {h5_path_or_dir}")
        return

    if not h5_files:
        print(f"[Warning] No .h5 files found in '{h5_path_or_dir}'")
        return

    for h5_file in h5_files:
        if not os.path.exists(h5_file):
            print(f"[Error] File not found: {h5_file}")
            continue

        print(f"\n[Info] Processing file: {h5_file}")

        inspect_h5_structure(h5_file)

        wf = Waterfall(filename=h5_file)
        print("[Info] Waterfall object created", wf.header)
        wf.info()

        print("[Info] Starting FindDoppler search ...")
        t1 = time.time()
        fdop = FindDoppler(
            datafile=h5_file,
            max_drift=4,
            snr=10,
            out_dir=output_dir
        )
        fdop.search()
        elapsed_time = time.time() - t1
        print(f"\n[Success] FindDoppler.search() completed in {elapsed_time:.2f} seconds for: {h5_file}")



if __name__ == '__main__':
    # 单文件搜索
    search_doppler('test_out/setigen_sim/waterfall.h5')

    # 目录搜索（递归）
    search_doppler('../data/BLIS692NS_EV/HIP17147')
