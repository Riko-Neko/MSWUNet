"""
This is a test script for the turbo_seti package. It searches for Doppler shifts in BLIS692NS_EV data.

OPPS! This script is outdated and needs to be updated to work with the latest turbo_seti version.

"""
import glob
import os
import tempfile
import time
import traceback

import h5py
import numpy as np
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

            print("\n[Info] Attributes of ‘data’:")
            for key, value in obj.attrs.items():
                print(f"{key}: {value}")


def search_doppler(h5_path_or_dir, output_dir='./test_out/truboseti_blis692ns', rescue=False):
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
        if rescue:
            wf = load_waterfall_compatibly(h5_file)
        else:
            wf = Waterfall(h5_file)
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


def load_waterfall_compatibly(h5_file):
    """
    Attempts to load the HDF5 file using Waterfall.
    If the file cannot be loaded directly, rewraps the 'data' dataset into a new HDF5 file.
    Returns a Waterfall object or None if both attempts fail.
    """

    print(f"\n[Info] Trying to load Waterfall file: {h5_file}")
    try:
        wf = Waterfall(h5_file)
        _ = wf.container  # 强制访问底层 container，确保 header 读取
        print("[Info] Successfully loaded with Waterfall.")
        return wf
    except Exception as e:
        print(f"[Warning] Waterfall could not load original h5: {e}")
        traceback.print_exc()
    except BaseException as e:
        print(f"[Critical] Waterfall triggered a critical error: {e}")
        traceback.print_exc()

    print("[Info] Trying to wrap 'data' into a new .h5...")

    # 尝试包装为新 h5
    tmp_dir = tempfile.mkdtemp()
    wrapped_file = os.path.join(tmp_dir, "wrapped_data.h5")
    try:
        with h5py.File(h5_file, 'r') as f_in:
            with h5py.File(wrapped_file, 'w') as f_out:
                f_out.create_dataset('data', data=np.array(f_in['data']))

                for key, value in f_out['data'].attrs.items():
                    f_out['data'].attrs[key] = value

                if 'mask' in f_in:
                    f_out.create_dataset('mask', data=np.array(f_in['mask']))

                # Add required attributes for Waterfall to accept the file
                f_out.attrs['CLASS'] = b'FILTERBANK'
                f_out.attrs['VERSION'] = b'1.0'
                f_out.attrs['TELESCOP'] = b'GBT'

        wf = Waterfall(wrapped_file)
        _ = wf.container  # again force reading
        print("[Info] Successfully loaded wrapped .h5 with Waterfall.")
        return wf
    except Exception as e2:
        print(f"[Error] Failed to load wrapped .h5: {e2}")
        traceback.print_exc()
        return None


if __name__ == '__main__':
    # Search for Doppler shifts in a single .h5 file
    # search_doppler('test_out/setigen_sim/waterfall.h5')

    # Search for Doppler shifts in a directory of .h5 files
    search_doppler('../data/BLIS692NS_EV/HIP17147', rescue=False)
