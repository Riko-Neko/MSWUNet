"""
This is a test script for the turbo_seti package. It searches for Doppler shifts in BLIS692NS_EV data.

OPPS! This script may be outdated and needs to be updated to work with the latest turbo_seti version.

"""
import glob
import os
import time

import h5py
from blimpy import Waterfall
from turbo_seti.find_doppler.find_doppler import FindDoppler

from TruboSETI_filgen import print_waterfall_debug_info


def inspect_h5_structure(h5_file):
    with h5py.File(h5_file, 'r') as f:
        def printname(name):
            print(name)

        print("[\033[32mInfo\033[0m] All group/dataset paths in file:")
        f.visit(printname)

        print("\n[\033[32mInfo\033[0m] Top-level keys:")
        print(list(f.keys()))

        if 'data' in f:
            obj = f['data']
            if isinstance(obj, h5py.Dataset):
                print("`data` is a Dataset.")
            elif isinstance(obj, h5py.Group):
                print("`data` is a Group. It contains:")
                print(list(obj.keys()))

                print("\n[\033[32mInfo\033[0m] Recursively listing contents under 'data':")

                def visit_data(name):
                    print(name)

                obj.visit(visit_data)

            print("\n[\033[32mInfo\033[0m] Attributes of ‘data’:")
            for key, value in obj.attrs.items():
                print(f"{key}: {value}")


def search_doppler_h5(h5_path_or_dir, out_dir='./test_out/truboseti_blis692ns', drift_rate: float = 2.0,
                      snr: float = 10, check=True, **kwargs):
    os.makedirs(out_dir, exist_ok=True)

    if os.path.isfile(h5_path_or_dir) and h5_path_or_dir.endswith('.h5'):
        h5_files = [h5_path_or_dir]
    elif os.path.isdir(h5_path_or_dir):
        h5_files = glob.glob(os.path.join(h5_path_or_dir, '**', '*.h5'), recursive=True)
    else:
        print(f"[\033[31mError\033[0m] Invalid path: {h5_path_or_dir}")
        return

    if not h5_files:
        print(f"[\033[33mWarning\033[0m] No .h5 files found in '{h5_path_or_dir}'")
        return

    for h5_file in h5_files:
        if not os.path.exists(h5_file):
            print(f"[\033[31mError\033[0m] File not found: {h5_file}")
            continue

        print(f"\n[\033[32mInfo\033[0m] Processing file: {h5_file}")

        if check:
            inspect_h5_structure(h5_file)

        wf = Waterfall(h5_file)

        print("[\033[32mInfo\033[0m] Waterfall object created", wf.header)
        wf.info()
        if check:
            print_waterfall_debug_info(wf)
            input("Press Enter to continue...")

        print("[\033[32mInfo\033[0m] Starting FindDoppler search ...")
        t1 = time.time()
        fdop = FindDoppler(h5_file,
                           max_drift=drift_rate,
                           snr=snr,
                           out_dir=out_dir,
                           **kwargs)
        fdop.search()
        elapsed_time = time.time() - t1
        print(f"\n[\033[32mInfo\033[0m] FindDoppler.search() completed in {elapsed_time:.2f} seconds for: {h5_file}")


if __name__ == '__main__':
    h5_file = max(glob.glob("test_out/setigen_wf_sim/wf_sim_*.h5"), key=os.path.getctime)
    # fil_file = '../data/BLIS692NS/BLIS692NS_EV/HIP17147/spliced_blc0001020304050607_guppi_57523_69379_HIP17147_0015.gpuspec.0000.h5'
    out_dir = f'./test_out/truboseti_blis692ns/{os.path.splitext(os.path.basename(h5_file))[0]}'
    drift_rate = 2.0
    snr = 10
    check = True

    # Extended options
    kwargs = {
        'min_drift': 0.00001,
        'coarse_chans': None,
        'obs_info': None,
        'flagging': False,
        'n_coarse_chan': None,
        'kernels': None,
        'gpu_backend': False,
        'gpu_id': 0,
        'precision': 1,
        'append_output': False,
        'blank_dc': True,
    }

    search_doppler_h5(h5_file, out_dir=out_dir, drift_rate=drift_rate, snr=snr, check=check, **kwargs)
