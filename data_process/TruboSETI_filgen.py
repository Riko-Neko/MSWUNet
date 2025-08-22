import os

from blimpy import Waterfall
from turbo_seti.find_doppler.find_doppler import FindDoppler


def print_waterfall_debug_info(wf):
    print("[\033[34mDEBUG\033[0m]] Waterfall object information:")
    print(f"  Filename: {getattr(wf, 'filename', 'N/A')}")
    print(f"  File extension: {getattr(wf, 'ext', 'N/A')}")

    print("\n[\033[34mDEBUG\033[0m] Container-level metadata:")
    print(f"  Number of integrations in file: {getattr(wf, 'n_ints_in_file', 'N/A')}")
    print(f"  File shape (time, frequency): {getattr(wf, 'file_shape', 'N/A')}")
    print(f"  File size (bytes): {getattr(wf, 'file_size_bytes', 'N/A')}")
    print(f"  Selection shape: {getattr(wf, 'selection_shape', 'N/A')}")
    print(f"  Number of frequency channels: {getattr(wf, 'n_channels_in_file', 'N/A')}")

    print("\n[\033[34mDEBUG\033[0m] Header information:")
    header = getattr(wf, 'header', None)
    if header:
        for key, value in header.items():
            print(f"    {key}: {value}")
    else:
        print("    No header available.")


def doppler_search_fil(fil_file: str, out_dir: str = './test_out/truboseti_blis692ns', drift_rate: float = 2.0,
                       snr: float = 10, check: bool = True, **kwargs):
    """
    Perform turbo_seti Doppler search on a .fil file.

    Parameters:
    - fil_file: str, path to the Filterbank file
    - out_dir: str, directory to output the results
    - drift_rate: float, maximum drift rate (Hz/s), default is 2.0
    """
    if not os.path.exists(fil_file):
        print(f"[\033[31mError\033[0m] File not found: {fil_file}")
        return

    if not fil_file.endswith('.fil'):
        print(f"[\033[31mError\033[0m] Invalid file type: {fil_file} is not a .fil file")
        return

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print(f"[\033[32mInfo\033[0m] Starting Doppler search for: {fil_file}")
    print(f"[\033[32mInfo\033[0m] Output directory: {out_dir}")

    # Create Waterfall object
    wf = Waterfall(fil_file)
    print("[\033[32mInfo\033[0m] Waterfall object created", wf.header)
    wf.info()
    if check:
        print_waterfall_debug_info(wf)
        input("\n[\033[32mInfo\033[0m] Press Enter to continue...")

    # Perform Doppler search
    doppler = FindDoppler(fil_file,
                          max_drift=drift_rate,
                          snr=snr,
                          out_dir=out_dir,
                          **kwargs)
    doppler.search()

    print(f"[\033[32mInfo\033[0m] Search complete for {fil_file}")


if __name__ == "__main__":
    # fil_file = '../data/BLIS692NS/BLIS692NS_data/spliced_blc00010203040506o7o0111213141516o7o0212223242526o7o031323334353637_guppi_58060_26569_HIP17147_0021.gpuspec.0002.fil'
    fil_file = '../data/BLIS692NS/BLIS692NS_data/spliced_blc00010203040506o7o0111213141516o7o0212223242526o7o031323334353637_guppi_58060_26569_HIP17147_0021.gpuspec.0000_chunk30720000_part0.fil'
    # fil_file = '../data/BLIS692NS/BLIS692NS_data/spliced_blc00010203040506o7o0111213141516o7o0212223242526o7o031323334353637_guppi_58060_26569_HIP17147_0021.gpuspec.0000.fil'
    out_dir = f'./test_out/truboseti_blis692ns/{os.path.basename(fil_file).split('.', 1)[0]}'
    drift_rate = 5.0
    snr = 10
    check = True

    # Extended options
    kwargs = {
        'min_drift': 0.05,
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

    doppler_search_fil(fil_file, out_dir=out_dir, drift_rate=drift_rate, snr=snr, check=check, **kwargs)

