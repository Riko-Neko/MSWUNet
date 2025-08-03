from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

from SETIdataset import DynamicSpectrumDataset


def ra_deg_to_hhmmss(ra_deg):
    """
    Convert RA from degrees to hhmmss.s format (hours, minutes, seconds).

    Parameters:
    ra_deg (float): Right ascension in degrees.

    Returns:
    float: RA in hhmmss.s format.
    """
    ra_hours = ra_deg / 15.0
    hours = int(ra_hours)
    minutes = int((ra_hours - hours) * 60)
    seconds = ((ra_hours - hours) * 60 - minutes) * 60
    return hours * 10000 + minutes * 100 + seconds


def dec_deg_to_ddmmss(dec_deg):
    """
    Convert DEC from degrees to ddmmss.s format (degrees, minutes, seconds).

    Parameters:
    dec_deg (float): Declination in degrees.

    Returns:
    float: DEC in ddmmss.s format.
    """
    degrees = int(dec_deg)
    minutes = int((abs(dec_deg) - abs(degrees)) * 60)
    seconds = ((abs(dec_deg) - abs(degrees)) * 60 - minutes) * 60
    sign = -1 if dec_deg < 0 else 1
    return sign * (abs(degrees) * 10000 + minutes * 100 + seconds)


def create_blimpy_compatible_h5(dataset, save_path, fch1=1420e6, foff=-0.1e6, ra_deg=123.4, dec_deg=-26.7):
    """
    Create a blimpy-compatible HDF5 file.

    Parameters:
    dataset: DynamicSpectrumDataset instance
    save_path: Save path (.h5)
    fch1: Starting frequency (Hz)
    foff: Frequency interval (Hz)
    ra_deg: Right ascension in degrees (converted to hhmmss.s)
    dec_deg: Declination in degrees (converted to ddmmss.s)
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Get data dimensions
    tchans = dataset.tchans
    fchans = dataset.fchans

    with h5py.File(save_path, 'w') as hf:
        # Create main dataset (3D: time, pol, freq)
        data_shape = (tchans, 1, fchans)  # (n_ints, n_ifs, n_chans)
        data_dset = hf.create_dataset('data', shape=data_shape, dtype=np.float32)

        # Generate and write data
        noisy, _, _ = dataset[0]
        data_dset[:] = np.array(noisy).reshape(data_shape).astype(np.float32)

        # Set blimpy-specific attributes on the file
        hf.attrs['CLASS'] = 'FILTERBANK'  # Indicates the type of data (blimpy-specific)
        hf.attrs['VERSION'] = '1.0'  # Version of the format (blimpy-specific)

        # Set standard SIGPROC header attributes on the dataset
        data_dset.attrs['telescope_id'] = 6  # Telescope ID (6 for GBT)
        data_dset.attrs['machine_id'] = 0  # Machine ID (0 for FAKE)
        data_dset.attrs['data_type'] = 1  # Data type (1 for filterbank)
        data_dset.attrs[
            'rawdatafile'] = f'simulated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.fil'  # Original data file name
        data_dset.attrs['source_name'] = 'SIMULATED'  # Source name
        data_dset.attrs['barycentric'] = 0  # 1 if barycentered, 0 otherwise
        data_dset.attrs['pulsarcentric'] = 0  # 1 if pulsarcentric, 0 otherwise
        data_dset.attrs['az_start'] = 0.0  # Telescope azimuth at start (degrees)
        data_dset.attrs['za_start'] = 0.0  # Telescope zenith angle at start (degrees)
        data_dset.attrs['src_raj'] = ra_deg_to_hhmmss(ra_deg)  # RA in hhmmss.s format
        data_dset.attrs['src_dej'] = dec_deg_to_ddmmss(dec_deg)  # DEC in ddmmss.s format
        data_dset.attrs['tstart'] = 50000.0  # Start time in MJD
        data_dset.attrs['tsamp'] = float(dataset.dt)  # Time sample interval (seconds)
        data_dset.attrs['nbits'] = 32  # Number of bits per sample (32 for float32)
        data_dset.attrs['nchans'] = fchans  # Number of frequency channels
        data_dset.attrs['nifs'] = 1  # Number of IF channels
        data_dset.attrs['fch1'] = float(fch1)  # Center frequency of first channel (Hz)
        data_dset.attrs['foff'] = float(foff)  # Channel bandwidth (MHz)

        print(f"[\033[32mINFO\033[0m] Successfully created compatible file: {save_path}")
        print(f"[\033[32mINFO\033[0m] Data shape: {data_dset.shape}")


if __name__ == "__main__":
    fchans = 1024
    tchans = 128
    df = 7.5
    dt = 1.0
    bandwidth = df * fchans / 1e6  # MHz
    test_dataset = DynamicSpectrumDataset(
        tchans=tchans, fchans=fchans, df=df, dt=dt,
        drift_min=-1.0, drift_max=1.0,
        snr_min=10.0, snr_max=20.0,
        width_min=5, width_max=7.5,
        num_signals=(1, 1),
        noise_std_min=0.05, noise_std_max=0.1
    )

    create_blimpy_compatible_h5(
        dataset=test_dataset,
        save_path=f"./test_out/setigen_wf_sim/wf_sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5",
        fch1=1420e6,
        foff=bandwidth
    )
