import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def execute_hits(patch: np.ndarray, tsamp: float, foff: float, max_drift: float = 4.0, min_drift: float = 0.0,
                 snr_threshold: float = 10.0) -> pd.DataFrame:
    """
    Computes hits from a given waterfall patch tensor using reconstructed turbo_seti logic.

    Parameters:
    - patch: np.ndarray of shape (n_time, n_freq) containing power values.
    - tsamp: Time resolution per integration in seconds.
    - foff: Frequency resolution per channel in Hz (can be negative if frequencies decrease with channel index).
    - max_drift: Maximum drift rate to search (Hz/s).
    - min_drift: Minimum drift rate to search (Hz/s).
    - snr_threshold: Minimum SNR for a hit to be considered.

    Returns:
    - pd.DataFrame with columns: DriftRate, SNR, Uncorrected_Frequency, freq_start, freq_end.
      Frequencies are relative positions starting from 0 Hz.
    """
    n_time, n_freq = patch.shape
    if n_time < 2 or n_freq < 2:
        return pd.DataFrame(columns=['DriftRate', 'SNR', 'Uncorrected_Frequency', 'freq_start', 'freq_end'])

    delta_t = (n_time - 1) * tsamp
    k_max = int(np.ceil(max_drift * delta_t / np.abs(foff)))
    k_min = int(np.floor(min_drift * delta_t / np.abs(foff)))
    k_values = np.arange(k_min, k_max + 1)

    hits = []

    for k in k_values:
        drift_rate = k * foff / delta_t
        if not (min_drift <= drift_rate <= max_drift):
            continue

        step = k / (n_time - 1.0)
        spec = np.zeros(n_freq)

        chan_grid = np.arange(n_freq)
        for t in range(n_time):
            shifted = chan_grid + step * t
            shifted_int = np.round(shifted).astype(int)
            valid = (shifted_int >= 0) & (shifted_int < n_freq)
            spec[valid] += patch[t, shifted_int[valid]]

        # Compute robust statistics
        med = np.median(spec)
        mad = np.median(np.abs(spec - med))
        if mad == 0:
            continue
        sigma = mad * 1.4826  # Approximate std for Gaussian

        snr_spec = (spec - med) / sigma

        # Find local maxima above threshold
        peaks, properties = find_peaks(snr_spec, height=snr_threshold)

        for idx in range(len(peaks)):
            chan = peaks[idx]
            snr = properties['peak_heights'][idx]
            uncorr_freq = chan * foff  # Relative, fch1=0

            freq_end_calc = uncorr_freq + drift_rate * delta_t
            freq_start = min(uncorr_freq, freq_end_calc)
            freq_end = max(uncorr_freq, freq_end_calc)

            hits.append({
                'DriftRate': drift_rate,
                'SNR': snr,
                'Uncorrected_Frequency': uncorr_freq,
                'freq_start': freq_start,
                'freq_end': freq_end
            })

    df_hits = pd.DataFrame(hits)
    if df_hits.empty:
        return df_hits

    # Optional: sort by SNR descending
    df_hits = df_hits.sort_values(by='SNR', ascending=False).reset_index(drop=True)

    return df_hits
