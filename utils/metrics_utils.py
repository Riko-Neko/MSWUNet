import math
import os

import networkx as nx
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from scipy.ndimage import maximum_filter
from scipy.signal import find_peaks

from gen.SETIdataset import DynamicSpectrumDataset


def SNR_filter(tensor_2d: torch.Tensor, top_fraction: float = 0.002, min_pixels: int = 50, ) -> float:
    """
    Estimate the overall SNR from a 2D spectrogram.

    Definition:
      1. Estimate the background noise σ using the median of the entire image plus MAD:
           median = median(x)
           MAD    = median(|x - median|)
           σ      ≈ 1.4826 * MAD
      2. Take the top_fraction brightest pixels (at least min_pixels) and calculate their average mean_top.
      3. Return SNR = (mean_top - median) / σ

    Parameters:
        tensor_2d: 2D Tensor, shape (T, F) or (1, T, F) / (C, T, F)
        top_fraction: The proportion of the brightest pixels participating in the "signal average", e.g., 0.002 ≈ 0.2%
        min_pixels: The minimum number of pixels to be involved in the calculation to avoid instability when the patch is too small or top_fraction is too low.

    Returns:
        Python float scalar SNR; returns 0.0 if noise estimation fails or if there is an anomaly in the data.
    """
    if tensor_2d.ndim == 3:
        tensor_2d = tensor_2d.squeeze(0)
    if tensor_2d.ndim != 2:
        raise ValueError(f"[\033[31mError\033[0m] Expects 2D tensor, got shape {tuple(tensor_2d.shape)}")

    x = tensor_2d.detach().float().cpu().view(-1)
    if x.numel() < 4:
        return 0.0

    median = x.median()
    mad = (x - median).abs().median()
    sigma = mad * 1.4826

    if sigma <= 0 or torch.isnan(sigma) or torch.isinf(sigma):
        return 0.0

    k = max(min_pixels, int(top_fraction * x.numel()))
    k = min(k, x.numel())
    top_vals, _ = torch.topk(x, k)

    mean_top = top_vals.mean()
    snr = (mean_top - median) / sigma
    snr_val = float(snr.item())

    if not math.isfinite(snr_val):
        return 0.0
    return snr_val


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
        sigma = mad * 1.4826

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


def execute_hits_peaks(patch: np.ndarray, tsamp: float, foff: float, max_drift: float = 4.0, min_drift: float = 0.0,
                       snr_threshold: float = 10.0) -> pd.DataFrame:
    """
    Computes hits from a given waterfall patch tensor using peak detection and linear fitting for signal lines.

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

    # Global statistics for SNR calculation
    med_global = np.median(patch)
    mad_global = np.median(np.abs(patch - med_global))
    if mad_global == 0:
        return pd.DataFrame(columns=['DriftRate', 'SNR', 'Uncorrected_Frequency', 'freq_start', 'freq_end'])
    sigma_global = mad_global * 1.4826

    # Parameters for peak detection
    pixel_snr_threshold = 2.0
    min_points = n_time // 2
    max_chan_dist = 1

    # Collect peaks from each time slice
    points = []
    for t in range(n_time):
        spec_t = patch[t, :]
        med_t = np.median(spec_t)
        mad_t = np.median(np.abs(spec_t - med_t))
        if mad_t == 0:
            continue
        sigma_t = mad_t * 1.4826
        snr_spec_t = (spec_t - med_t) / sigma_t
        peaks_t, props_t = find_peaks(snr_spec_t, height=pixel_snr_threshold)
        for idx in range(len(peaks_t)):
            points.append((t, peaks_t[idx], props_t['peak_heights'][idx]))

    if not points:
        return pd.DataFrame(columns=['DriftRate', 'SNR', 'Uncorrected_Frequency', 'freq_start', 'freq_end'])

    # Build graph for connected components (chains of peaks)
    G = nx.Graph()
    for i in range(len(points)):
        G.add_node(i)

    for i in range(len(points)):
        t1, chan1, _ = points[i]
        for j in range(i + 1, len(points)):
            t2, chan2, _ = points[j]
            if abs(t1 - t2) == 1 and abs(chan1 - chan2) <= max_chan_dist:
                G.add_edge(i, j)

    hits = []
    for comp in nx.connected_components(G):
        if len(comp) < min_points:
            continue
        comp_idx = list(comp)
        comp_points = [points[k] for k in comp_idx]
        t_arr = np.array([p[0] for p in comp_points])
        chan_arr = np.array([p[1] for p in comp_points])

        # Linear fit
        slope, intercept = np.polyfit(t_arr, chan_arr, 1)
        drift_rate = slope * foff / tsamp
        if not (min_drift <= drift_rate <= max_drift):
            continue

        # Compute SNR along the fitted line
        total = 0.0
        count = 0
        for t in range(n_time):
            chan_t = intercept + slope * t
            chan_idx = round(chan_t)
            if 0 <= chan_idx < n_freq:
                chan_int = int(chan_idx)
                total += patch[t, chan_int]
                count += 1

        if count == 0:
            continue

        snr = (total - med_global * count) / (sigma_global * np.sqrt(count))
        if snr < snr_threshold:
            continue

        uncorr_freq = intercept * foff
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

    # Sort by SNR descending
    df_hits = df_hits.sort_values(by='SNR', ascending=False).reset_index(drop=True)

    return df_hits


def execute_hits_hough(patch: np.ndarray, tsamp: float, foff: float, max_drift: float = 4.0, min_drift: float = 0.0,
                       snr_threshold: float = 10.0, min_abs_drift: float = 0.0,
                       merge_tol: float = 0.05) -> pd.DataFrame:
    """
    Computes hits from a given waterfall patch using Hough transform with:
    - duplicate suppression (merge_tol),
    - optional minimum absolute drift filtering (min_abs_drift),
    - non-maximum suppression in Hough space.

    Parameters:
    - patch: np.ndarray of shape (n_time, n_freq) containing power values.
    - tsamp: Time resolution per integration in seconds.
    - foff: Frequency resolution per channel in Hz.
    - max_drift: Maximum drift rate to search (Hz/s).
    - min_drift: Minimum drift rate to search (Hz/s).
    - snr_threshold: Minimum SNR for a hit to be considered.
    - min_abs_drift: Minimum absolute drift rate to record a hit (Hz/s).
    - merge_tol: Tolerance for merging similar hits (controls strictness of duplicate suppression).

    Returns:
    - pd.DataFrame with columns: DriftRate, SNR, Uncorrected_Frequency, freq_start, freq_end.
    """
    n_time, n_freq = patch.shape
    if n_time < 2 or n_freq < 2:
        return pd.DataFrame(columns=['DriftRate', 'SNR', 'Uncorrected_Frequency', 'freq_start', 'freq_end'])

    delta_t = (n_time - 1) * tsamp

    # Global statistics for SNR
    med_global = np.median(patch)
    mad_global = np.median(np.abs(patch - med_global))
    if mad_global == 0:
        return pd.DataFrame(columns=['DriftRate', 'SNR', 'Uncorrected_Frequency', 'freq_start', 'freq_end'])
    sigma_global = mad_global * 1.4826

    # Peak detection
    pixel_snr_threshold = 2.0
    points = []
    for t in range(n_time):
        spec_t = patch[t, :]
        med_t = np.median(spec_t)
        mad_t = np.median(np.abs(spec_t - med_t))
        if mad_t == 0:
            continue
        sigma_t = mad_t * 1.4826
        snr_spec_t = (spec_t - med_t) / sigma_t
        peaks_t, props_t = find_peaks(snr_spec_t, height=pixel_snr_threshold)
        for idx in range(len(peaks_t)):
            points.append((t, peaks_t[idx], props_t['peak_heights'][idx]))

    if not points:
        return pd.DataFrame(columns=['DriftRate', 'SNR', 'Uncorrected_Frequency', 'freq_start', 'freq_end'])

    # Hough accumulator
    drift_rates = np.linspace(min_drift, max_drift, 200)
    intercept_bins = n_freq
    accumulator = np.zeros((len(drift_rates), intercept_bins))
    for (t, chan, _) in points:
        for i, dr in enumerate(drift_rates):
            slope = dr * tsamp / foff
            intercept = chan - slope * t
            if 0 <= intercept < intercept_bins:
                accumulator[i, int(intercept)] += 1

    # Non-maximum suppression
    footprint = np.ones((3, 3))
    local_max = (accumulator == maximum_filter(accumulator, footprint=footprint))
    threshold_votes = max(3, n_time // 3)
    candidate_idxs = np.argwhere(local_max & (accumulator >= threshold_votes))

    hits = []
    for i, j in candidate_idxs:
        drift_rate = drift_rates[i]
        if abs(drift_rate) < min_abs_drift:
            continue  # skip hits with too small drift
        intercept = j

        # Compute SNR along the line
        total = 0.0
        count = 0
        for t in range(n_time):
            chan_t = intercept + (drift_rate * tsamp / foff) * t
            chan_idx = round(chan_t)
            if 0 <= chan_idx < n_freq:  # 确保索引在有效范围内
                total += patch[t, int(chan_idx)]
                count += 1
        if count == 0:
            continue

        snr = (total - med_global * count) / (sigma_global * np.sqrt(count))
        if snr < snr_threshold:
            continue

        uncorr_freq = intercept * foff
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

    # Sort by SNR descending
    df_hits = df_hits.sort_values(by='SNR', ascending=False).reset_index(drop=True)

    # Merge similar hits to remove duplicates (merge_tol controls strictness)
    merged = []
    for _, row in df_hits.iterrows():
        duplicate = False
        for k in merged:
            if (abs(row['DriftRate'] - k['DriftRate']) <= merge_tol and
                    abs(row['Uncorrected_Frequency'] - k['Uncorrected_Frequency']) <= merge_tol * abs(
                        row['DriftRate'])):
                duplicate = True
                break
        if not duplicate:
            merged.append(row)
    df_hits = pd.DataFrame(merged)

    return df_hits


if __name__ == "__main__":
    tchans = 144
    fchans = 1024
    df = 7.5
    dt = 1.0
    drift_min = -4.0
    drift_max = 4.0
    # drift_min_abs = df // (tchans * dt)
    drift_min_abs = 0.0
    dataset = DynamicSpectrumDataset(tchans=tchans, fchans=fchans, df=df, dt=dt, fch1=None, ascending=True,
                                     drift_min=drift_min, drift_max=drift_max, drift_min_abs=drift_min_abs,
                                     snr_min=50.0, snr_max=60.0, width_min=10, width_max=15, num_signals=(1, 1),
                                     noise_std_min=0.025, noise_std_max=0.05)

    out_dir = "../pipeline/log/metrics_test"
    os.makedirs(out_dir, exist_ok=True)

    num_samples = 10
    for i in range(num_samples):
        noisy_spec, clean_spec, rfi_mask, phy_prob = dataset[i]  # Random generation each time
        # spectrum = noisy_spec
        spectrum = clean_spec


        def add_gaussian_noise(clean_spec, noise_level=0.01):
            noise = np.random.normal(loc=0.0, scale=noise_level, size=clean_spec.shape)
            noisy_spec = clean_spec + noise
            return noisy_spec


        spectrum = add_gaussian_noise(spectrum, noise_level=0.1)

        # Plot spectrogram
        spec = spectrum.squeeze()
        plt.figure(figsize=(15, 3))
        plt.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
        plt.title(f"Clean spectrogram #{i}")
        plt.colorbar()

        # Execute hits_peaks on spectrogram
        patch = spectrum.squeeze()
        foff = -df if not dataset.ascending else df  # Since ascending=False, foff negative
        # hits = execute_hits(patch, tsamp=dt, foff=foff, max_drift=drift_max, min_drift=drift_min_abs, snr_threshold=10.0)
        # hits = execute_hits_peaks(patch, tsamp=dt, foff=foff, max_drift=drift_max, min_drift=drift_min_abs,snr_threshold=10.0)
        hits = execute_hits_hough(patch, tsamp=dt, foff=foff, max_drift=drift_max, min_drift=drift_min,
                                  snr_threshold=10.0, min_abs_drift=drift_min_abs, merge_tol=1000)


        # Function to plot lines for hits
        def plot_hits_lines(hits, color='r', label='Fit'):
            if hits.empty:
                return

            for i, (_, hit) in enumerate(hits.iterrows()):
                drift_rate = hit['DriftRate']
                uncorr_freq = hit['Uncorrected_Frequency']
                chan_start = uncorr_freq / foff
                drift_rate_chan = drift_rate * dt / foff
                t_vals = np.arange(0, tchans)
                chan_vals = chan_start + drift_rate_chan * t_vals
                plt.plot(chan_vals, t_vals, color + '--', label=label if i == 0 else None, alpha=0.3)
            plt.legend()


        # Plot lines for peaks fit method
        plot_hits_lines(hits)

        plt.tight_layout()
        clean_path = os.path.join(out_dir, f"clean_{i:03d}.png")
        plt.savefig(clean_path)
        plt.close()
        print(f"[\033[32mInfo\033[0m] Saved clean plot with Peaks Fit lines to {clean_path}")

        # Print to console
        print(f"\n[\033[36mMetrics\033[0m] \nSample {i}: phy_prob = {phy_prob}")
        print("Peaks Fit hits:")
        if hits.empty:
            print("No hits detected.")
        else:
            print(hits)

        # Save hits to CSV
        hits_path = os.path.join(out_dir, f"hits_{i:03d}.csv")
        hits.to_csv(hits_path, index=False)
        print(f"[\033[32mInfo\033[0m] Saved hits to {hits_path}")
