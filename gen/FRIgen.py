import numpy as np


def add_rfi(signal_spec, rfi_params, noise_std=0.0, use_real_fil=False):
    """
    Add RFI to the clean spectrum and create an RFI mask.

    Parameters:
    signal_spec : 2D array, shape (time_frames, n_chan)
        The clean dynamic spectrum (background + signal).
    rfi_params : dict
        Dictionary containing RFI parameters:
            'NBC': int, number of narrowband continuous RFI
            'NBC_amp': float, amplitude multiplier for NBC RFI
            'NBT': int, number of narrowband transient RFI
            'NBT_amp': float, amplitude multiplier for NBT RFI
            'BBT': int, number of broad band transient RFI
            'BBT_amp': float, amplitude multiplier for BBT RFI
    noise_std : float
        Standard deviation of the background noise, used to scale RFI strengths.

    Returns:
    injected_spec : 2D array, shape (time_frames, n_chan)
        The noisy spectrum with RFI added.
    rfi_mask : 2D array, shape (time_frames, n_chan), dtype=bool
        Mask indicating where RFI is present.
    """
    time_frames, n_chan = signal_spec.shape
    injected_spec = signal_spec.copy()
    rfi_mask = np.zeros_like(signal_spec, dtype=bool)

    if rfi_params:
        # Narrow Band Continuous (NBC) RFI
        if np.random.rand() < 1:
            for _ in range(rfi_params.get('NBC', 0)):
                chan_idx = np.random.randint(0, n_chan)
                rfi_strength = rfi_params.get('NBC_amp', 5.0) * noise_std
                if use_real_fil:
                    rfi_strength *= 1e12
                injected_spec[:, chan_idx] += rfi_strength
                rfi_mask[:, chan_idx] = True

        # Narrow Band Transient (NBT) RFI
        if np.random.rand() < 1:
            for _ in range(rfi_params.get('NBT', 0)):
                chan_idx = np.random.randint(0, n_chan)
                frame_start = np.random.randint(0, time_frames)
                duration_max = min(5, time_frames - frame_start)
                if duration_max <= 1:
                    continue
                duration = np.random.randint(1, duration_max)
                rfi_strength = rfi_params.get('NBT_amp', 8.0) * noise_std * np.random.randn()
                if use_real_fil:
                    rfi_strength *= 1e12
                injected_spec[frame_start:frame_start + duration, chan_idx] += rfi_strength
                rfi_mask[frame_start:frame_start + duration, chan_idx] = True

        # Broad Band Transient (BBT) RFI
        if np.random.rand() < 1:
            for _ in range(rfi_params.get('BBT', 0)):
                t_len = np.random.randint(max(1, time_frames // 20), max(2, time_frames // 5))
                frame_start = np.random.randint(0, max(1, time_frames - t_len))
                affected_frames = slice(frame_start, frame_start + t_len)
                affected_chans = slice(0, n_chan)
                rfi_strength = rfi_params.get('BBT_amp', 10.0) * noise_std
                if use_real_fil:
                    rfi_strength *= 1e12
                # --- Time edge 10% taper（raised-cosine / Hann half-window）---
                edge = max(1, int(0.3 * t_len))  # 上下各 30%
                w = np.ones(t_len, dtype=injected_spec.dtype)
                ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, edge, endpoint=False))
                w[:edge] = ramp
                w[-edge:] = ramp[::-1]
                injected_spec[affected_frames, affected_chans] += (rfi_strength * w)[:, None]
                rfi_mask[affected_frames, affected_chans] = True

    return injected_spec, rfi_mask
