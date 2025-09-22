import numpy as np


def add_rfi(signal_spec, rfi_params, noise_std=0.0):
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
        for _ in range(rfi_params.get('NBC', 0)):
            chan_idx = np.random.randint(0, n_chan)
            rfi_strength = rfi_params.get('NBC_amp', 5.0) * noise_std
            injected_spec[:, chan_idx] += rfi_strength
            rfi_mask[:, chan_idx] = True

        # Narrow Band Transient (NBT) RFI
        for _ in range(rfi_params.get('NBT', 0)):
            chan_idx = np.random.randint(0, n_chan)
            frame_start = np.random.randint(0, time_frames)
            duration_max = min(5, time_frames - frame_start)
            if duration_max <= 1:
                continue
            duration = np.random.randint(1, duration_max)
            rfi_strength = rfi_params.get('NBT_amp', 8.0) * noise_std * np.random.randn()
            injected_spec[frame_start:frame_start + duration, chan_idx] += rfi_strength
            rfi_mask[frame_start:frame_start + duration, chan_idx] = True

        # Broad Band Transient (BBT) RFI
        for _ in range(rfi_params.get('BBT', 0)):
            frame_idx = np.random.randint(0, time_frames)
            chan_start = np.random.randint(0, n_chan // 2)
            chan_width = np.random.randint(n_chan // 10, n_chan // 2)
            rfi_strength = rfi_params.get('BBT_amp', 10.0) * noise_std
            affected_chans = slice(chan_start, min(n_chan, chan_start + chan_width))
            injected_spec[frame_idx, affected_chans] += rfi_strength
            rfi_mask[frame_idx, affected_chans] = True

    return injected_spec, rfi_mask
