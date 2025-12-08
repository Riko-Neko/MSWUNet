import platform
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset

# from blimpy import Waterfall
from external.Waterfall import Waterfall

system = platform.system()
if system == 'Windows':
    matplotlib.use('TkAgg')
elif system == 'Darwin':
    matplotlib.use('MacOSX')
else:  # Linux
    try:
        matplotlib.use('TkAgg')
    except:
        matplotlib.use('Agg')


class SETIWaterFullDataset(Dataset):
    def __init__(self, file_path, patch_t, patch_f, overlap_pct=0.02, device="cpu", ignore_polarization=False,
                 stokes_mode="I", t_adaptive=True):
        """
        A dataset class that extracts time-frequency patches from SETI filterbank files (.fil),
        optionally combining multiple polarization files into Stokes parameters.

        Args:
            file_path (str | list[str]): Single file path or a list of polarization file paths.
            patch_t (int): Patch size along the time axis.
            patch_f (int): Patch size along the frequency axis.
            overlap_pct (float): Overlap ratio between adjacent patches.
            device (str): Device used when grabbing data ("cpu" or "cuda").
            ignore_polarization (bool):
                If True, combines polarization data according to stokes_mode (default: "I").
            stokes_mode (str):
                "I" → total intensity (xx + yy)
                "Q" → linear polarization difference (xx − yy)
                (reserved for "U", "V" in future)
            t_adaptive (bool): adapt to different t channels (default: True)
        """
        self.device = device
        self.ignore_polarization = ignore_polarization
        self.stokes_mode = stokes_mode.upper()

        if ignore_polarization:
            assert isinstance(file_path, (list, tuple)) and len(file_path) >= 2, \
                "When ignore_polarization=True, file_path must be a list of .fil files (e.g., ['xx_pol0.fil', 'xx_pol1.fil'])."
            # Ensure filenames differ only by polarization suffix
            base_names = [Path(f).stem
                          .replace("_pol0", "")
                          .replace("_pol1", "")
                          .replace("_pol2", "")
                          .replace("_pol3", "") for f in file_path]
            assert len(set(base_names)) == 1, \
                "All polarization files must have identical names except for the 'pol' part."
            self.obs_list = [Waterfall(fp, load_data=True) for fp in file_path]
            self.obs = self.obs_list[0]  # Reference polarization
        else:
            self.obs = Waterfall(file_path, load_data=True)
            self.obs_list = [self.obs]

        self.tchans = self.obs.selection_shape[0]
        self.fchans = self.obs.selection_shape[2]
        self.freqs = self.obs.get_freqs()
        if self.freqs[0] < self.freqs[-1]:
            self.ascending = True
        else:
            self.ascending = False

        if not t_adaptive:
            assert patch_t <= self.tchans, "patch_t larger than available time channels."
        else:
            patch_t = self.tchans
        assert patch_f <= self.fchans, "patch_f larger than available frequency channels."

        overlap_t = round(patch_t * overlap_pct)
        stride_t = patch_t - overlap_t
        overlap_f = round(patch_f * overlap_pct)
        stride_f = patch_f - overlap_f

        self.start_t_list = list(range(0, self.tchans - patch_t + 1, stride_t))
        if self.start_t_list and self.start_t_list[-1] + patch_t < self.tchans:
            self.start_t_list.append(self.tchans - patch_t)

        self.start_f_list = list(range(0, self.fchans - patch_f + 1, stride_f))
        if self.start_f_list and self.start_f_list[-1] + patch_f < self.fchans:
            self.start_f_list.append(self.fchans - patch_f)

        self.patch_t = patch_t
        self.patch_f = patch_f

    def __len__(self):
        return len(self.start_t_list) * len(self.start_f_list)

    def __getitem__(self, index):
        i = index // len(self.start_f_list)
        j = index % len(self.start_f_list)
        start_t = self.start_t_list[i]
        start_f = self.start_f_list[j]
        end_t = start_t + self.patch_t
        end_f = start_f + self.patch_f

        # Determine frequency range
        if self.ascending:
            f_start, f_stop = self.freqs[start_f], self.freqs[end_f - 1]
        else:
            f_start, f_stop = self.freqs[end_f - 1], self.freqs[start_f]

        # Combine polarizations if required
        if self.ignore_polarization:
            assert len(self.obs_list) >= 2, "At least two polarization files are required for Stokes combination."
            try:
                _, data_xx = self.obs_list[0].grab_data(f_start, f_stop, start_t, end_t, device=self.device)
                _, data_yy = self.obs_list[1].grab_data(f_start, f_stop, start_t, end_t, device=self.device)
            except TypeError:
                _, data_xx = self.obs_list[0].grab_data(f_start, f_stop, start_t, end_t)
                _, data_yy = self.obs_list[1].grab_data(f_start, f_stop, start_t, end_t)

            if self.stokes_mode == "I":
                data = data_xx + data_yy
            elif self.stokes_mode == "Q":
                data = data_xx - data_yy
            else:
                raise NotImplementedError(f"Stokes mode '{self.stokes_mode}' not supported yet.")
        else:
            try:
                _, data = self.obs.grab_data(f_start, f_stop, start_t, end_t, device=self.device)
            except TypeError:
                _, data = self.obs.grab_data(f_start, f_stop, start_t, end_t)

        # Normalize
        mean = np.mean(data)
        std = np.std(data)
        if std < 1e-10:
            std = 1.0
        data = (data - mean) / std

        # Convert to tensor
        patch_tensor = torch.from_numpy(data).float().unsqueeze(0)
        return patch_tensor, (start_t, start_f)

    def get_patch(self, row, col):
        """Get a patch tensor and its corresponding frequency/time range."""
        index = row * len(self.start_f_list) + col
        patch_tensor, (start_t, start_f) = self.__getitem__(index)
        end_t = start_t + self.patch_t
        end_f = start_f + self.patch_f

        if self.ascending:
            f_min, f_max = self.freqs[start_f], self.freqs[end_f - 1]
        else:
            f_min, f_max = self.freqs[end_f - 1], self.freqs[start_f]

        return patch_tensor, (f_min, f_max), (start_t, end_t)


def plot_dataset_item(dataset, index=0, cmap='viridis', log_scale=False):
    """
    Visualize a single data item (patch) from SETIDataset

    Args:
        dataset: Instance of SETIDataset
        index: Index of the data item to visualize
        cmap: Matplotlib colormap
        log_scale: Whether to use logarithmic color scaling
    """
    # Get data and position info
    patch_tensor, (start_t, start_f) = dataset[index]
    # print("[\033[33mDebug\033[0m] Data shape:", patch_tensor.shape)
    data = patch_tensor.squeeze(0).numpy()  # Remove channel dim -> (T, F)

    # Calculate frequency range
    freqs = dataset.freqs
    if freqs[0] < freqs[-1]:  # Ascending order
        f_start = freqs[start_f]
        f_stop = freqs[start_f + dataset.patch_f - 1]
    else:  # Descending order
        f_start = freqs[start_f + dataset.patch_f - 1]
        f_stop = freqs[start_f]

    # Calculate time range (using indices)
    time_indices = np.arange(start_t, start_t + dataset.patch_t)

    # Create figure
    plt.figure(figsize=(12, 8))

    # Prepare plot data (transpose to make time X-axis, frequency Y-axis)
    plot_data = data.T

    # Set color normalization
    norm = plt.Normalize()
    if log_scale:
        from matplotlib.colors import LogNorm
        # Avoid zeros/negative values
        plot_data = np.clip(plot_data, np.percentile(plot_data[plot_data > 0], 1e-10), None)
        norm = LogNorm(vmin=plot_data.min(), vmax=plot_data.max())

    # Plot time-frequency diagram
    im = plt.imshow(plot_data, aspect='auto', origin='lower', cmap=cmap, norm=norm,
                    extent=[time_indices.min(), time_indices.max(), min(f_start, f_stop), max(f_start, f_stop)])

    # Add labels and title
    plt.xlabel('Time Index')
    plt.ylabel('Frequency (MHz)')
    plt.title(
        f'SETI Data Patch (Index: {index})\n'
        f'Time: [{time_indices.min()}-{time_indices.max()}] | '
        f'Freq: [{min(f_start, f_stop):.6f}-{max(f_start, f_stop):.6f}] MHz'
    )

    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Intensity (log)' if log_scale else 'Intensity')

    plt.tight_layout()
    plt.show()


# Usage example
if __name__ == "__main__":
    # Initialize dataset
    fname = '../data/BLIS692NS/BLIS692NS_data/spliced_blc00010203040506o7o0111213141516o7o0212223242526o7o031323334353637_guppi_58060_26569_HIP17147_0021.gpuspec.0000_chunk30720000_part0.fil'
    # fname = "../data/BLIS692NS/BLIS692NS_data/spliced_blc00010203040506o7o0111213141516o7o0212223242526o7o031323334353637_guppi_58060_26569_HIP17147_0021.gpuspec.0002.fil"
    dataset = SETIWaterFullDataset(file_path=fname, patch_t=16, patch_f=1024, overlap_pct=0.02)

    # Randomly check some patch
    for i in np.random.choice(len(dataset), 10, replace=False):
        plot_dataset_item(dataset, i, log_scale=True)

    # Check middle and edge patch
    # plot_dataset_item(dataset, 0, log_scale=True)  # first patch
    # plot_dataset_item(dataset, len(dataset) // 2, log_scale=True)  # middle patch
    # plot_dataset_item(dataset, -1, log_scale=True)  # last patch
