import platform

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from blimpy import Waterfall
from torch.utils.data import Dataset

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
    def __init__(self, file_path, patch_t, patch_f, overlap_pct=0.05):
        self.obs = Waterfall(file_path, load_data=True)  # Load is a MUST
        # obs.data_shape is (tchans, n_pols, fchans)
        self.tchans = self.obs.selection_shape[0]
        self.fchans = self.obs.selection_shape[2]
        self.freqs = self.obs.get_freqs()
        assert patch_t <= self.tchans, "patch_t larger than tchans"
        assert patch_f <= self.fchans, "patch_f larger than fchans"

        overlap_t = round(patch_t * overlap_pct)
        stride_t = patch_t - overlap_t
        overlap_f = round(patch_f * overlap_pct)
        stride_f = patch_f - overlap_f

        # Calculate time dimension's starting index
        start_t_list = list(range(0, self.tchans - patch_t + 1, stride_t))
        if start_t_list and start_t_list[-1] + patch_t < self.tchans:
            start_t_list.append(self.tchans - patch_t)

        # Calculate frequency dimension's starting index
        start_f_list = list(range(0, self.fchans - patch_f + 1, stride_f))
        if start_f_list and start_f_list[-1] + patch_f < self.fchans:
            start_f_list.append(self.fchans - patch_f)

        self.start_t_list = start_t_list
        self.start_f_list = start_f_list
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

        # Calculate frequency range
        if self.freqs[0] < self.freqs[-1]:  # ⬆️
            f_start = self.freqs[start_f]
            f_stop = self.freqs[end_f - 1]
        else:  # ⬇️
            f_start = self.freqs[end_f - 1]
            f_stop = self.freqs[start_f]

        # Load data
        patch_freqs, data = self.obs.grab_data(f_start, f_stop, start_t, end_t)

        # Normalize
        mean = np.mean(data)
        std = np.std(data)
        if std < 1e-10:
            std = 1.0
        data = (data - mean) / std

        # To tensor
        patch_tensor = torch.from_numpy(data).float().unsqueeze(0)  # (1, patch_t, patch_f)

        return patch_tensor, (start_t, start_f)

    def get_patch(self, row, col):
        index = row * len(self.start_f_list) + col
        patch_tensor, (start_t, start_f) = self.__getitem__(index)
        end_t = start_t + self.patch_t
        end_f = start_f + self.patch_f

        if self.freqs[0] < self.freqs[-1]:  # Ascending
            f_min = self.freqs[start_f]
            f_max = self.freqs[end_f - 1]
        else:  # Descending
            f_min = self.freqs[end_f - 1]
            f_max = self.freqs[start_f]

        freq_range = (f_min, f_max)
        time_range = (start_t, end_t)

        return patch_tensor, freq_range, time_range


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
    dataset = SETIWaterFullDataset(
        file_path="../data/BLIS692NS/BLIS692NS_data/spliced_blc00010203040506o7o0111213141516o7o0212223242526o7o031323334353637_guppi_58060_26569_HIP17147_0021.gpuspec.0002.fil",
        # Replace with actual file path
        patch_t=144,
        patch_f=1024,
        overlap_pct=0.02
    )

    # Randomly check some patch
    for i in np.random.choice(len(dataset), 10, replace=False):
        plot_dataset_item(dataset, i, log_scale=True)

    # Check middle and edge patch
    # plot_dataset_item(dataset, 0, log_scale=True)  # first patch
    # plot_dataset_item(dataset, len(dataset) // 2, log_scale=True)  # middle patch
    # plot_dataset_item(dataset, -1, log_scale=True)  # last patch
