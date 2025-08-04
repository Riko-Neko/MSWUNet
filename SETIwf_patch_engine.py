import torch
from blimpy import Waterfall
from torch.utils.data import Dataset


class SETIDataset(Dataset):
    def __init__(self, file_path, patch_t, patch_f, overlap_pct=0.05):
        self.obs = Waterfall(file_path, load_data=False)
        # Assuming obs.data_shape is (tchans, n_pols, fchans)
        self.tchans = self.obs.data_shape[0]
        self.fchans = self.obs.data_shape[2]
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
        data, patch_freqs = self.obs.grab_data(f_start, f_stop, start_t, end_t)
        # Assuming data is (patch_t, n_pols, patch_f)，Choose the first polarization
        if data.shape[1] == 1:
            data = data[:, 0, :]
        else:
            raise ValueError("Multiple polarizations not handled")

        # To tensor
        patch_tensor = torch.from_numpy(data).float().unsqueeze(0)  # (1, patch_t, patch_f)
        return patch_tensor, (start_t, start_f)
