from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pipeline.patch_engine import SETIWaterFullDataset


def plot_patch(patch_tensor, freq_range, time_range, row_idx, col_idx, drift=None, snr=None,
               save_dir="visual/specific", prefix="patch", log_scale=False):
    """
    Plot a single patch with given metadata.

    Args:
        patch_tensor: Torch tensor patch (C, T, F) or (T, F)
        freq_range: (freq_min, freq_max)
        time_range: (time_start, time_end)
        row_idx, col_idx: Patch indices
        drift: DriftRate value (optional)
        snr: SNR value (optional)
        save_dir: Directory to save plots
        prefix: Prefix for filename
        log_scale: Use logarithmic color scale
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # prepare data
    patch_np = patch_tensor.squeeze(0).numpy()  # (T, F)
    freq_min, freq_max = freq_range
    time_start, time_end = time_range

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    if log_scale:
        # avoid log(0)
        patch_plot = np.log10(np.abs(patch_np) + 1e-6)
    else:
        patch_plot = patch_np

    im = ax.imshow(
        patch_plot,
        aspect="auto",
        origin="lower",
        extent=[freq_min, freq_max, time_start, time_end],
        cmap="viridis"
    )

    # labels and title
    ax.set_ylabel("Time (seconds)", fontsize=12)
    ax.set_xlabel("Frequency (MHz)", fontsize=12)

    title = f"Patch (Row {row_idx}, Col {col_idx})"
    if drift is not None and snr is not None:
        title += f"\nDriftRate: {drift:.6f}, SNR: {snr:.2f}"
    ax.set_title(title, fontsize=14)

    # color bar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Intensity (log)" if log_scale else "Intensity", fontsize=12)

    plt.tight_layout()

    filename = save_path / f"{prefix}_row{row_idx}_col{col_idx}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved patch visualization to {filename}")


def visualize_sort(ml_df, dataset, n=10, by='DriftRate', log_scale=False, save_dir='visual'):
    """
    Visualize the top n unique patches sorted by a metric from the ML DataFrame.
    Ensures that duplicate (row, col) patches are not counted in n.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    sorted_df = ml_df.sort_values(by=by, ascending=False)

    unique_df = sorted_df.drop_duplicates(subset=['cell_row', 'cell_col'])
    top_n = unique_df.head(n)

    for _, row in top_n.iterrows():
        try:
            row_idx, col_idx = int(row['cell_row']), int(row['cell_col'])

            patch_tensor, freq_range, time_range = dataset.get_patch(row_idx, col_idx)

            plot_patch(
                patch_tensor,
                freq_range,
                time_range,
                row_idx,
                col_idx,
                drift=row['DriftRate'],
                snr=row['SNR'],
                save_dir=save_path,
                prefix=f"top_{by.lower()}",
                log_scale=log_scale
            )
        except Exception as e:
            print(f"Error visualizing patch row {row['cell_row']}, col {row['cell_col']}: {e}")


def visualize_specific(dataset, row_idx, col_idx, drift=None, snr=None, log_scale=False,
                       save_dir="visual/specific"):
    """
    Visualize a single specific patch.
    """
    patch_tensor, freq_range, time_range = dataset.get_patch(row_idx, col_idx)
    plot_patch(
        patch_tensor,
        freq_range,
        time_range,
        row_idx,
        col_idx,
        drift=drift,
        snr=snr,
        save_dir=save_dir,
        prefix="specific",
        log_scale=log_scale
    )


# Usage example
if __name__ == "__main__":
    from ML import load_ML_dat

    # mlf = '../../pipeline/log/spliced_blc00010203040506o7o0111213141516o7o0212223242526o7o031323334353637_guppi_58060_26569_HIP17147_0021.gpuspec.0002/hits_20250818_181041.dat'
    # fname = "../../data/BLIS692NS/BLIS692NS_data/spliced_blc00010203040506o7o0111213141516o7o0212223242526o7o031323334353637_guppi_58060_26569_HIP17147_0021.gpuspec.0002.fil"

    mlf = '../../pipeline/log/spliced_blc00010203040506o7o0111213141516o7o0212223242526o7o031323334353637_guppi_58060_26569_HIP17147_0021.gpuspec.0000_chunk30720000_part0/hits_20250819_135738.dat'
    fname = "../../data/BLIS692NS/BLIS692NS_data/spliced_blc00010203040506o7o0111213141516o7o0212223242526o7o031323334353637_guppi_58060_26569_HIP17147_0021.gpuspec.0000_chunk30720000_part0.fil"

    ml_df = load_ML_dat(mlf, mode='detection')
    dataset = SETIWaterFullDataset(file_path=fname, patch_t=116, patch_f=1024, overlap_pct=0.02)

    # visualize top-N
    # visualize_sort(ml_df, dataset, n=50, by='DriftRate', log_scale=False, save_dir=f'visual/{Path(fname).stem}/drift')
    visualize_sort(ml_df, dataset, n=50, by='SNR', log_scale=False, save_dir=f'visual/{Path(fname).stem}/snr')

    # visualize a single patch
    # visualize_specific(dataset, row_idx=0, col_idx=26290, drift=0.0123, snr=45.6, log_scale=False,save_dir=f'visual/{Path(fname).stem}/specific')
