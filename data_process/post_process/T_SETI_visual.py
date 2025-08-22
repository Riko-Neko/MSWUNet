import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from data_process.post_process.T_SETI import load_seti_dat
from external.Waterfall import Waterfall


# from blimpy import Waterfall


def visualize_specific(wf, hit_row, fchans=1024, foff=None, log_scale=False, save_dir="visual", prefix="hit"):
    """
    Visualize a single TurboSETI hit's frequency window (expanded after), and annotate the target signal range.

    Args:
        wf (Waterfall): A loaded Waterfall object.
        hit_row (pd.Series): A row from DataFrame, containing freq_start, freq_end, Drift_Rate, SNR, etc.
        fchans (int): The number of channels in the visualization window.
        foff (float): Frequency resolution (MHz), used to calculate the expanded width.
        log_scale (bool): Whether to display in logarithmic scale.
        save_dir (str): Directory to save the plots.
        prefix (str): Prefix for the filename.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Signal center and window bandwidth
    f_center = 0.5 * (hit_row["freq_start"] + hit_row["freq_end"])
    half_bw = (fchans / 2) * abs(foff)
    f_start = f_center - half_bw
    f_stop = f_center + half_bw

    # Fetch data
    freqs, data = wf.grab_data(f_start=f_start, f_stop=f_stop)

    # Plot waterfall
    plt.figure(figsize=(10, 6))
    if log_scale:
        plt.imshow(
            np.log10(data + 1e-6),
            aspect="auto",
            extent=[freqs[0], freqs[-1], 0, data.shape[0]],
            origin="lower",
            cmap="viridis"
        )
    else:
        plt.imshow(
            data,
            aspect="auto",
            extent=[freqs[0], freqs[-1], 0, data.shape[0]],
            origin="lower",
            cmap="viridis"
        )

    plt.colorbar(label="Power (a.u.)")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Time bins")
    plt.title(f"SNR={hit_row['SNR']:.1f}, Drift={hit_row['Drift_Rate']:.2f} Hz/s")

    # Save
    fname = f"{prefix}_row{hit_row.name}_freq{f_center:.6f}.png"
    plt.savefig(save_path / fname, dpi=200)
    plt.close()


def visualize_sort(
        wf,
        dat_df,
        n=10,
        by="SNR",
        fchans=256,
        foff=None,
        log_scale=False,
        save_dir="visual_hits"
):
    """
    Sort TurboSETI output dat table and visualize the top n hits.
    If some hits fail to plot, keep going until n successful plots are saved.
    """
    dat_sorted = dat_df.sort_values(by=by, ascending=False)

    success_count = 0
    tried = 0
    for idx, row in dat_sorted.iterrows():
        tried += 1
        try:
            visualize_specific(
                wf,
                row,
                fchans=fchans,
                foff=foff,
                log_scale=log_scale,
                save_dir=save_dir,
                prefix=f"top_{by.lower()}"
            )
            success_count += 1
            if success_count >= n:
                print(f"Finished: {success_count} visualizations saved (tried {tried}).")
                break
        except Exception as e:
            print(f"Failed hit row {row.name}: {e}")

    if success_count < n:
        print(f"Only {success_count}/{n} visualizations succeeded (data exhausted).")


if __name__ == "__main__":
    # setif = 'data_process/test_out/truboseti_blis692ns/spliced_blc00010203040506o7o0111213141516o7o0212223242526o7o031323334353637_guppi_58060_26569_HIP17147_0021/spliced_blc00010203040506o7o0111213141516o7o0212223242526o7o031323334353637_guppi_58060_26569_HIP17147_0021.gpuspec.0002.dat'
    # fname = "../../data/BLIS692NS/BLIS692NS_data/spliced_blc00010203040506o7o0111213141516o7o0212223242526o7o031323334353637_guppi_58060_26569_HIP17147_0021.gpuspec.0002.fil"

    setif = '../test_out/truboseti_blis692ns/spliced_blc00010203040506o7o0111213141516o7o0212223242526o7o031323334353637_guppi_58060_26569_HIP17147_0021/spliced_blc00010203040506o7o0111213141516o7o0212223242526o7o031323334353637_guppi_58060_26569_HIP17147_0021.gpuspec.0000.dat'
    # fname = "../../data/BLIS692NS/BLIS692NS_data/spliced_blc00010203040506o7o0111213141516o7o0212223242526o7o031323334353637_guppi_58060_26569_HIP17147_0021.gpuspec.0000_chunk30720000_part0.fil"
    fname = "../../data/BLIS692NS/BLIS692NS_data/spliced_blc00010203040506o7o0111213141516o7o0212223242526o7o031323334353637_guppi_58060_26569_HIP17147_0021.gpuspec.0000.fil"

    wf = Waterfall(fname, load_data=True)

    dat_df = load_seti_dat(setif)

    visualize_sort(wf, dat_df, n=10, by="SNR", fchans=1024, foff=wf.header["foff"],
                   save_dir=f'visual/{Path(fname).stem}/snr')
    visualize_sort(wf, dat_df, n=10, by="Drift_Rate", fchans=1024, foff=wf.header["foff"],
                   save_dir=f'visual/{Path(fname).stem}/drift')

    # visualize_specific(wf, dat_df.iloc[0], fchans=1024, foff=wf.header["foff"], log_scale=True,
    #                    save_dir=f'visual/{Path(fname).stem}/specific')
