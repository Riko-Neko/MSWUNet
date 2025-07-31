import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, ifft

from dataset import generate_dataset


def low_pass_filter(spectrum, freqs, cutoff_ratio = 0.1):
    """
    Apply a low-pass filter to the spectrum using Fourier transform.

    Parameters:
        spectrum: Spectrum array (F,)
        freqs: Frequency array (F,)
        cutoff_ratio: Proportion of frequencies to retain (default: 0.1)
    Returns:
        Filtered spectrum
    """
    spectrum_fft = fft(spectrum)
    cutoff_index = int(len(freqs) * cutoff_ratio)
    spectrum_fft[cutoff_index:] = 0
    filtered_spectrum = ifft(spectrum_fft)
    return np.real(filtered_spectrum)


def visualize_sample_spectrum(freqs, clean, noisy, sample_index = 0, data_type = 'synthetic',
                              plot_gradient = False, plot_filtered = False, cutoff_ratio = 0.1, prob = None):
    """
    Visualize a spectrum sample (clean and noisy), with optional gradient and low-pass filtered plots.

    Parameters:
        freqs: Frequency array
        clean: Clean spectrum array (N, F)
        noisy: Noisy spectrum array (N, F)
        sample_index: Index of the sample to visualize
        data_type: 'synthetic' or 'alfalfa'
        plot_gradient: Whether to plot the spectrum gradient (default: False)
        plot_filtered: Whether to plot the low-pass filtered spectrum (default: False)
        cutoff_ratio: Cutoff ratio for the low-pass filter (default: 0.1)
        prob: Binary mask array indicating valid regions (N, F)
    """
    os.makedirs("samples_plot", exist_ok = True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"samples_plot/{data_type}_sample_{sample_index}_{timestamp}.png"

    # Extract sample data
    clean_sample = clean[sample_index]
    noisy_sample = noisy[sample_index]

    # Dynamic subplot creation
    num_plots = 1 + plot_gradient + plot_filtered
    fig_width = 5 * num_plots
    fig, axes = plt.subplots(1, num_plots, figsize = (fig_width, 5), squeeze = False)
    axes = axes.flatten()  # Ensure axes is always a list

    # Main spectrum plot
    ax1 = axes[0]
    ax1.plot(freqs, clean_sample, label = 'Clean Spectrum', linewidth = 2)
    ax1.plot(freqs, noisy_sample, label = 'Noisy Spectrum', alpha = 0.7)

    # Add prob markers if available
    if prob is not None:
        prob_sample = prob[sample_index]
        mask = prob_sample == 1
        ax1.scatter(freqs[mask], clean_sample[mask], color = 'red', s = 10,
                    zorder = 3, label = 'Mask (prob=1)', alpha = 0.6)

    ax1.set_xlabel("Frequency (MHz)")
    ax1.set_ylabel("Flux")
    ax1.set_title("Original Spectra")
    ax1.legend()

    current_axis = 1

    # Plot gradient if requested
    if plot_gradient:
        clean_grad = np.gradient(clean_sample, freqs)
        noisy_grad = np.gradient(noisy_sample, freqs)
        ax = axes[current_axis]
        ax.plot(freqs, clean_grad, label = 'Clean Gradient', linewidth = 2)
        ax.plot(freqs, noisy_grad, label = 'Noisy Gradient', alpha = 0.7)
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Gradient")
        ax.set_title("Spectrum Gradient")
        ax.legend()
        current_axis += 1

    # Plot filtered spectrum if requested
    if plot_filtered:
        filtered_noisy = low_pass_filter(noisy_sample, freqs, cutoff_ratio = cutoff_ratio)
        ax = axes[current_axis]
        ax.plot(freqs, noisy_sample, label = 'Noisy Spectrum', alpha = 0.7)
        ax.plot(freqs, filtered_noisy, label = 'Filtered Spectrum', color = 'green')
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Flux")
        ax.set_title("Low-pass Filtered Spectrum")
        ax.legend()

    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
    print(f"Saved plot to: {fname}")


if __name__ == '__main__':
    freqs = np.linspace(1419.205, 1421.605, 1024)
    clean, noisy, prob = generate_dataset(data_type = 'synthetic', freqs = freqs, num_samples = 100)

    for i in range(50):
        # Plot with additional plots
        visualize_sample_spectrum(freqs, clean, noisy, sample_index = i, data_type = 'synthetic', plot_gradient = False,
                                  plot_filtered = False, cutoff_ratio = 0.05, prob = prob)
