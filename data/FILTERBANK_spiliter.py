import os

import numpy as np
from blimpy import Waterfall  # Assuming the code is in a module named 'waterfall.py'


def slice_filterbank_by_freq(filename, chunk_size, part_index, output_format='fil', max_load=None):
    """
    Slice a large filterbank file by frequency channels into a specified chunk and save it.

    Args:
        filename (str): Path to the input .fil or .h5 file.
        chunk_size (int): Number of frequency channels per chunk (e.g., 100000).
        part_index (int): 0-based index of the chunk to extract and save (e.g., 0 for first chunk).
        output_format (str): 'fil' (default) or 'h5' for output file format.
        max_load (float, optional): Max GB to load per chunk (passed to Waterfall).

    Returns:
        str: Path to the output file.

    Raises:
        ValueError: If part_index is out of range or invalid params.
    """
    # Step 1: Load header only (no data) to get metadata
    wf_header = Waterfall(filename, load_data=False)
    header = wf_header.header
    nchans = header['nchans']
    fch1 = header['fch1']  # First channel freq (MHz)
    foff = header['foff']  # Offset per channel (MHz, can be negative)

    # Calculate total chunks
    total_chunks = int(np.ceil(nchans / chunk_size))
    if part_index < 0 or part_index >= total_chunks:
        raise ValueError(f"part_index {part_index} out of range (0 to {total_chunks - 1})")

    # Step 2: Compute frequency array (handles ascending/descending)
    if foff < 0:
        freqs = np.linspace(fch1, fch1 + (nchans - 1) * foff, nchans)  # Descending
    else:
        freqs = np.linspace(fch1, fch1 + (nchans - 1) * foff, nchans)  # Ascending

    # Step 3: Calculate channel indices for this chunk
    chan_start = part_index * chunk_size
    chan_end = min((part_index + 1) * chunk_size, nchans)

    # Step 4: Determine f_start and f_stop for the chunk
    # - For descending freqs (foff < 0), f_start is higher, f_stop lower
    # - Add small epsilon to f_stop to include the last channel (due to argmin in grab_data)
    epsilon = abs(foff) / 2.0  # Half-channel width to ensure inclusion
    if foff < 0:
        f_start = freqs[chan_start]
        f_stop = freqs[chan_end - 1] - epsilon  # Adjust to include end
    else:
        f_start = freqs[chan_start]
        f_stop = freqs[chan_end - 1] + epsilon

    # Step 5: Load only this frequency slice (full tchans preserved)
    wf_chunk = Waterfall(filename, f_start=f_start, f_stop=f_stop, load_data=True, max_load=max_load)

    # Verify the slice shape (tchans x 1 x chunk_chans)
    expected_chans = chan_end - chan_start
    if wf_chunk.selection_shape[wf_chunk.freq_axis] != expected_chans:
        raise ValueError(f"Slice mismatch: expected {expected_chans} chans, got {wf_chunk.header['nchans']}")

    # Step 6: Generate output filename
    base_name, ext = os.path.splitext(filename)
    output_ext = '.fil' if output_format == 'fil' else '.h5'
    output_filename = f"{base_name}_chunk{chunk_size}_part{part_index}{output_ext}"

    # Step 7: Write the sliced chunk to file
    if output_format == 'fil':
        wf_chunk.write_to_fil(output_filename)
    elif output_format == 'h5':
        wf_chunk.write_to_hdf5(output_filename)
    else:
        raise ValueError("output_format must be 'fil' or 'h5'")

    print(
        f"Saved chunk {part_index} (channels {chan_start}-{chan_end - 1}, freq {f_start:.3f}-{f_stop:.3f} MHz) to {output_filename}")
    return output_filename


if __name__ == '__main__':
    fname = './BLIS692NS/BLIS692NS_data/spliced_blc00010203040506o7o0111213141516o7o0212223242526o7o031323334353637_guppi_58060_26569_HIP17147_0021.gpuspec.0002.fil'
    # Slice into chunks of 100,000 channels, save the 2nd chunk (index 1: channels 100,000-199,999)
    output_file = slice_filterbank_by_freq(fname, chunk_size=102400, part_index=0, output_format='fil',
                                           max_load=3.0)
    # Output: large_data_chunk100000_part1.fil (format identical, but fchans=100,000)
