import argparse
import os

import numpy as np
from blimpy import Waterfall


def slice_filterbank_by_freq(filename, chunk_size=None, part_index=None, output_format='fil', max_load=None,
                             f_start=None, f_stop=None, output_dir=None):
    """
    Slice a large filterbank file by frequency channels or by explicit freq range, then save it to specified directory.
    """
    # Step 1: Load header only (no data) to get metadata
    wf_header = Waterfall(filename, load_data=False)
    header = wf_header.header
    nchans = header['nchans']
    fch1 = header['fch1']
    foff = header['foff']

    # Step 2: Handle explicit frequency range mode
    if f_start is not None and f_stop is not None:
        print(f"[Info] Using explicit frequency range mode: {f_start}-{f_stop} MHz")
        wf_chunk = Waterfall(filename, f_start=f_start, f_stop=f_stop,
                             load_data=True, max_load=max_load)
        chan_start, chan_end = None, None
    else:
        if chunk_size is None or part_index is None:
            raise ValueError("Must specify either (f_start, f_stop) or (chunk_size, part_index)")

        total_chunks = int(np.ceil(nchans / chunk_size))
        if part_index < 0 or part_index >= total_chunks:
            raise ValueError(f"part_index {part_index} out of range (0 to {total_chunks - 1})")

        # Compute freqs
        freqs = np.linspace(fch1, fch1 + (nchans - 1) * foff, nchans)
        chan_start = part_index * chunk_size
        chan_end = min((part_index + 1) * chunk_size, nchans)

        # Determine f_start/f_stop
        epsilon = abs(foff) / 2.0
        if foff < 0:
            f_start = freqs[chan_start]
            f_stop = freqs[chan_end - 1] - epsilon
        else:
            f_start = freqs[chan_start]
            f_stop = freqs[chan_end - 1] + epsilon

        wf_chunk = Waterfall(filename, f_start=f_start, f_stop=f_stop,
                             load_data=True, max_load=max_load)

        expected_chans = chan_end - chan_start
        if wf_chunk.selection_shape[wf_chunk.freq_axis] != expected_chans:
            raise ValueError(f"Slice mismatch: expected {expected_chans} chans, got {wf_chunk.header['nchans']}")

    # Step 3: Output filename
    base_name, ext = os.path.splitext(os.path.basename(filename))
    output_ext = '.fil' if output_format == 'fil' else '.h5'

    if f_start is not None and f_stop is not None:
        output_filename = f"{base_name}_f{f_start:.2f}-{f_stop:.2f}{output_ext}"
    else:
        output_filename = f"{base_name}_chunk{chunk_size}_part{part_index}{output_ext}"

    # Step 4: Determine output path
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
    else:
        output_path = os.path.join(os.path.dirname(filename), output_filename)

    # Step 5: Write
    if output_format == 'fil':
        wf_chunk.write_to_fil(output_path)
    elif output_format == 'h5':
        wf_chunk.write_to_hdf5(output_path)
    else:
        raise ValueError("output_format must be 'fil' or 'h5'")

    print(f"Saved to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Slice filterbank file by frequency channels or explicit range")
    parser.add_argument("filename", help="Path to input .fil/.h5 file")

    # Chunk mode args
    parser.add_argument("--chunk_size", type=int, help="Number of frequency channels per chunk")
    parser.add_argument("--part_index", type=int, help="0-based index of the chunk to extract")

    # Frequency mode args
    parser.add_argument("--f_start", type=float, help="Start frequency in MHz")
    parser.add_argument("--f_stop", type=float, help="Stop frequency in MHz")

    parser.add_argument("--output_format", choices=["fil", "h5"], default="fil", help="Output file format")
    parser.add_argument("--max_load", type=float, help="Max GB to load per chunk")
    parser.add_argument("--output_dir", type=str,
                        help="Output directory for sliced files (default: same as input file)")

    args = parser.parse_args()

    slice_filterbank_by_freq(
        filename=args.filename,
        chunk_size=args.chunk_size,
        part_index=args.part_index,
        output_format=args.output_format,
        max_load=args.max_load,
        f_start=args.f_start,
        f_stop=args.f_stop,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
    """
    Example usage:
    # Slice into chunks of 100,000 channels, save the 2nd chunk (index 1: channels 100,000-199,999)
    output_file = slice_filterbank_by_freq(fname, chunk_size=10240000, part_index=0, output_format='fil', max_load=3.0)
    # Output: large_data_chunk100000_part1.fil (format identical, but fchans=100,000)
    
    fname = '/media/rikoneko/c9f14f4a-975f-47f0-95c8-bb4804224489/yy/Kepler-438_M01_pol2.fil'
    - python FILTERBANK_spiliter.py --f_start 1120 --f_stop 1150
    
    fname = './BLIS692NS/BLIS692NS_data/spliced_blc00010203040506o7o0111213141516o7o0212223242526o7o031323334353637_guppi_58060_26569_HIP17147_0021.gpuspec.0002.fil'
    
    # Kepler-438_M01:
    python FILTERBANK_spiliter.py --f_start 1139.5800449 --f_stop 1142.3103847 --output_dir ./33exoplanets/ /data/Raid0/obs_data/33exoplanets/yy/Kepler-438_M01_pol2.fil
    python FILTERBANK_spiliter.py --f_start 1139.5800449 --f_stop 1142.3103847 --output_dir ./33exoplanets/ /data/Raid0/obs_data/33exoplanets/xx/Kepler-438_M01_pol1.fil
    
    # HD-180617_M04:
    python FILTERBANK_spiliter.py --f_start 1404.0 --f_stop 1404.1 --output_dir ./33exoplanets/ /data/Raid0/obs_data/33exoplanets/xx/HD-180617_M04_pol1.fil
    python FILTERBANK_spiliter.py --f_start 1404.0 --f_stop 1404.1 --output_dir ./33exoplanets/ /data/Raid0/obs_data/33exoplanets/yy/HD-180617_M04_pol2.fil

    for many files in linux:

for f in /data/Raid0/obs_data/33exoplanets/*/Gliese-486_M*; do
  python FILTERBANK_spiliter.py \
    --f_start 1152.1435 \
    --f_stop 1152.1450 \
    --output_dir ./33exoplanets/ \
    "$f"
done

for f in /data/Raid0/obs_data/33exoplanets/*/Kepler-438_M*; do
  python FILTERBANK_spiliter.py \
    --f_start 1319.99887 \
    --f_stop 1320.00087 \
    --output_dir ./33exoplanets/ \
    "$f"
done

for f in /data/Raid0/obs_data/33exoplanets/*/K2-155_M*; do
  python FILTERBANK_spiliter.py \
    --f_start 1148.415 \
    --f_stop 1148.418 \
    --output_dir ./33exoplanets/ \
    "$f"
done

for f in /data/Raid0/obs_data/33exoplanets/*/*M{08,10,12,14,16,18}*; do
  python FILTERBANK_spiliter.py \
    --f_start 1278 \
    --f_stop 1380 \
    --output_dir ./33exoplanets/bg/clean \
    "$f"
done

for f in /data/Raid0/obs_data/33exoplanets/*/*M{08,10,12,14,16,18}*; do
  python FILTERBANK_spiliter.py --f_start 1217.6 --f_stop 1237.6 --output_dir ./33exoplanets/bg/pollution "$f"
  python FILTERBANK_spiliter.py --f_start 1258.29 --f_stop 1278.75 --output_dir ./33exoplanets/bg/pollution "$f"
done
    """
