#!/usr/bin/env python3
import sys
from blimpy import Waterfall
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt


def print_info(wf: Waterfall):
    print("=== Waterfall.info() Output ===")
    try:
        wf.info()
    except Exception as e:
        print(f"Error calling wf.info(): {e}")

    print("\n=== wf.header dict ===")
    hdr = getattr(wf, 'header', None)
    if hdr is None:
        print("No wf.header attribute found.")
    else:
        for k, v in hdr.items():
            print(f"{k}: {v}")


def main():
    if len(sys.argv) < 2 or len(sys.argv) > 4:
        print("Usage: python FILTERBANK_checker.py <filterbank_file.fil/.h5> [f_start] [f_stop]")
        print("  f_start, f_stop: optional floats, frequency range in MHz")
        sys.exit(1)

    fname = sys.argv[1]
    f_start = float(sys.argv[2]) if len(sys.argv) >= 3 else None
    f_stop = float(sys.argv[3]) if len(sys.argv) == 4 else None

    try:
        wf = Waterfall(fname, f_start=f_start, f_stop=f_stop)
    except Exception as e:
        print(f"Error opening file {fname}: {e}")
        sys.exit(1)

    print_info(wf)
    try:
        plt.figure(figsize=(10, 6))
        wf.plot_waterfall(f_start=f_start, f_stop=f_stop)
        plt.show()
    except Exception as e:
        print(f"Error plotting waterfall for {fname}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()