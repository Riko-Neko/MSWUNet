#!/usr/bin/env python3
import sys

import matplotlib
from blimpy import Waterfall

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

"""
Kepler-438_M01 yy:
python FILTERBANK_checker.py /media/rikoneko/c9f14f4a-975f-47f0-95c8-bb4804224489/yy/Kepler-438_M01_pol2.fil 1140.6035 1140.6045
python data/FILTERBANK_checker.py data/33exoplanets/Kepler-438_M01_pol2_f1120.00-1150.00.fil 1124.9355 1124.9375
python data/FILTERBANK_checker.py data/33exoplanets/Kepler-438_M01_pol2_f1120.00-1150.00.fil 1130.4087 1130.430
python data/FILTERBANK_checker.py data/33exoplanets/Kepler-438_M01_pol2_f1120.00-1150.00.fil 1125.0154 1125.017
python data/FILTERBANK_checker.py data/33exoplanets/Kepler-438_M01_pol2_f1120.00-1150.00.fil 1125.019 1125.023

HD-180617_M04 xx:
python FILTERBANK_checker.py /media/rikoneko/4c8904cd-457e-43a8-81d8-e3f5c0e60128/xx/HD-180617_M04_pol1.fil 1404.0496 1404.0504
python data/FILTERBANK_checker.py ./data/33exoplanets/HD-180617_M04_pol1_f1400.00-1410.00.fil 1404.0496 1404.0573
python data/FILTERBANK_checker.py ./data/33exoplanets/HD-180617_M04_pol1_f1400.00-1410.00.fil 1406.9505 1406.9515
python data/FILTERBANK_checker.py ./data/33exoplanets/HD-180617_M04_pol1_f1400.00-1410.00.fil 1403.866 1403.867

HD-218566_M01 yy:
python FILTERBANK_checker.py /media/rikoneko/c9f14f4a-975f-47f0-95c8-bb4804224489/yy/HD-218566_M01_pol2.fil 1140.6035 1140.6045
"""
