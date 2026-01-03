# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FAST SETI Visual Checker (XX / YY / I)

Inputs:
- One target CSV (e.g., total_*.csv or candidates CSV) containing at least:
  group_id, freq_start, freq_end
  and optionally: DriftRate, SNR, class_id, confidence, csv_id

- Two directories:
  XX_DIR: contains raw filterbank files for XX polarization
  YY_DIR: contains raw filterbank files for YY polarization

File matching rule:
- For each event, locate *{group_id}_M01* in XX_DIR and YY_DIR (first match wins).

Plots:
Window-1: width = 2 * signal_width (|freq_end - freq_start|), centered at mid freq, ~4:3
Window-2: fixed width = 0.00096 MHz, centered at mid freq, ~4:3 (only if --detailed)
Window-3: width = 0.008 MHz (8 kHz), centered at mid freq, ~30:1 long aspect (only if --detailed)

For each window, generate 3 images:
- XX
- YY
- I = XX + YY (total intensity)

Outputs (grouped):
./filter_workflow/candidates/visual/<timestamp>/
  XX/1/, (XX/2/, XX/3/ if detailed)
  YY/1/, ...
  I/1/, ...
  event_meta.csv   (NO metadata on images, all metadata stored here)
"""

import argparse
import glob
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from blimpy import Waterfall

# ====== Defaults (edit here) ======
DEFAULT_TARGET_CSV = "../filter_workflow/tmp/1.csv"
# DEFAULT_TARGET_CSV = "../filter_workflow/tmp/2.csv"
# DEFAULT_TARGET_CSV = "../filter_workflow/tmp/3.csv"
DEFAULT_YY_DIR = "/data/Raid0/obs_data/33exoplanets/yy/"
DEFAULT_XX_DIR = "/data/Raid0/obs_data/33exoplanets/xx/"
DEFAULT_OUTPUT_ROOT = "./filter_workflow/candidates/visual"

# ====== Window configs ======
FIXED_W2_MHZ = 0.00096
W3_MHZ = 0.008  # 8 kHz

# figure aspect / sizes
FIGSIZE_43 = (12, 9)  # ~4:3
FIGSIZE_30_1 = (30, 1.2)  # ~30:1 (long time axis)

# save config
DEFAULT_DPI = 200
DEFAULT_FMT = "png"


# -----------------------------
# Utilities
# -----------------------------
def ensure_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def make_timestamp_dir(root: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = root / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def find_m01_file(folder: Path, group_id: str) -> Optional[Path]:
    """
    Find the first file matching *{group_id}_M01* with extensions .fil or .h5.
    """
    patterns = [
        str(folder / f"*{group_id}_M01*.fil"),
        str(folder / f"*{group_id}_M01*.h5"),
    ]
    matches: List[str] = []
    for pat in patterns:
        matches.extend(glob.glob(pat))
    matches = sorted(set(matches))
    if not matches:
        return None
    return Path(matches[0]).resolve()


def compute_windows(freq_start_mhz: float, freq_end_mhz: float) -> Dict[int, Tuple[float, float, float, float]]:
    """
    Return dict: window_id -> (center, f_start, f_stop, width)
    """
    f0 = 0.5 * (freq_start_mhz + freq_end_mhz)
    w_sig = abs(freq_end_mhz - freq_start_mhz)

    # Window-1: 2x signal width, but avoid zero-width edge case
    w1 = max(2.0 * w_sig, 1e-6)  # 1e-6 MHz = 1 Hz minimal safeguard
    w2 = FIXED_W2_MHZ
    w3 = W3_MHZ

    def bounds(center: float, width: float) -> Tuple[float, float]:
        return center - 0.5 * width, center + 0.5 * width

    f1s, f1e = bounds(f0, w1)
    f2s, f2e = bounds(f0, w2)
    f3s, f3e = bounds(f0, w3)

    return {
        1: (f0, f1s, f1e, w1),
        2: (f0, f2s, f2e, w2),
        3: (f0, f3s, f3e, w3),
    }


def load_waterfall_slice(fname: Path, f_start: float, f_stop: float) -> Waterfall:
    """
    Use blimpy Waterfall to load a narrow frequency slice.
    f_start/f_stop are in MHz.
    """
    return Waterfall(str(fname), f_start=float(f_start), f_stop=float(f_stop))


def extract_2d_data(wf: Waterfall) -> np.ndarray:
    """
    Try to extract a (time, freq) 2D array from blimpy Waterfall.
    """
    data = wf.data
    arr = np.array(data)
    arr = np.squeeze(arr)

    # Common cases:
    # - (time, freq)
    # - (time, 1, freq)
    # - (1, time, freq)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        # pick the first polarization axis if present
        # try to reduce to (time, freq)
        # heuristic: choose the axis of size 1
        for ax in range(3):
            if arr.shape[ax] == 1:
                arr2 = np.squeeze(arr, axis=ax)
                if arr2.ndim == 2:
                    return arr2
        # fallback: take first slice
        return np.squeeze(arr[0])
    raise ValueError(f"Unexpected waterfall data shape: {arr.shape}")


def save_image_from_array(
        arr: np.ndarray,
        save_path: Path,
        figsize: Tuple[float, float],
        dpi: int,
        fmt: str,
):
    """
    Save image without axes/labels (metadata stored separately).
    """
    plt.figure(figsize=figsize)
    # You can adjust scaling if desired; here we keep default linear display.
    plt.imshow(arr, aspect="auto", origin="lower")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(save_path.with_suffix(f".{fmt}"), dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Visual checker: render XX/YY/I images around event frequencies.")
    ap.add_argument("--target_csv", type=str, default=DEFAULT_TARGET_CSV, help="Path to target events CSV.")
    ap.add_argument("--xx_dir", type=str, default=DEFAULT_XX_DIR, help="Directory containing XX filterbank files.")
    ap.add_argument("--yy_dir", type=str, default=DEFAULT_YY_DIR, help="Directory containing YY filterbank files.")
    ap.add_argument("--out_root", type=str, default=DEFAULT_OUTPUT_ROOT, help="Output root directory.")
    ap.add_argument("--detailed", action="store_true", help="If set, also render window-2 and window-3.")
    ap.add_argument("--dpi", type=int, default=DEFAULT_DPI, help="Output image DPI.")
    ap.add_argument("--fmt", type=str, default=DEFAULT_FMT, choices=["png", "jpg", "jpeg", "pdf"], help="Image format.")
    ap.add_argument("--max_events", type=int, default=0, help="If >0, only process first N events (debug).")
    args = ap.parse_args()

    target_csv = Path(args.target_csv).expanduser().resolve()
    xx_dir = Path(args.xx_dir).expanduser().resolve()
    yy_dir = Path(args.yy_dir).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()

    if not target_csv.exists():
        print(f"[\033[31mError\033[0m] Target CSV not found: {target_csv}")
        sys.exit(1)
    if not xx_dir.exists():
        print(f"[\033[31mError\033[0m] XX_DIR not found: {xx_dir}")
        sys.exit(1)
    if not yy_dir.exists():
        print(f"[\033[31mError\033[0m] YY_DIR not found: {yy_dir}")
        sys.exit(1)

    df = pd.read_csv(target_csv)

    required = ["group_id", "freq_start", "freq_end"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[\033[31mError\033[0m] Missing required columns: {missing}")
        print(f"[\033[31mError\033[0m] Present columns: {list(df.columns)}")
        sys.exit(1)

    # normalize numeric columns
    df["freq_start"] = ensure_numeric(df["freq_start"])
    df["freq_end"] = ensure_numeric(df["freq_end"])

    # drop invalid rows
    before = len(df)
    df = df.dropna(subset=["group_id", "freq_start", "freq_end"]).copy()
    after = len(df)
    if after != before:
        print(f"[\033[32mInfo\033[0m] Dropped {before - after} rows due to missing group_id/freq_start/freq_end.")

    if args.max_events and args.max_events > 0:
        df = df.head(args.max_events).copy()
        print(f"[\033[32mInfo\033[0m] Debug mode: processing first {len(df)} events.")

    # output structure
    ts_dir = make_timestamp_dir(out_root)
    print("\n==============================")
    print("[\033[32mInfo\033[0m] VISUAL CHECK")
    print("==============================")
    print(f"[\033[32mInfo\033[0m] Target CSV: {target_csv}")
    print(f"[\033[32mInfo\033[0m] XX_DIR    : {xx_dir}")
    print(f"[\033[32mInfo\033[0m] YY_DIR    : {yy_dir}")
    print(f"[\033[32mInfo\033[0m] Output dir: {ts_dir}")
    print(f"[\033[32mInfo\033[0m] Detailed  : {args.detailed}")
    print(f"[\033[32mInfo\033[0m] Format/DPI: {args.fmt}/{args.dpi}")

    pols = ["XX", "YY", "I"]
    windows = [1] + ([2, 3] if args.detailed else [])

    # create folders
    for pol in pols:
        for w in windows:
            (ts_dir / pol / str(w)).mkdir(parents=True, exist_ok=True)

    # Prepare meta collector (no metadata on images)
    meta_rows: List[Dict] = []

    # Process events
    for i, row in df.reset_index(drop=True).iterrows():
        group_id = str(row["group_id"])
        f_start = float(row["freq_start"])
        f_end = float(row["freq_end"])
        f0 = 0.5 * (f_start + f_end)

        xx_file = find_m01_file(xx_dir, group_id)
        yy_file = find_m01_file(yy_dir, group_id)

        if xx_file is None or yy_file is None:
            print(
                f"[\033[33mWarn\033[0m] Missing XX/YY file for group_id={group_id} (XX={xx_file}, YY={yy_file}). Skipped.")
            continue

        win_map = compute_windows(f_start, f_end)

        # Extract metadata fields (if present)
        meta = {
            "event_idx": i,
            "group_id": group_id,
            "file_xx": str(xx_file),
            "file_yy": str(yy_file),
            "freq_start_mhz": f_start,
            "freq_end_mhz": f_end,
            "freq_center_mhz": f0,
            "csv_source": row["csv_id"] if "csv_id" in df.columns else "",
            "DriftRate": row["DriftRate"] if "DriftRate" in df.columns else "",
            "SNR": row["SNR"] if "SNR" in df.columns else "",
            "class_id": row["class_id"] if "class_id" in df.columns else "",
            "confidence": row["confidence"] if "confidence" in df.columns else "",
        }

        # For each window, render XX/YY/I
        for w in windows:
            _, ws, we, ww = win_map[w]

            # choose figure size
            figsize = FIGSIZE_43 if w in (1, 2) else FIGSIZE_30_1

            # Load slices
            try:
                wf_xx = load_waterfall_slice(xx_file, ws, we)
                wf_yy = load_waterfall_slice(yy_file, ws, we)
                arr_xx = extract_2d_data(wf_xx)
                arr_yy = extract_2d_data(wf_yy)

                # Make sure shapes align for I
                # If they differ slightly, crop to the common minimum shape.
                t = min(arr_xx.shape[0], arr_yy.shape[0])
                f = min(arr_xx.shape[1], arr_yy.shape[1])
                arr_xx = arr_xx[:t, :f]
                arr_yy = arr_yy[:t, :f]
                arr_i = arr_xx + arr_yy

            except Exception as e:
                print(f"[\033[33mWarn\033[0m] Failed to load slice for group_id={group_id}, window={w}: {e}")
                continue

            # Build basename (no metadata on image)
            base = f"{group_id}_M01_evt{i:06d}_fc{f0:.6f}MHz_w{w}_bw{ww:.6f}MHz"

            # Save images
            save_image_from_array(arr_xx, ts_dir / "XX" / str(w) / base, figsize, args.dpi, args.fmt)
            save_image_from_array(arr_yy, ts_dir / "YY" / str(w) / base, figsize, args.dpi, args.fmt)
            save_image_from_array(arr_i, ts_dir / "I" / str(w) / base, figsize, args.dpi, args.fmt)

        # record meta once per event (covers all windows)
        meta_rows.append(meta)

        if (len(meta_rows) % 19) == 0:
            print(f"[\033[32mInfo\033[0m] Processed {len(meta_rows)}/{len(df)} events...")

    # Write metadata table (no metadata on images)
    meta_df = pd.DataFrame(meta_rows)
    meta_path = ts_dir / "event_meta.csv"
    meta_df.to_csv(meta_path, index=False)
    print(f"\n[\033[32mInfo\033[0m] Metadata written: {meta_path}")
    print(f"[\033[32mInfo\033[0m] Done!")


if __name__ == "__main__":
    main()
