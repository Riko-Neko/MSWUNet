# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FAST SETI Visual Checker (XX / YY / I) + Multi-Beam Support

Inputs:
- One target CSV (e.g., total_*.csv or candidates CSV) containing at least:
  group_id, freq_start, freq_end
  and optionally: DriftRate, SNR, class_id, confidence, csv_id

- Two directories:
  XX_DIR: contains raw filterbank files for XX polarization
  YY_DIR: contains raw filterbank files for YY polarization

File matching rule:
- Default: only beam M01
- If --multi_beam is enabled: render M01–M19 (inclusive)
- For each event+beam, locate *{group_id}_Mxx* in XX_DIR and YY_DIR (first match wins).

Plots:
Window-fit_width: width = 2 * signal_width (|freq_end - freq_start|), centered at mid freq, ~4:3
Window-fixed_width: fixed width = 0.00096 MHz, centered at mid freq, ~4:3 (only if --detailed)
Window-8k: width = 0.008 MHz (8 kHz), centered at mid freq, ~30:1 long aspect (only if --detailed)

Rendering rule (simplified):
- fit_width:
    - default: render XX, YY, I
    - if polarization plotting is disabled: render ONLY I
- fixed_width and 8k (only if --detailed): render ONLY I (no XX/YY)

Outputs (grouped):
./filter_workflow/candidates/visual/<timestamp>/
  XX/fit_width/                 (only if polarization plotting enabled)
  YY/fit_width/                 (only if polarization plotting enabled)
  I/fit_width/, (I/fixed_width/, I/8k/ if detailed)
  event_meta.csv   (NO metadata on images, all metadata stored here)

Multi-beam output:
- Images are distinguished by filename containing "_Mxx_".
- Metadata table contains one row per (event, beam).
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
from matplotlib.ticker import MultipleLocator, MaxNLocator, FormatStrFormatter

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from config.settings import Settings

# ====== Progress bar (new) ======
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None

if Settings.PROD:
    import matplotlib as mpl

    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['font.size'] = 20
    mpl.rcParams['font.weight'] = 'semibold'
    mpl.rcParams['axes.titleweight'] = 'bold'
    mpl.rcParams['axes.labelweight'] = 'bold'


def progress(iterable, total: int, desc: str):
    """
    Wrapper to provide a progress bar.
    - If tqdm is available: use a real progress bar.
    - Else: fall back to lightweight console progress updates.
    """
    if tqdm is not None:
        return tqdm(iterable, total=total, desc=desc, unit="evt")
    else:
        def _gen():
            k = 0
            for item in iterable:
                k += 1
                if k == 1 or k % 5 == 0 or k == total:
                    endc = "\n" if k == total else "\r"
                    print(f"[\033[32mInfo\033[0m] {desc}: {k}/{total}", end=endc, flush=True)
                yield item

        return _gen()


# ====== Defaults configs ======
# DEFAULT_TARGET_CSV = "./data_process/post_process/filter_workflow/candidates/20260108_165059_veto_dnu7.5Hz_x3/on_candidates.csv"
DEFAULT_TARGET_CSV = ROOT / "./data_process/post_process/filter_workflow/tmp/k215519b.csv"
# DEFAULT_TARGET_CSV = ROOT / "./data_process/post_process/filter_workflow/tmp/fp2.csv"
# DEFAULT_YY_DIR = "/data/Raid0/obs_data/33exoplanets/yy/"
# DEFAULT_XX_DIR = "/data/Raid0/obs_data/33exoplanets/xx/"
DEFAULT_YY_DIR = ROOT / "./data/33exoplanets/yy/tmp"
DEFAULT_XX_DIR = ROOT / "./data/33exoplanets/xx/tmp"
DEFAULT_OUTPUT_ROOT = ROOT / "./data_process/post_process/visual_val/visual/vis"

# ====== Window configs ======
FIXED_W2_MHZ = 0.0064
W3_MHZ = 0.008  # 8 kHz

# save config
DEFAULT_DPI = 300
DEFAULT_FMT = "png"

# beam range (for --multi_beam)
BEAM_MIN = 1
BEAM_MAX = 19

# Robust color scaling percentiles (shared per event+window+pol in multi_beam mode)
VMIN_PCT = 2.0
VMAX_PCT = 98.0

# FAST 19-beam layout (3-4-5-4-3) with Mxx numbering
# Row1:  M18 M17 M16
# Row2:  M19 M07 M06 M15
# Row3:  M08 M02 M01 M05 M14
# Row4:  M09 M03 M04 M13
# Row5:  M10 M11 M12
FAST_LAYOUT = [
    [18, 17, 16],
    [19, 7, 6, 15],
    [8, 2, 1, 5, 14],
    [9, 3, 4, 13],
    [10, 11, 12],
]


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


def find_beam_file(folder: Path, group_id: str, beam_id: int) -> Optional[Path]:
    """
    Find the first file matching *{group_id}_Mxx* with extensions .fil or .h5.
    """
    mtag = f"M{beam_id:02d}"
    patterns = [
        str(folder / f"*{group_id}_{mtag}*.fil"),
        str(folder / f"*{group_id}_{mtag}*.h5"),
    ]
    matches: List[str] = []
    for pat in patterns:
        matches.extend(glob.glob(pat))
    matches = sorted(set(matches))
    if not matches:
        return None
    return Path(matches[0]).resolve()


def compute_windows(freq_start_mhz: float, freq_end_mhz: float) -> Dict[str, Tuple[float, float, float, float]]:
    """
    Return dict: window_name -> (center, f_start, f_stop, width)
    """
    f0 = 0.5 * (freq_start_mhz + freq_end_mhz)
    w_sig = abs(freq_end_mhz - freq_start_mhz)

    # fit_width: 2x signal width, avoid zero-width edge case
    w_fit = max(2.0 * w_sig, 7.5 * 10 * 1e-6)
    w_fixed = FIXED_W2_MHZ
    w_8k = W3_MHZ

    def bounds(center: float, width: float) -> Tuple[float, float]:
        return center - 0.5 * width, center + 0.5 * width

    f_fit_s, f_fit_e = bounds(f0, w_fit)
    f_fixed_s, f_fixed_e = bounds(f0, w_fixed)
    f_8k_s, f_8k_e = bounds(f0, w_8k)

    return {
        "fit_width": (f0, f_fit_s, f_fit_e, w_fit),
        "fixed_width": (f0, f_fixed_s, f_fixed_e, w_fixed),
        "8k": (f0, f_8k_s, f_8k_e, w_8k),
    }


def load_waterfall_slice(fname: Path, f_start: float, f_stop: float) -> Waterfall:
    """
    Use blimpy Waterfall to load a narrow frequency slice.
    f_start/f_stop are in MHz.
    """
    return Waterfall(str(fname), f_start=float(f_start), f_stop=float(f_stop))


def extract_2d_data(wf: Waterfall) -> np.ndarray:
    """
    Extract a (time, freq) 2D array from blimpy Waterfall.
    Returns arr_tf with shape (time, freq).
    """
    arr = np.array(wf.data)
    arr = np.squeeze(arr)

    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        for ax in range(3):
            if arr.shape[ax] == 1:
                arr2 = np.squeeze(arr, axis=ax)
                if arr2.ndim == 2:
                    return arr2
        return np.squeeze(arr[0])
    raise ValueError(f"Unexpected waterfall data shape: {arr.shape}")


def maybe_flip_freq(arr: np.ndarray, wf: Waterfall) -> np.ndarray:
    hdr = wf.header
    if hdr.get("foff", -1.0) < 0:
        return arr[:, ::-1]
    return arr


def get_tsamp_seconds(wf: Waterfall) -> float:
    """
    Try to infer time sampling (seconds) for proper time axis.
    Fallback: 1.0 if not available.
    """
    for obj in [wf, getattr(wf, "container", None)]:
        if obj is None:
            continue
        hdr = getattr(obj, "header", None)
        if isinstance(hdr, dict) and "tsamp" in hdr:
            try:
                return float(hdr["tsamp"])
            except Exception:
                pass
    return 1.0


def safe_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x)


def robust_vmin_vmax(arr_list: List[np.ndarray]) -> Tuple[float, float]:
    """
    Robust color scaling across a list of arrays.
    """
    vals = []
    for a in arr_list:
        if a is None:
            continue
        aa = np.asarray(a)
        aa = aa[np.isfinite(aa)]
        if aa.size:
            vals.append(aa)

    if not vals:
        return 0.0, 1.0

    allv = np.concatenate(vals)
    vmin = float(np.percentile(allv, VMIN_PCT))
    vmax = float(np.percentile(allv, VMAX_PCT))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = float(np.min(allv))
        vmax = float(np.max(allv))
        if vmax <= vmin:
            vmax = vmin + 1.0
    return vmin, vmax


def make_pol_dirs(ts_dir: Path, detailed: bool, plot_pol: bool):
    # Simplified output structure:
    # - XX/fit_width (optional)
    # - YY/fit_width (optional)
    # - I/fit_width (+ I/fixed_width, I/8k if detailed)
    if plot_pol:
        (ts_dir / "XX" / "fit_width").mkdir(parents=True, exist_ok=True)
        (ts_dir / "YY" / "fit_width").mkdir(parents=True, exist_ok=True)
    (ts_dir / "I" / "fit_width").mkdir(parents=True, exist_ok=True)
    if detailed:
        (ts_dir / "I" / "fixed_width").mkdir(parents=True, exist_ok=True)
        (ts_dir / "I" / "8k").mkdir(parents=True, exist_ok=True)


# -----------------------------
# Plot helpers (axes + colorbar)
# -----------------------------
def plot_single_panel(
        arr_tf: np.ndarray,
        f_start: float,
        f_stop: float,
        tsamp: float,
        title: str,
        save_path: Path,
        dpi: int,
        fmt: str,
        figsize: Tuple[float, float],
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
):
    """
    Plot one waterfall with correct axes:
      X = Frequency (MHz)
      Y = Time (s)
    arr_tf is (time, freq).
    """
    if arr_tf is None:
        return

    n_t, _n_f = arr_tf.shape
    tmax = tsamp * n_t

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    im = ax.imshow(
        arr_tf,
        aspect="auto",
        origin="lower",
        extent=[f_start, f_stop, 0.0, tmax],
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Time (s)")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Intensity (arb.)")

    fig.tight_layout()
    fig.savefig(save_path.with_suffix(f".{fmt}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_mosaic_19beams(
        beam_to_arr: Dict[int, np.ndarray],
        f_start: float,
        f_stop: float,
        tsamp: float,
        title: str,
        save_path: Path,
        dpi: int,
        fmt: str,
        figsize: Tuple[float, float],
):
    arrs = [beam_to_arr.get(b) for row in FAST_LAYOUT for b in row]
    vmin, vmax = robust_vmin_vmax([a for a in arrs if a is not None])

    tmax = None
    for a in arrs:
        if a is not None:
            tmax = tsamp * a.shape[0]
            break
    if tmax is None:
        tmax = 1.0

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows=5, ncols=11, width_ratios=[1] * 10 + [0.22], wspace=0.10, hspace=0.06)

    im_for_cbar = None
    axes_for_cbar = []

    for r in range(5):
        row_beams = FAST_LAYOUT[r]
        n = len(row_beams)
        if n == 0:
            continue

        start = (10 - 2 * n) // 2

        for i, beam in enumerate(row_beams):
            c0 = start + 2 * i
            ax = fig.add_subplot(gs[r, c0:c0 + 2])
            axes_for_cbar.append(ax)

            arr_tf = beam_to_arr.get(beam)
            if arr_tf is None:
                ax.text(0.5, 0.5, f"M{beam:02d}\nMissing", ha="center", va="center", transform=ax.transAxes)
                ax.set_xlim(f_start, f_stop)
                ax.set_ylim(0.0, tmax)
            else:
                n_t, _n_f = arr_tf.shape
                tmax_local = tsamp * n_t
                im = ax.imshow(
                    arr_tf,
                    aspect="auto",
                    origin="lower",
                    extent=[f_start, f_stop, 0.0, tmax_local],
                    vmin=vmin,
                    vmax=vmax,
                )
                im_for_cbar = im
                ax.text(
                    0.03, 0.97, f"M{beam:02d}",
                    transform=ax.transAxes,
                    ha="left", va="top",
                    fontsize=12,
                    color="white",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.45, linewidth=0),
                )

            if r == 4:
                ax.set_xlabel("Freq (MHz)" if i == 1 else "")
                ax.tick_params(labelbottom=True, labelsize=15)
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.4f'))
                ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            else:
                ax.set_xlabel("")
                ax.xaxis.set_major_locator(MultipleLocator(0.0001))
                ax.tick_params(labelbottom=False)

            if i == 0:
                ax.tick_params(labelleft=True)
                ax.set_ylabel("Time (s)" if r == 2 else "")
            else:
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)

    if not Settings.PROD:
        fig.suptitle(title, y=0.995)

    if im_for_cbar is not None:
        cax = fig.add_subplot(gs[:, 10])
        cbar = fig.colorbar(im_for_cbar, cax=cax)
        cbar.set_label("Intensity (arb.)")

    fig.subplots_adjust(top=0.965, bottom=0.045, left=0.055, right=0.945)

    fig.savefig(save_path.with_suffix(f".{fmt}"), dpi=dpi)
    plt.close(fig)


def plot_vertical_stack_19beams(
        beam_to_arr: Dict[int, np.ndarray],
        f_start: float,
        f_stop: float,
        tsamp: float,
        title: str,
        save_path: Path,
        dpi: int,
        fmt: str,
        figsize: Tuple[float, float],
):
    """
    Window 8k multi-beam: vertically stack 19 beams, no gaps (hspace=0).
    Correct axes:
      X = Frequency (MHz)
      Y = Time (s)
    """
    beams = list(range(BEAM_MIN, BEAM_MAX + 1))
    arrs = [beam_to_arr.get(b) for b in beams]
    vmin, vmax = robust_vmin_vmax([a for a in arrs if a is not None])

    tmax = None
    for a in arrs:
        if a is not None:
            tmax = tsamp * a.shape[0]
            break
    if tmax is None:
        tmax = 1.0

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows=len(beams), ncols=1, hspace=0.0)

    axes = []
    im_for_cbar = None

    for i, beam in enumerate(beams):
        ax = fig.add_subplot(gs[i, 0], sharex=axes[0] if axes else None)
        axes.append(ax)

        arr_tf = beam_to_arr.get(beam)
        if arr_tf is None:
            ax.text(0.02, 0.5, f"M{beam:02d} Missing", ha="left", va="center", transform=ax.transAxes)
            ax.set_xlim(f_start, f_stop)
            ax.set_ylim(0.0, tmax)
        else:
            n_t, _n_f = arr_tf.shape
            tmax_local = tsamp * n_t
            im = ax.imshow(
                arr_tf,
                aspect="auto",
                origin="lower",
                extent=[f_start, f_stop, 0.0, tmax_local],
                vmin=vmin,
                vmax=vmax,
            )
            im_for_cbar = im

        ax.set_title(f"M{beam:02d}", loc="left", fontsize=9)

        if i != len(beams) - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Frequency (MHz)")

        if i == 0:
            ax.set_ylabel("Time (s)")
        else:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)

    fig.suptitle(title, y=0.995)

    if im_for_cbar is not None:
        cbar = fig.colorbar(im_for_cbar, ax=axes, fraction=0.015, pad=0.01)
        cbar.set_label("Intensity (arb.)")

    fig.subplots_adjust(hspace=0.0, top=0.97, bottom=0.04, left=0.06, right=0.92)
    fig.savefig(save_path.with_suffix(f".{fmt}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Visual checker: render XX/YY/I images around event frequencies.")
    ap.add_argument("--target_csv", type=str, default=DEFAULT_TARGET_CSV, help="Path to target events CSV.")
    ap.add_argument("--xx_dir", type=str, default=DEFAULT_XX_DIR, help="Directory containing XX filterbank files.")
    ap.add_argument("--yy_dir", type=str, default=DEFAULT_YY_DIR, help="Directory containing YY filterbank files.")
    ap.add_argument("--out_root", type=str, default=DEFAULT_OUTPUT_ROOT, help="Output root directory.")
    ap.add_argument("--detailed", action="store_true", help="If set, also render fixed_width and 8k (I only).")
    ap.add_argument("--multi_beam", action="store_true", help="If set, render all beams as a mosaic/stack per event.")
    ap.add_argument("--plot_pol", action="store_true",
                    help="If set, render XX/YY for fit_width. Otherwise render I only.")
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

    df["freq_start"] = ensure_numeric(df["freq_start"])
    df["freq_end"] = ensure_numeric(df["freq_end"])

    before = len(df)
    df = df.dropna(subset=["group_id", "freq_start", "freq_end"]).copy()
    after = len(df)
    if after != before:
        print(f"[\033[32mInfo\033[0m] Dropped {before - after} rows due to missing group_id/freq_start/freq_end.")

    if args.max_events and args.max_events > 0:
        df = df.head(args.max_events).copy()
        print(f"[\033[32mInfo\033[0m] Debug mode: processing first {len(df)} events.")

    ts_dir = make_timestamp_dir(out_root)
    make_pol_dirs(ts_dir, args.detailed, args.plot_pol)

    print("\n==============================")
    print("[\033[32mInfo\033[0m] VISUAL CHECK")
    print("==============================")
    print(f"[\033[32mInfo\033[0m] Target CSV : {target_csv}")
    print(f"[\033[32mInfo\033[0m] XX_DIR     : {xx_dir}")
    print(f"[\033[32mInfo\033[0m] YY_DIR     : {yy_dir}")
    print(f"[\033[32mInfo\033[0m] Output dir : {ts_dir}")
    print(f"[\033[32mInfo\033[0m] Detailed   : {args.detailed}")
    print(f"[\033[32mInfo\033[0m] Multi-beam : {args.multi_beam}")
    print(f"[\033[32mInfo\033[0m] Plot pol   : {args.plot_pol}")
    print(f"[\033[32mInfo\033[0m] Format/DPI : {args.fmt}/{args.dpi}")

    meta_rows: List[Dict] = []

    beam_list = list(range(BEAM_MIN, BEAM_MAX + 1)) if args.multi_beam else [1]

    window_names = ["fit_width"] + (["fixed_width", "8k"] if args.detailed else [])

    df_iter = df.reset_index(drop=True).iterrows()
    for i, row in progress(df_iter, total=len(df), desc="Events"):
        group_id = str(row["group_id"])
        f_start = float(row["freq_start"])
        f_end = float(row["freq_end"])
        f0 = 0.5 * (f_start + f_end)

        win_map = compute_windows(f_start, f_end)

        base_meta = {
            "event_idx": i,
            "group_id": group_id,
            "freq_start_mhz": f_start,
            "freq_end_mhz": f_end,
            "freq_center_mhz": f0,
            "csv_source": safe_str(row["csv_id"]) if "csv_id" in df.columns else "",
            "DriftRate": safe_str(row["DriftRate"]) if "DriftRate" in df.columns else "",
            "SNR": safe_str(row["SNR"]) if "SNR" in df.columns else "",
            "class_id": safe_str(row["class_id"]) if "class_id" in df.columns else "",
            "confidence": safe_str(row["confidence"]) if "confidence" in df.columns else "",
        }

        # --------------------------
        # MULTI-BEAM
        # --------------------------
        if args.multi_beam:
            cache: Dict[str, Dict[str, Dict[int, np.ndarray]]] = {w: {"XX": {}, "YY": {}, "I": {}} for w in
                                                                  window_names}
            file_map_xx: Dict[int, Optional[Path]] = {}
            file_map_yy: Dict[int, Optional[Path]] = {}

            for beam_id in beam_list:
                file_map_xx[beam_id] = find_beam_file(xx_dir, group_id, beam_id)
                file_map_yy[beam_id] = find_beam_file(yy_dir, group_id, beam_id)

            rendered_any = False
            tsamp_ref = 1.0

            for wname in window_names:
                _, ws, we, ww = win_map[wname]

                # store/publish XX/YY only if (fit_width AND plot_pol)
                store_xx_yy = (wname == "fit_width") and args.plot_pol

                for beam_id in beam_list:
                    xx_file = file_map_xx.get(beam_id)
                    yy_file = file_map_yy.get(beam_id)
                    if xx_file is None or yy_file is None:
                        continue

                    try:
                        wf_xx = load_waterfall_slice(xx_file, ws, we)
                        wf_yy = load_waterfall_slice(yy_file, ws, we)
                        arr_xx = maybe_flip_freq(extract_2d_data(wf_xx), wf_xx)
                        arr_yy = maybe_flip_freq(extract_2d_data(wf_yy), wf_yy)

                        t = min(arr_xx.shape[0], arr_yy.shape[0])
                        f = min(arr_xx.shape[1], arr_yy.shape[1])
                        arr_xx = arr_xx[:t, :f]
                        arr_yy = arr_yy[:t, :f]
                        arr_i = arr_xx + arr_yy

                        tsamp_ref = get_tsamp_seconds(wf_xx)

                        if store_xx_yy:
                            cache[wname]["XX"][beam_id] = arr_xx
                            cache[wname]["YY"][beam_id] = arr_yy
                        cache[wname]["I"][beam_id] = arr_i

                        rendered_any = True
                    except Exception as e:
                        print(
                            f"[\033[33mWarn\033[0m] Failed load group_id={group_id} beam=M{beam_id:02d} window={wname}: {e}")
                        continue

                # Plotting policy:
                # - fit_width: (XX,YY,I) if plot_pol else (I only)
                # - fixed_width & 8k: I only
                if wname == "fit_width" and args.plot_pol:
                    pols_to_plot = ["XX", "YY", "I"]
                else:
                    pols_to_plot = ["I"]

                for pol in pols_to_plot:
                    beam_to_arr = cache[wname][pol]
                    base = f"{group_id}_evt{i:06d}_fc{f0:.6f}MHz_w{wname}_bw{ww:.6f}MHz_{pol}_MB"
                    save_dir = ts_dir / pol / wname
                    save_path = save_dir / base
                    title = f"{pol} | group={group_id} | evt={i} | f0={f0:.6f} MHz | window={wname}"

                    if wname in ("fit_width", "fixed_width"):
                        plot_mosaic_19beams(beam_to_arr=beam_to_arr, f_start=ws, f_stop=we, tsamp=tsamp_ref,
                                            title=title, save_path=save_path, dpi=args.dpi, fmt=args.fmt,
                                            figsize=(18, 14), )
                    else:
                        plot_vertical_stack_19beams(beam_to_arr=beam_to_arr, f_start=ws, f_stop=we, tsamp=tsamp_ref,
                                                    title=title, save_path=save_path, dpi=args.dpi, fmt=args.fmt,
                                                    figsize=(18, 0.8 * 19), )

            for beam_id in beam_list:
                meta = dict(base_meta)
                meta.update(
                    {"beam_id": beam_id, "file_xx": str(file_map_xx.get(beam_id)) if file_map_xx.get(beam_id) else "",
                     "file_yy": str(file_map_yy.get(beam_id)) if file_map_yy.get(beam_id) else "",
                     "rendered": "1" if rendered_any else "0",
                     "windows": ",".join(window_names) if rendered_any else "", "mode": "multi_beam_mosaic_stack", })
                meta_rows.append(meta)
            continue

        # --------------------------
        # SINGLE-BEAM (M01)
        # --------------------------
        beam_id = 1
        xx_file = find_beam_file(xx_dir, group_id, beam_id)
        yy_file = find_beam_file(yy_dir, group_id, beam_id)

        if xx_file is None or yy_file is None:
            print(f"[\033[33mWarn\033[0m] Missing XX/YY file for group_id={group_id}, beam=M{beam_id:02d}. Skipped.")
            meta = dict(base_meta)
            meta.update({"beam_id": beam_id, "file_xx": str(xx_file) if xx_file else "",
                         "file_yy": str(yy_file) if yy_file else "", "rendered": "0", "windows": "",
                         "mode": "single_beam", })
            meta_rows.append(meta)
            continue

        rendered_any = False
        for wname in window_names:
            _, ws, we, ww = win_map[wname]
            try:
                wf_xx = load_waterfall_slice(xx_file, ws, we)
                wf_yy = load_waterfall_slice(yy_file, ws, we)
                arr_xx = maybe_flip_freq(extract_2d_data(wf_xx), wf_xx)
                arr_yy = maybe_flip_freq(extract_2d_data(wf_yy), wf_yy)

                t = min(arr_xx.shape[0], arr_yy.shape[0])
                f = min(arr_xx.shape[1], arr_yy.shape[1])
                arr_xx = arr_xx[:t, :f]
                arr_yy = arr_yy[:t, :f]
                arr_i = arr_xx + arr_yy

                tsamp = get_tsamp_seconds(wf_xx)

                # Plotting policy:
                # - fit_width: (XX,YY,I) if plot_pol else (I only)
                # - fixed_width & 8k: I only
                if wname == "fit_width" and args.plot_pol:
                    plot_list = [("XX", arr_xx), ("YY", arr_yy), ("I", arr_i)]
                else:
                    plot_list = [("I", arr_i)]

                for pol, arr in plot_list:
                    vmin, vmax = robust_vmin_vmax([arr])
                    base = f"{group_id}_M{beam_id:02d}_evt{i:06d}_fc{f0:.6f}MHz_w{wname}_bw{ww:.6f}MHz_{pol}"
                    save_path = ts_dir / pol / wname / base
                    title = f"{pol} | group={group_id} | M{beam_id:02d} | evt={i} | f0={f0:.6f} MHz | window={wname}"

                    figsize = (12, 9) if wname in ("fit_width", "fixed_width") else (18, 6)
                    plot_single_panel(arr_tf=arr, f_start=ws, f_stop=we, tsamp=tsamp, title=title, save_path=save_path,
                                      dpi=args.dpi, fmt=args.fmt, figsize=figsize, vmin=vmin, vmax=vmax, )

                rendered_any = True
            except Exception as e:
                print(f"[\033[33mWarn\033[0m] Failed to load slice for group_id={group_id}, window={wname}: {e}")
                continue

        meta = dict(base_meta)
        meta.update({"beam_id": beam_id, "file_xx": str(xx_file), "file_yy": str(yy_file),
                     "rendered": "1" if rendered_any else "0",
                     "windows": ",".join(window_names) if rendered_any else "", "mode": "single_beam", })
        meta_rows.append(meta)

    meta_df = pd.DataFrame(meta_rows)
    meta_path = ts_dir / "event_meta.csv"
    meta_df.to_csv(meta_path, index=False)
    print(f"\n[\033[32mInfo\033[0m] Metadata written: {meta_path}")
    print(f"[\033[32mInfo\033[0m] Done!")


if __name__ == "__main__":
    main()
