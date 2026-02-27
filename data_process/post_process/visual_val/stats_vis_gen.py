# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FAST SETI Stats + Optional Source Vis Generator

1) Read a CSV (e.g., filter output), draw histogram of Uncorrected_Frequency vs count.
   - Bin width = df (one frequency channel) in Hz, converted to MHz.
2) Stats add-on: Uncorrected_Frequency - SNR scatter plot
   - Color points by group_id (categorical)
3) Optional source-level visualization (I-only), following uploaded vis framework defaults:
   - Read XX/YY source files from default directories
   - For each row in CSV, render ONE image: I = XX + YY around fc=Uncorrected_Frequency
   - All source images go to: <out_dir>/vis_source/

Notes:
- Style must match uploaded code, including Settings.PROD behavior.
- Default dirs/configs follow source_level_vis_gen.py patterns.
"""

import argparse
import glob
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from blimpy import Waterfall

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from config.settings import Settings

# ====== tqdm progress bar ======
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None

if Settings.PROD:
    import matplotlib as mpl
    from matplotlib import font_manager as fm

    mpl.rcParams["font.family"] = "Times New Roman"
    if not any("Times New Roman" in f.name for f in fm.fontManager.ttflist):
        print("[\033[33mWarning\033[0m] Fonts 'Times New Roman' not found, falling back to 'Serif'")
        mpl.rcParams["font.family"] = "Liberation Serif"
    mpl.rcParams["font.size"] = 15
    mpl.rcParams["font.weight"] = "semibold"
    mpl.rcParams["axes.titleweight"] = "bold"
    mpl.rcParams["axes.labelweight"] = "bold"


def progress(iterable, total: int, desc: str):
    if tqdm is not None:
        return tqdm(iterable, total=total, desc=desc)
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


# =========================
# Configs (default aligned)
# =========================
# CSV columns
COL_GROUP_ID = "group_id"
COL_BEAM_ID = "beam_id"
COL_FREQ_MHZ = "Uncorrected_Frequency"
COL_SNR_DEFAULT = "SNR"  # preferred SNR column name

# Histogram bin = df
DF_HZ = 7.450580597
DF_MHZ = DF_HZ * 1e-6

# Default frequency bounds (used for histogram axis and safe binning)
DEFAULT_F_MIN = 1050.0
DEFAULT_F_MAX = 1450.0

# Working directory
WD = ROOT / "data_process/post_process/analysis_out/20260207_204327_F[r1050-1450]_SNR[off]_DR[t-0.038024±0.006297]_G[n4]_B[off]_match"
DEFAULT_TARGET_CSV = WD / "filter_out.csv"  # Input target
OUTPUT_ROOT = WD  # Output root

# ---- Source vis defaults ----
DEFAULT_YY_DIR = "/data/Raid0/obs_data/33exoplanets/yy/"
DEFAULT_XX_DIR = "/data/Raid0/obs_data/33exoplanets/xx/"
# DEFAULT_YY_DIR = ROOT / "./data/33exoplanets/yy/"
# DEFAULT_XX_DIR = ROOT / "./data/33exoplanets/xx/"

# Window config
FIXED_W2_MHZ = 0.00192
DEFAULT_DPI = 300
DEFAULT_FMT = "png"
VMIN_PCT = 2.0
VMAX_PCT = 98.0
FIGSIZE = (20, 8)
GROUP_ORDER = ["Ross-128", "GJ-9066"]
NBS_EVENT_FREQ = 1148.4167512225800
TOL = 1e-9  # MHz tolerance for locating signal of interest
LABEL = "NBS 260108"


def ensure_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_output_dir(base: Path, tag: str) -> Path:
    out_dir = base / f"{now_tag()}_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def robust_vmin_vmax(arr: np.ndarray) -> Tuple[float, float]:
    aa = np.asarray(arr)
    aa = aa[np.isfinite(aa)]
    if aa.size == 0:
        return 0.0, 1.0

    vmin = float(np.percentile(aa, VMIN_PCT))
    vmax = float(np.percentile(aa, VMAX_PCT))

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = float(np.min(aa))
        vmax = float(np.max(aa))
        if vmax <= vmin:
            vmax = vmin + 1.0
    return vmin, vmax


def bounds(center_mhz: float, width_mhz: float) -> Tuple[float, float]:
    return center_mhz - 0.5 * width_mhz, center_mhz + 0.5 * width_mhz


def load_waterfall_slice(fname: Path, f_start: float, f_stop: float) -> Waterfall:
    return Waterfall(str(fname), f_start=float(f_start), f_stop=float(f_stop))


def maybe_flip_freq(arr: np.ndarray, wf: Waterfall) -> np.ndarray:
    hdr = wf.header
    if hdr.get("foff", -1.0) < 0:
        return arr[:, ::-1]
    return arr


def extract_2d_data(wf: Waterfall) -> np.ndarray:
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


def get_tsamp_seconds(wf: Waterfall) -> float:
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


def plot_single_panel(arr_tf: np.ndarray, f_start: float, f_stop: float, tsamp: float,
                      title: str, save_path: Path, dpi: int, fmt: str,
                      figsize: Tuple[float, float], vmin: float, vmax: float):
    n_t, _n_f = arr_tf.shape
    tmax = tsamp * n_t

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    im = ax.imshow(arr_tf, aspect="auto", origin="lower", extent=[f_start, f_stop, 0.0, tmax], vmin=vmin, vmax=vmax, )
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Time (s)")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Intensity (arb.)")

    fig.tight_layout()
    fig.savefig(save_path.with_suffix(f".{fmt}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_freq_hist(freq_mhz: np.ndarray, out_path: Path,
                   df_mhz: float, fmin: float, fmax: float):
    freq_mhz = freq_mhz[np.isfinite(freq_mhz)]
    if freq_mhz.size == 0:
        print("[\033[33mWarning\033[0m] No finite frequencies to plot histogram.")
        return

    # Clip to avoid insane bins if CSV contains out-of-band garbage
    lo = max(float(np.min(freq_mhz)), fmin)
    hi = min(float(np.max(freq_mhz)), fmax)
    if hi <= lo:
        lo = float(np.min(freq_mhz))
        hi = float(np.max(freq_mhz))
        if hi <= lo:
            hi = lo + df_mhz

    # Bin edges with width=df_mhz
    n_bins = int(np.ceil((hi - lo) / df_mhz))
    n_bins = max(n_bins, 1)
    edges = lo + df_mhz * np.arange(n_bins + 1)

    fig = plt.figure(figsize=(16, 7))
    ax = fig.add_subplot(111)
    ax.hist(freq_mhz, bins=edges)

    ax.set_xlabel("Uncorrected Frequency (MHz)")
    ax.set_ylabel("Count")
    title = f"Frequency Count Histogram (bin=df={DF_HZ:.6f} Hz)"
    ax.set_title(title if not Settings.PROD else "Frequency Count Histogram")

    fig.tight_layout()
    fig.savefig(out_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)


def infer_snr_column(df: pd.DataFrame) -> Optional[str]:
    # preferred exact match
    if COL_SNR_DEFAULT in df.columns:
        return COL_SNR_DEFAULT
    # case-insensitive exact match
    lower_map = {c.lower(): c for c in df.columns}
    if "snr" in lower_map:
        return lower_map["snr"]
    # fuzzy contains
    cands = [c for c in df.columns if "snr" in c.lower()]
    if len(cands) > 0:
        return cands[0]
    return None


def plot_freq_snr_scatter(freq_mhz: np.ndarray, snr: np.ndarray, groups: np.ndarray, out_path: Path):
    m = np.isfinite(freq_mhz) & np.isfinite(snr)
    if np.count_nonzero(m) == 0:
        print("[\033[33mWarning\033[0m] No finite (freq, snr) pairs to plot scatter.")
        return

    freq_mhz = freq_mhz[m]
    snr = snr[m]
    groups = groups[m].astype(str)

    rest = [g for g in pd.unique(groups) if g not in GROUP_ORDER]
    uniq = np.array(GROUP_ORDER + rest)
    n_g = len(uniq)

    # categorical colors
    cmap = plt.get_cmap("tab20")
    colors = {g: cmap(i % 20) for i, g in enumerate(uniq)}

    fig = plt.figure(figsize=(16, 7))
    ax = fig.add_subplot(111)

    # plot each group separately for consistent legend
    for g in uniq:
        idx = groups == g
        ax.scatter(freq_mhz[idx], snr[idx], s=30, alpha=0.85, label=g, c=[colors[g]], edgecolors="none", )

    ax.set_xlabel("Uncorrected Frequency (MHz)")
    ax.set_ylabel("SNR")
    ax.set_title("Uncorrected Frequency vs SNR" if Settings.PROD else f"Uncorrected Frequency vs SNR (groups={n_g})")

    # legend strategy: avoid unreadable mega-legend
    if n_g <= 20:
        ax.legend(loc="best", fontsize=12, frameon=False)
    else:
        # Too many groups: show no legend, but save mapping
        ax.text(0.99, 0.01, f"Legend suppressed (groups={n_g} > 20). See group_color_map.csv", transform=ax.transAxes,
                ha="right", va="bottom")
        map_df = pd.DataFrame({"group_id": uniq, "color_idx_tab20": [i % 20 for i in range(n_g)]})
        map_df.to_csv(out_path.with_suffix("").with_name("group_color_map.csv"), index=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_deltaf_scan(freq_mhz: np.ndarray, out_path: Path, *, df_min_hz: float, df_max_hz: float, df_step_hz: float,
                     mark_known: Optional[List[float]] = None):
    m = np.isfinite(freq_mhz)
    if np.count_nonzero(m) == 0:
        print("[\033[33mWarning\033[0m] No finite frequencies for Δf scan.")
        return

    freq_mhz = np.asarray(freq_mhz, dtype=float)[m]
    freq_hz = freq_mhz * 1e6

    deltaf_grid = np.arange(df_min_hz, df_max_hz, df_step_hz)
    R_vals = np.zeros_like(deltaf_grid, dtype=float)

    N = len(freq_hz)

    for i, df_hz in enumerate(deltaf_grid):
        phase = 2.0 * np.pi * freq_hz / df_hz
        vec = np.exp(1j * phase)
        R_vals[i] = np.abs(np.sum(vec) / N)

    # ---- plotting ----
    fig = plt.figure(figsize=(16, 7))
    ax = fig.add_subplot(111)
    ax.margins(x=0.015, y=0.025)
    ax.plot(deltaf_grid * 1e-3, R_vals, linewidth=1.35, alpha=0.85)

    ax.set_xlabel("Δf (kHz)")
    ax.set_ylabel("Rayleigh locking statistic R(Δf)")
    ax.set_title("Δf Scan via Rayleigh Comb Test")

    ax.grid(True, alpha=0.3)

    # mark known engineering frequencies if provided
    if mark_known is not None:
        for val in mark_known:
            ax.axvline(val * 1e-3, linestyle="--", linewidth=1.2, color="orange", alpha=0.6)

    # mark global maximum
    idx_max = np.argmax(R_vals)
    df_best = deltaf_grid[idx_max]
    R_best = R_vals[idx_max]

    x_best = df_best * 1e-3
    y_offset = 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    x_offset = 0.005 * (ax.get_xlim()[1] - ax.get_xlim()[0])
    ax.scatter(x_best, R_best, s=180, color="red", edgecolor="black", zorder=5, marker="*")
    ax.text(df_best * 1e-3 + x_offset, R_best - y_offset, f"  Δf = {df_best * 1e-3:.4f} kHz, R = {R_best:.4f}",
            verticalalignment="bottom")

    fig.tight_layout()
    fig.savefig(out_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"[\033[32mInfo\033[0m] Δf best = {df_best:.6f} Hz | R = {R_best:.6f}")

    return df_best


def plot_modf_residual(freq_mhz: np.ndarray, groups: np.ndarray, delta_f_mhz: float, f0_mhz: float,
                       out_path: Path, *, wrap_centered: bool = True, candidate_mask: Optional[np.ndarray] = None,
                       candidate_label: Optional[str] = None):
    m = np.isfinite(freq_mhz) & np.isfinite(delta_f_mhz) & (delta_f_mhz > 0)
    if np.count_nonzero(m) == 0:
        print("[\033[33mWarning\033[0m] No finite frequencies to plot residual/phase.")
        return

    freq_mhz = np.asarray(freq_mhz, dtype=float)[m]
    groups = np.asarray(groups)[m].astype(str)

    if candidate_mask is not None:
        candidate_mask = np.asarray(candidate_mask, dtype=bool)
        if candidate_mask.shape != np.asarray(freq_mhz).shape:
            candidate_mask = candidate_mask[m]

    df = float(delta_f_mhz)
    x = freq_mhz - float(f0_mhz)

    # wrapped residual in MHz
    if wrap_centered:
        r_mhz = (x + 0.5 * df) % df - 0.5 * df
        r_min_khz, r_max_khz = -(df * 1e3) / 2.0, (df * 1e3) / 2.0
        theta_ticks = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
        theta_ticklabels = [r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"]
    else:
        r_mhz = x % df
        r_min_khz, r_max_khz = 0.0, df * 1e3
        theta_ticks = [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
        theta_ticklabels = [r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"]

    r_khz = r_mhz * 1e3

    # group ordering & colors (match your style)
    rest = [g for g in pd.unique(groups) if g not in GROUP_ORDER]
    uniq = np.array(GROUP_ORDER + rest)
    n_g = len(uniq)

    cmap = plt.get_cmap("tab20")
    colors = {g: cmap(i % 20) for i, g in enumerate(uniq)}

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111)

    # plot each group separately
    for g in uniq:
        idx = groups == g
        if np.count_nonzero(idx) == 0:
            continue
        ax.scatter(r_khz[idx], freq_mhz[idx],
                   s=30, alpha=0.85, label=g,
                   c=[colors[g]], edgecolors="none")

    # highlight candidate(s)
    if candidate_mask is not None and np.any(candidate_mask):
        ax.scatter(r_khz[candidate_mask], freq_mhz[candidate_mask], s=140, marker="*", color="red", edgecolors="black",
                   zorder=5, label=candidate_label)

    ax.set_xlim(r_min_khz, r_max_khz)
    ax.set_xlabel(rf"Wrapped residual $r(f)$ (kHz), $\Delta f$={df * 1e3:.3f} kHz")
    ax.set_ylabel("Uncorrected Frequency (MHz)")

    title = rf"Comb test via residual/phase locking ($f \approx f_0 + k\Delta f$), $f_0$={float(f0_mhz):.6f} MHz"
    ax.set_title(title if Settings.PROD else f"{title} (groups={n_g}, N={len(freq_mhz)})")
    ax.grid(True, alpha=0.25)

    # legend strategy: same as your style
    if n_g <= 20:
        ax.legend(loc="best", fontsize=12, frameon=False)
    else:
        ax.text(0.99, 0.01, f"Legend suppressed (groups={n_g} > 20). See group_color_map.csv",
                transform=ax.transAxes, ha="right", va="bottom")
        map_df = pd.DataFrame({"group_id": uniq, "color_idx_tab20": [i % 20 for i in range(n_g)]})
        map_df.to_csv(out_path.with_suffix("").with_name("group_color_map.csv"), index=False)

    # top axis: phase residual theta mapped from residual
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())

    # residual-to-phase mapping: theta = 2π r/Δf -> r = Δf * theta / (2π)
    tick_pos_khz = (df * 1e3) * (np.array(theta_ticks) / (2.0 * np.pi))
    ax_top.set_xticks(tick_pos_khz)
    ax_top.set_xticklabels(theta_ticklabels)
    ax_top.set_xlabel(r"Phase residual $\theta$ (rad)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)


def infer_group_id_from_path(p: Path, beam_id: int) -> Optional[str]:
    name = p.name
    mtag = f"_M{beam_id:02d}"
    if mtag not in name:
        return None
    gid = name.split(mtag)[0]
    gid = gid.strip("_- ")
    return gid if gid else None


def build_group_file_map(folder: Path, beam_id: int) -> Dict[str, Path]:
    mtag = f"M{beam_id:02d}"
    patterns = [
        str(folder / f"*_{mtag}*.fil"),
        str(folder / f"*_{mtag}*.h5"),
    ]
    files: List[Path] = []
    for pat in patterns:
        files.extend([Path(x).resolve() for x in glob.glob(pat)])
    files = sorted(set(files))

    out: Dict[str, Path] = {}
    for f in files:
        gid = infer_group_id_from_path(f, beam_id)
        if gid is None:
            continue
        if gid not in out:
            out[gid] = f
    return out


def run_source_vis(df: pd.DataFrame, out_dir: Path):
    xx_dir = Path(DEFAULT_XX_DIR).expanduser().resolve()
    yy_dir = Path(DEFAULT_YY_DIR).expanduser().resolve()
    vis_dir = out_dir / "vis_source"
    vis_dir.mkdir(parents=True, exist_ok=True)

    if not xx_dir.exists():
        print(f"[\033[31mError\033[0m] XX_DIR not found: {xx_dir}")
        sys.exit(1)
    if not yy_dir.exists():
        print(f"[\033[31mError\033[0m] YY_DIR not found: {yy_dir}")
        sys.exit(1)

    # Beam maps only for beams appearing in df
    beams = sorted({int(b) for b in df[COL_BEAM_ID].dropna().astype(int).tolist()})
    if len(beams) == 0:
        print("[\033[33mWarning\033[0m] No valid beam_id rows for source vis.")
        return

    print(f"[\033[32mInfo\033[0m] Source vis beams: {beams}")
    print(f"[\033[32mInfo\033[0m] XX_DIR: {xx_dir}")
    print(f"[\033[32mInfo\033[0m] YY_DIR: {yy_dir}")
    print(f"[\033[32mInfo\033[0m] vis_source: {vis_dir}")
    print(f"[\033[32mInfo\033[0m] Window fixed_width: {FIXED_W2_MHZ} MHz | Format/DPI: {DEFAULT_FMT}/{DEFAULT_DPI}")

    xx_maps = {b: build_group_file_map(xx_dir, b) for b in beams}
    yy_maps = {b: build_group_file_map(yy_dir, b) for b in beams}

    meta_rows: List[Dict] = []
    rows = df[[COL_GROUP_ID, COL_BEAM_ID, COL_FREQ_MHZ]].copy()
    rows[COL_GROUP_ID] = rows[COL_GROUP_ID].astype(str)
    rows[COL_BEAM_ID] = ensure_numeric(rows[COL_BEAM_ID]).astype("Int64")
    rows[COL_FREQ_MHZ] = ensure_numeric(rows[COL_FREQ_MHZ])

    for r in progress(rows.itertuples(index=False), total=len(rows), desc="Render I (source)"):
        gid = getattr(r, COL_GROUP_ID)
        beam = getattr(r, COL_BEAM_ID)
        fc = getattr(r, COL_FREQ_MHZ)

        if beam is None or not np.isfinite(fc):
            meta_rows.append(
                {"group_id": gid, "beam_id": beam, "fc_mhz": fc, "rendered": "0", "err": "invalid_row", "out_img": ""})
            continue

        beam = int(beam)
        xx_file = xx_maps.get(beam, {}).get(gid)
        yy_file = yy_maps.get(beam, {}).get(gid)

        f_start, f_stop = bounds(float(fc), FIXED_W2_MHZ)

        rowm = {
            "group_id": gid,
            "beam_id": beam,
            "fc_mhz": float(fc),
            "f_start_mhz": f_start,
            "f_stop_mhz": f_stop,
            "file_xx": str(xx_file) if xx_file else "",
            "file_yy": str(yy_file) if yy_file else "",
            "rendered": "0",
            "out_img": "",
            "err": "",
        }

        if xx_file is None or yy_file is None:
            rowm["err"] = "missing_xx_or_yy"
            meta_rows.append(rowm)
            continue

        try:
            wf_xx = load_waterfall_slice(xx_file, f_start, f_stop)
            wf_yy = load_waterfall_slice(yy_file, f_start, f_stop)

            arr_xx = maybe_flip_freq(extract_2d_data(wf_xx), wf_xx)
            arr_yy = maybe_flip_freq(extract_2d_data(wf_yy), wf_yy)

            t = min(arr_xx.shape[0], arr_yy.shape[0])
            f = min(arr_xx.shape[1], arr_yy.shape[1])
            arr_i = arr_xx[:t, :f] + arr_yy[:t, :f]

            tsamp = get_tsamp_seconds(wf_xx)
            vmin, vmax = robust_vmin_vmax(arr_i)

            base = f"{gid}_M{beam:02d}_fc{float(fc):.6f}MHz_fixed_width_I"
            save_path = vis_dir / base

            title = (
                f"I | group={gid} | M{beam:02d} | f0={float(fc):.6f} MHz | fixed_width"
                if not Settings.PROD
                else f"{gid} | I | M{beam:02d} | f0={float(fc):.4f} MHz"
            )

            plot_single_panel(
                arr_tf=arr_i,
                f_start=f_start,
                f_stop=f_stop,
                tsamp=tsamp,
                title=title,
                save_path=save_path,
                dpi=DEFAULT_DPI,
                fmt=DEFAULT_FMT,
                figsize=FIGSIZE,
                vmin=vmin,
                vmax=vmax,
            )

            rowm["rendered"] = "1"
            rowm["out_img"] = str(save_path.with_suffix(f".{DEFAULT_FMT}"))
            meta_rows.append(rowm)

        except Exception as e:
            rowm["err"] = repr(e)
            meta_rows.append(rowm)
            print(
                f"[\033[33mWarning\033[0m] I render failed. gid={gid} beam=M{beam:02d} fc={float(fc):.6f}MHz err={repr(e)}")

    meta_df = pd.DataFrame(meta_rows)
    meta_path = out_dir / "vis_source_meta.csv"
    meta_df.to_csv(meta_path, index=False)
    print(f"[\033[32mInfo\033[0m] Source vis metadata: {meta_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Generate stats histogram, scatter and optional comb residual diagnostic.")
    ap.add_argument("--input_csv", type=str, default=DEFAULT_TARGET_CSV,
                    help="Input CSV path.")

    # Histogram controls
    ap.add_argument("--df_hz", type=float, default=DF_HZ * 5e5,
                    help="Histogram bin width in Hz (default: df * 5e5).")
    ap.add_argument("--hist_fmin", type=float, default=DEFAULT_F_MIN,
                    help="Histogram min frequency (MHz).")
    ap.add_argument("--hist_fmax", type=float, default=DEFAULT_F_MAX,
                    help="Histogram max frequency (MHz).")

    # Comb diagnostic (single key control parameter)
    ap.add_argument("--comb_df_hz", type=float, default=None,
                    help="If set, run comb residual/phase diagnostic using this Δf (Hz).")

    # Output controls
    ap.add_argument("--output_dir", type=str, default=str(OUTPUT_ROOT),
                    help="Output directory.")

    # Optional source vis
    ap.add_argument("--render_source", action="store_true", default=False,
                    help="Enable source-level visualization.")

    args = ap.parse_args()

    in_csv = Path(args.input_csv).expanduser().resolve()
    if not in_csv.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {in_csv}")

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n==============================")
    print("[\033[32mInfo\033[0m] STATS + VIS GEN")
    print("==============================")
    print(f"[\033[32mInfo\033[0m] Input CSV : {in_csv}")
    print(f"[\033[32mInfo\033[0m] Output dir: {out_dir}")
    if args.comb_df_hz is not None:
        print(f"[\033[32mInfo\033[0m] Comb Δf   : {args.comb_df_hz}")

    df = pd.read_csv(in_csv)

    # --- required columns ---
    required = [COL_GROUP_ID, COL_BEAM_ID, COL_FREQ_MHZ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df[COL_BEAM_ID] = ensure_numeric(df[COL_BEAM_ID])
    df[COL_FREQ_MHZ] = ensure_numeric(df[COL_FREQ_MHZ])

    df = df.dropna(subset=[COL_GROUP_ID, COL_BEAM_ID, COL_FREQ_MHZ]).copy()

    # ---- Histogram ----
    hist_path = out_dir / "freq_hist.png"
    plot_freq_hist(
        freq_mhz=df[COL_FREQ_MHZ].to_numpy(dtype=float),
        out_path=hist_path,
        df_mhz=float(args.df_hz) * 1e-6,
        fmin=float(args.hist_fmin),
        fmax=float(args.hist_fmax),
    )
    print(f"[\033[32mInfo\033[0m] Histogram saved: {hist_path}")

    # ---- Freq-SNR scatter ----
    snr_col = infer_snr_column(df)
    if snr_col is None:
        raise KeyError("Cannot find SNR column.")
    df[snr_col] = ensure_numeric(df[snr_col])

    scat_path = out_dir / "freq_snr_scatter.png"
    plot_freq_snr_scatter(
        freq_mhz=df[COL_FREQ_MHZ].to_numpy(dtype=float),
        snr=df[snr_col].to_numpy(dtype=float),
        groups=df[COL_GROUP_ID].astype(str).to_numpy(),
        out_path=scat_path,
    )
    print(f"[\033[32mInfo\033[0m] Freq-SNR scatter saved: {scat_path}")

    # ---- Δf scan ----
    scan_path = out_dir / "deltaf_scan.png"

    df_best = plot_deltaf_scan(
        freq_mhz=df[COL_FREQ_MHZ].to_numpy(dtype=float),
        out_path=scan_path,
        df_min_hz=1000,
        df_max_hz=150000,
        df_step_hz=1,
        mark_known=[33333.3333, 125000]
    )
    print(f"[\033[32mInfo\033[0m] Δf scan saved: {scan_path}")

    # ---- Comb residual / phase diagnostic ----
    df_comb = df_best if args.comb_df_hz is None else args.comb_df_hz
    print(f"[\033[32mInfo\033[0m] Running comb diagnostic with Δf={df_comb} Hz")

    comb_path = out_dir / "comb_residual_phase.png"

    # choose f0 as minimum frequency (simple deterministic choice)
    f0_mhz = float(df[COL_FREQ_MHZ].min())
    freq_mhz = df[COL_FREQ_MHZ].to_numpy(dtype=float)

    plot_modf_residual(
        freq_mhz=freq_mhz,
        groups=df[COL_GROUP_ID].astype(str).to_numpy(),
        delta_f_mhz=float(df_comb) * 1e-6,
        f0_mhz=f0_mhz,
        out_path=comb_path,
        candidate_mask=np.isclose(freq_mhz, NBS_EVENT_FREQ, atol=TOL),
        candidate_label=LABEL
    )

    print(f"[\033[32mInfo\033[0m] Comb diagnostic saved: {comb_path}")

    # ---- Optional source vis ----
    if args.render_source:
        if Settings.PROD:
            mpl.rcParams["font.size"] = 45
        run_source_vis(df=df, out_dir=out_dir)

    print("\n[\033[32mInfo\033[0m] Done!")


if __name__ == "__main__":
    main()
