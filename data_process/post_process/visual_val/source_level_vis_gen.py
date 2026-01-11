# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FAST SETI Source-level Frequency Visual Checker (I-only, M01, fixed_width)

Simplified version:
- No CSV inputs.
- XX_DIR and YY_DIR already contain ALL sources (all group_id files).
- Frequency list is hard-coded in config.
- For each frequency:
    - Create folder: out_root/<timestamp>/freq_<MHz>MHz/
    - For each group_id that has both XX and YY files for M01:
        - Render ONE image: total intensity I = XX + YY in fixed_width window

File matching rule (consistent with your engine):
- group_id is inferred from filenames by locating "...{group_id}_M01..."
- For each group_id, pick first match in XX_DIR and YY_DIR.

Output:
out_root/<timestamp>/
  freq_1140.604000MHz/
    <group_id>_M01_fc1140.604000MHz_fixed_width_I.png
    ...
  freq_1333.327000MHz/
    ...
  source_level_meta.csv
"""

import glob
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from blimpy import Waterfall

# ====== tqdm progress bar (new) ======
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None


def progress(iterable, total: int, desc: str):
    """
    Wrapper to provide a progress bar.
    - If tqdm is available: use a real progress bar.
    - Else: fall back to lightweight console progress updates.
    """
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


# ====== Configs ======
DEFAULT_YY_DIR = "/data/Raid0/obs_data/33exoplanets/yy/"
DEFAULT_XX_DIR = "/data/Raid0/obs_data/33exoplanets/xx/"
DEFAULT_OUTPUT_ROOT = "../filter_workflow/candidates/source_level_vis"

# frequency points (MHz) to check
FREQ_LIST_MHZ: List[float] = [1140.6040, 1066.6787, 1324.9838, 1148.4167]

# window config
FIXED_W2_MHZ = 0.00192
DEFAULT_DPI = 300
DEFAULT_FMT = "png"  # png/jpg/jpeg/pdf
VMIN_PCT = 2.0
VMAX_PCT = 98.0
BEAM_ID = 1  # M01
FIGSIZE = (20, 8)


# -----------------------------
# Utilities
# -----------------------------
def make_timestamp_dir(root: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = root / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def bounds(center_mhz: float, width_mhz: float) -> Tuple[float, float]:
    return center_mhz - 0.5 * width_mhz, center_mhz + 0.5 * width_mhz


def load_waterfall_slice(fname: Path, f_start: float, f_stop: float) -> Waterfall:
    return Waterfall(str(fname), f_start=float(f_start), f_stop=float(f_stop))


def extract_2d_data(wf: Waterfall) -> np.ndarray:
    arr = np.array(wf.data)
    arr = np.squeeze(arr)

    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        # squeeze singleton axis
        for ax in range(3):
            if arr.shape[ax] == 1:
                arr2 = np.squeeze(arr, axis=ax)
                if arr2.ndim == 2:
                    return arr2
        # fallback
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


def plot_single_panel(arr_tf: np.ndarray, f_start: float, f_stop: float, tsamp: float, title: str, save_path: Path,
                      dpi: int, fmt: str, figsize: Tuple[float, float], vmin: float, vmax: float, ):
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


def infer_group_id_from_path(p: Path, beam_id: int) -> Optional[str]:
    """
    Infer group_id from filename by finding the substring right before _Mxx.

    Example filename:
      ".../GJ273_20210525_M01_xx.fil"
    group_id => "GJ273_20210525"

    If pattern not found, return None.
    """
    name = p.name
    mtag = f"_M{beam_id:02d}"
    if mtag not in name:
        return None
    # take everything before "_Mxx"
    group_id = name.split(mtag)[0]
    group_id = group_id.strip("_- ")
    return group_id if group_id else None


def build_group_file_map(folder: Path, beam_id: int) -> Dict[str, Path]:
    """
    Map: group_id -> first matching file path for that group_id & beam_id
    """
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
        # first match wins
        if gid not in out:
            out[gid] = f
    return out


def run_source_level_vis():
    xx_dir = Path(DEFAULT_XX_DIR).expanduser().resolve()
    yy_dir = Path(DEFAULT_YY_DIR).expanduser().resolve()
    out_root = Path(DEFAULT_OUTPUT_ROOT).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if not xx_dir.exists():
        print(f"[\033[31mError\033[0m] XX_DIR not found: {xx_dir}")
        sys.exit(1)
    if not yy_dir.exists():
        print(f"[\033[31mError\033[0m] YY_DIR not found: {yy_dir}")
        sys.exit(1)

    ts_dir = make_timestamp_dir(out_root)

    print("\n==============================")
    print("[\033[32mInfo\033[0m] SOURCE-LEVEL VIS CHECK")
    print("==============================")
    print(f"XX_DIR     : {xx_dir}")
    print(f"YY_DIR     : {yy_dir}")
    print(f"Output dir : {ts_dir}")
    print(f"Beam       : M{BEAM_ID:02d}")
    print(f"Window     : fixed_width = {FIXED_W2_MHZ} MHz")
    print(f"Freq points: {len(FREQ_LIST_MHZ)}")
    print(f"Format/DPI : {DEFAULT_FMT}/{DEFAULT_DPI}")

    # Build maps: group_id -> file
    xx_map = build_group_file_map(xx_dir, BEAM_ID)
    yy_map = build_group_file_map(yy_dir, BEAM_ID)

    # Intersect group_ids that have both XX & YY
    group_ids = sorted(set(xx_map.keys()) & set(yy_map.keys()))
    print(f"Found XX M01 groups: {len(xx_map)}")
    print(f"Found YY M01 groups: {len(yy_map)}")
    print(f"Usable groups (XX&YY): {len(group_ids)}")

    meta_rows: List[Dict] = []

    for fc in progress(FREQ_LIST_MHZ, total=len(FREQ_LIST_MHZ), desc="Frequencies"):
        freq_folder = ts_dir / f"freq_{fc:.6f}MHz"
        freq_folder.mkdir(parents=True, exist_ok=True)

        f_start, f_stop = bounds(fc, FIXED_W2_MHZ)

        for gid in progress(group_ids, total=len(group_ids), desc=f"Groups @ {fc:.6f}MHz"):
            xx_file = xx_map.get(gid)
            yy_file = yy_map.get(gid)

            row = {"group_id": gid, "beam_id": BEAM_ID, "freq_center_mhz": fc, "window": "fixed_width",
                   "window_width_mhz": FIXED_W2_MHZ, "f_start_mhz": f_start, "f_stop_mhz": f_stop,
                   "file_xx": str(xx_file) if xx_file else "", "file_yy": str(yy_file) if yy_file else "",
                   "rendered": "0", "out_img": "", "err": "", }

            if xx_file is None or yy_file is None:
                row["err"] = "missing_xx_or_yy"
                meta_rows.append(row)
                continue

            try:
                wf_xx = load_waterfall_slice(xx_file, f_start, f_stop)
                wf_yy = load_waterfall_slice(yy_file, f_start, f_stop)

                arr_xx = extract_2d_data(wf_xx)
                arr_yy = extract_2d_data(wf_yy)

                # align shapes (safe)
                t = min(arr_xx.shape[0], arr_yy.shape[0])
                f = min(arr_xx.shape[1], arr_yy.shape[1])
                arr_i = arr_xx[:t, :f] + arr_yy[:t, :f]

                tsamp = get_tsamp_seconds(wf_xx)
                vmin, vmax = robust_vmin_vmax(arr_i)

                base = f"{gid}_M{BEAM_ID:02d}_fc{fc:.6f}MHz_fixed_width_I"
                save_path = freq_folder / base
                title = f"I | group={gid} | M{BEAM_ID:02d} | f0={fc:.6f} MHz | fixed_width"

                plot_single_panel(arr_tf=arr_i, f_start=f_start, f_stop=f_stop, tsamp=tsamp, title=title,
                                  save_path=save_path, dpi=DEFAULT_DPI, fmt=DEFAULT_FMT, figsize=FIGSIZE, vmin=vmin,
                                  vmax=vmax, )

                row["rendered"] = "1"
                row["out_img"] = str(save_path.with_suffix(f".{DEFAULT_FMT}"))
                meta_rows.append(row)

            except Exception as e:
                row["err"] = repr(e)
                meta_rows.append(row)

    meta_df = pd.DataFrame(meta_rows)
    meta_path = ts_dir / "source_level_meta.csv"
    meta_df.to_csv(meta_path, index=False)

    print(f"\n[\033[32mInfo\033[0m] Metadata written: {meta_path}")
    print("[\033[32mInfo\033[0m] Done!")


if __name__ == "__main__":
    run_source_level_vis()
