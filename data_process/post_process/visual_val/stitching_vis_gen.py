#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FAST SETI Stitch Case Visualizer (standalone)

Input:
- A stitching output directory produced by your stitch script, containing:
    <stitch_out_dir>/case_summary.csv
    <stitch_out_dir>/case_members.csv
    <stitch_out_dir>/stats/case_stats.csv

Output (style aligned with your source-level vis):
<output_root>/<timestamp>/
  case_001_<group>_Mxx/
    <group>_Mxx_typical_case_001_3panel.png
  case_002_...
  case_vis_meta.csv

Rules:
- Typical case selection is based on summary inversion (clean-first).
- Only 2-member cases are visualized (n_members == 2).
- Must be adjacent patches: same cell_row, abs(cell_col diff) == 1.
- No multi-beam sweep, no polarization outputs: only I = XX + YY.
- 3 panels, 2 rows:
    Row 1: left patch view | right patch view (boundary centered)
           (each panel shades the opposite side to emphasize patch-specific view)
    Row 2: stitched view (from case_summary box), aligned window & vmin/vmax

Notes:
- Time coordinates might be seconds OR indices. We auto-detect and convert to seconds when needed.
"""

import argparse
import glob
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

try:
    from blimpy import Waterfall
except Exception:
    Waterfall = None  # type: ignore

# ====== tqdm progress bar (optional) ======
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None


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


# ====== Default Configs======
DEFAULT_YY_DIR = "/data/Raid0/obs_data/33exoplanets/yy/"
DEFAULT_XX_DIR = "/data/Raid0/obs_data/33exoplanets/xx/"
DEFAULT_OUTPUT_ROOT = "../filter_workflow/candidates/stitching_vis"

DEFAULT_DPI = 300
DEFAULT_FMT = "png"
VMIN_PCT = 2.0
VMAX_PCT = 98.0

# tight window padding
FREQ_PAD_FRAC = 0.25
MIN_FREQ_PAD_MHZ = 0.0002
TIME_PAD_FRAC = 0.25
MIN_TIME_PAD_S = 1.0

# overall aspect H:W ~= 1:2  -> figsize (W,H) = (20,10)
CASE_FIGSIZE = (20, 10)

# semi-transparent shading to show "this panel is this patch"
SHADE_ALPHA = 0.25


def make_timestamp_dir(root: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = root / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def infer_group_id_from_path(p: Path, beam_id: int) -> Optional[str]:
    """
    Infer group_id from filename by taking substring before _Mxx.
    Example:
      ".../GJ273_20210525_M01_xx.fil" -> "GJ273_20210525"
    """
    name = p.name
    mtag = f"_M{beam_id:02d}"
    if mtag not in name:
        return None
    gid = name.split(mtag)[0].strip("_- ")
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


def extract_2d_data(wf: "Waterfall") -> np.ndarray:
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


def get_tsamp_seconds(wf: "Waterfall") -> float:
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


def load_i_slice(xx_file: Path, yy_file: Path, f_start: float, f_stop: float) -> Tuple[np.ndarray, float]:
    if Waterfall is None:
        raise RuntimeError("blimpy is not available; cannot run visualization.")
    wf_xx = Waterfall(str(xx_file), f_start=float(f_start), f_stop=float(f_stop))
    wf_yy = Waterfall(str(yy_file), f_start=float(f_start), f_stop=float(f_stop))
    arr_xx = extract_2d_data(wf_xx)
    arr_yy = extract_2d_data(wf_yy)

    t = min(arr_xx.shape[0], arr_yy.shape[0])
    f = min(arr_xx.shape[1], arr_yy.shape[1])
    arr_i = arr_xx[:t, :f] + arr_yy[:t, :f]

    tsamp = get_tsamp_seconds(wf_xx)
    return arr_i, tsamp


def maybe_convert_time_to_seconds(t0: float, t1: float, tsamp: float, total_seconds: float) -> Tuple[float, float]:
    """
    Heuristic:
    - If t looks much larger than total_seconds -> treat as index, convert by tsamp.
    """
    if max(t0, t1) > total_seconds * 1.2:
        return t0 * tsamp, t1 * tsamp
    return t0, t1


def crop_time(arr_tf: np.ndarray, tsamp: float, t_start: float, t_stop: float) -> Tuple[np.ndarray, float, float]:
    n_t = arr_tf.shape[0]
    total_seconds = tsamp * n_t

    t_start2, t_stop2 = maybe_convert_time_to_seconds(t_start, t_stop, tsamp, total_seconds)
    t0 = max(0.0, float(min(t_start2, t_stop2)))
    t1 = min(float(max(t_start2, t_stop2)), float(total_seconds))

    i0 = int(np.floor(t0 / tsamp))
    i1 = int(np.ceil(t1 / tsamp))
    i0 = max(0, min(i0, n_t))
    i1 = max(i0 + 1, min(i1, n_t))

    return arr_tf[i0:i1, :], i0 * tsamp, i1 * tsamp


def interval_len(a: float, b: float) -> float:
    return float(abs(b - a))


def draw_box(ax, f0: float, f1: float, t0: float, t1: float, lw: float = 2.5):
    x = min(f0, f1)
    w = abs(f1 - f0)
    y = min(t0, t1)
    h = abs(t1 - t0)
    ax.add_patch(Rectangle((x, y), w, h, fill=False, linewidth=lw))


def shade_other_side(ax, f_start: float, f_stop: float, fb: float, keep_side: str):
    """
    keep_side: "L" or "R"
    shade opposite side to emphasize patch-specific panel.
    """
    if keep_side == "L":
        x0, x1 = fb, f_stop
    else:
        x0, x1 = f_start, fb
    if x1 > x0:
        ax.add_patch(Rectangle((x0, ax.get_ylim()[0]), x1 - x0, ax.get_ylim()[1] - ax.get_ylim()[0],
                               facecolor="white", alpha=SHADE_ALPHA, edgecolor="none", zorder=3))


def plot_case_3panel(arr_tf: np.ndarray, f_start: float, f_stop: float, t0: float, t1: float, vmin: float, vmax: float,
                     fb: float, title_left: str, title_right: str, title_bottom: str,
                     left_box: Tuple[float, float, float, float], right_box: Tuple[float, float, float, float],
                     stitched_box: Tuple[float, float, float, float], save_path: Path, dpi: int, fmt: str, ):
    fig = plt.figure(figsize=CASE_FIGSIZE)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], width_ratios=[1.0, 1.0])

    axL = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 1])
    axB = fig.add_subplot(gs[1, :])

    axes = [(axL, title_left), (axR, title_right), (axB, title_bottom)]

    for ax, ttl in axes:
        im = ax.imshow(
            arr_tf,
            aspect="auto",
            origin="lower",
            extent=[f_start, f_stop, t0, t1],
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Time (s)")
        ax.set_title(ttl)
        ax.axvline(fb, linestyle="--", linewidth=2.0)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Intensity (arb.)")

    # draw boxes
    draw_box(axL, left_box[0], left_box[1], left_box[2], left_box[3])
    draw_box(axR, right_box[0], right_box[1], right_box[2], right_box[3])
    draw_box(axB, stitched_box[0], stitched_box[1], stitched_box[2], stitched_box[3], lw=3.0)

    # shade opposite sides on top row (after axes extents set)
    shade_other_side(axL, f_start, f_stop, fb, keep_side="L")
    shade_other_side(axR, f_start, f_stop, fb, keep_side="R")

    fig.tight_layout()
    fig.savefig(save_path.with_suffix(f".{fmt}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def build_case_key(df: pd.DataFrame, use_stitched_cols: bool) -> pd.Series:
    """
    Key: group_id + beam_id + csv_id + class_id + stitched_freq_start/end + stitched_time_start/end
    """
    if use_stitched_cols:
        sf0 = df["stitched_freq_start"]
        sf1 = df["stitched_freq_end"]
        st0 = df["stitched_time_start"]
        st1 = df["stitched_time_end"]
    else:
        # case_summary uses freq/time columns as stitched already
        sf0 = df["freq_start"]
        sf1 = df["freq_end"]
        st0 = df["time_start"]
        st1 = df["time_end"]

    return (
            df["group_id"].astype(str) + "||" +
            df["beam_id"].astype(str) + "||" +
            df["csv_id"].astype(str) + "||" +
            df["class_id"].astype(str) + "||" +
            sf0.astype(str) + "||" +
            sf1.astype(str) + "||" +
            st0.astype(str) + "||" +
            st1.astype(str)
    )


def select_typical_cases(sum_df: pd.DataFrame, mem_df: pd.DataFrame, stats_df: pd.DataFrame,
                         n_cases: int) -> pd.DataFrame:
    """
    Clean-first:
      - n_members == 2
      - adjacent patches in members: same cell_row, abs(col diff) == 1
      - rank by: confidence desc, gSNR desc, SNR desc, stitched bandwidth desc
    """
    s = stats_df.copy()
    s = s[s["n_members"] == 2].copy()
    if s.empty:
        return s

    # build keys
    s["__key"] = build_case_key(s, use_stitched_cols=True)
    sum2 = sum_df.copy()
    sum2["__key"] = build_case_key(sum2, use_stitched_cols=False)

    merged = s.merge(
        sum2[["__key", "confidence", "gSNR", "SNR", "DriftRate", "Uncorrected_Frequency",
              "freq_start", "freq_end", "time_start", "time_end", "group_id", "beam_id", "csv_id", "class_id"]],
        on="__key",
        how="left"
    )

    # adjacency check via members
    mem2 = mem_df.copy()
    mem2["__key"] = build_case_key(mem2, use_stitched_cols=True)

    ok_keys = []
    for key in merged["__key"].tolist():
        sub = mem2[mem2["__key"] == key]
        if len(sub) != 2:
            continue
        rset = sorted(sub["cell_row"].astype(int).unique().tolist())
        cset = sorted(sub["cell_col"].astype(int).unique().tolist())
        if len(rset) != 1 or len(cset) != 2:
            continue
        if abs(cset[1] - cset[0]) != 1:
            continue
        ok_keys.append(key)

    merged = merged[merged["__key"].isin(ok_keys)].copy()
    if merged.empty:
        return merged

    merged["stitched_bw"] = (merged["freq_end"] - merged["freq_start"]).abs()

    merged = merged.sort_values(
        by=["confidence", "gSNR", "SNR", "stitched_bw"],
        ascending=[False, False, False, False],
    ).head(int(n_cases)).copy()

    return merged


def main():
    ap = argparse.ArgumentParser(description="Standalone stitch-case visualizer (reads stitch outputs).")
    ap.add_argument("--stitch_out_dir", type=str, required=True,
                    help="Stitch output directory containing case_summary.csv, case_members.csv, stats/case_stats.csv.")
    ap.add_argument("--xx_dir", type=str, default=DEFAULT_XX_DIR, help="XX .fil/.h5 folder.")
    ap.add_argument("--yy_dir", type=str, default=DEFAULT_YY_DIR, help="YY .fil/.h5 folder.")
    ap.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT,
                    help="Output root (timestamp folder will be created inside).")
    ap.add_argument("--n_cases", type=int, default=10, help="Number of typical cases to visualize (default=10).")
    ap.add_argument("--dpi", type=int, default=DEFAULT_DPI, help="Figure DPI (default=300).")
    ap.add_argument("--fmt", type=str, default=DEFAULT_FMT, help="Figure format: png/jpg/jpeg/pdf (default=png).")
    args = ap.parse_args()

    if Waterfall is None:
        print("[\033[31mError\033[0m] blimpy not available. Please install blimpy in your env.")
        return

    stitch_dir = Path(args.stitch_out_dir).expanduser().resolve()
    sum_path = stitch_dir / "case_summary.csv"
    mem_path = stitch_dir / "case_members.csv"
    stats_path = stitch_dir / "stats" / "case_stats.csv"

    if not sum_path.exists():
        print(f"[\033[31mError\033[0m] Missing: {sum_path}")
        return
    if not mem_path.exists():
        print(f"[\033[31mError\033[0m] Missing: {mem_path}")
        return
    if not stats_path.exists():
        print(f"[\033[31mError\033[0m] Missing: {stats_path}")
        return

    xx_dir = Path(args.xx_dir).expanduser().resolve()
    yy_dir = Path(args.yy_dir).expanduser().resolve()
    if not xx_dir.exists() or not yy_dir.exists():
        print(f"[\033[31mError\033[0m] XX/YY dirs not found. XX={xx_dir} YY={yy_dir}")
        return

    out_root = Path(args.output_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    ts_dir = make_timestamp_dir(out_root)
    print(f"[\033[32mInfo\033[0m] Output: {ts_dir}")

    sum_df = pd.read_csv(sum_path)
    mem_df = pd.read_csv(mem_path)
    stats_df = pd.read_csv(stats_path)

    # normalize dtypes
    for col in ["beam_id", "cell_row", "cell_col", "class_id", "n_members"]:
        for df in [sum_df, mem_df, stats_df]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    picked = select_typical_cases(sum_df, mem_df, stats_df, n_cases=int(args.n_cases))
    if picked.empty:
        print("[\033[32mInfo\033[0m] No eligible typical 2-member adjacent cases found.")
        return

    # build file maps for needed beams
    beams = sorted(set(int(b) for b in picked["beam_id"].dropna().tolist()))
    xx_maps = {b: build_group_file_map(xx_dir, b) for b in beams}
    yy_maps = {b: build_group_file_map(yy_dir, b) for b in beams}

    # prepare members with key
    mem2 = mem_df.copy()
    mem2["__key"] = build_case_key(mem2, use_stitched_cols=True)

    meta_rows: List[Dict[str, Any]] = []

    for rank, row in enumerate(progress(picked.itertuples(index=False), total=len(picked), desc="Render cases"),
                               start=1):
        gid = str(getattr(row, "group_id"))
        bid = int(getattr(row, "beam_id"))
        cid = str(getattr(row, "csv_id"))
        key = str(getattr(row, "__key"))

        conf = float(getattr(row, "confidence"))
        gsnr = float(getattr(row, "gSNR"))
        snr = float(getattr(row, "SNR"))
        drift = float(getattr(row, "DriftRate"))
        uncorr = float(getattr(row, "Uncorrected_Frequency"))

        f0 = float(getattr(row, "freq_start"))
        f1 = float(getattr(row, "freq_end"))
        t0s = float(getattr(row, "time_start"))
        t1s = float(getattr(row, "time_end"))

        sub = mem2[mem2["__key"] == key].copy()
        if len(sub) != 2:
            continue

        sub["cell_col"] = sub["cell_col"].astype(int)
        left = sub.sort_values("cell_col").iloc[0]
        right = sub.sort_values("cell_col").iloc[1]

        # boundary frequency (center line)
        fb = 0.5 * (float(left["freq_max"]) + float(right["freq_min"]))

        # tight frequency window centered on boundary
        bw = interval_len(f0, f1)
        fpad = max(bw * FREQ_PAD_FRAC, MIN_FREQ_PAD_MHZ)
        halfw = max(abs(fb - min(f0, f1)), abs(max(f0, f1) - fb)) + fpad
        f_start = fb - halfw
        f_stop = fb + halfw

        # files
        xx_file = xx_maps.get(bid, {}).get(gid)
        yy_file = yy_maps.get(bid, {}).get(gid)

        case_dir = ts_dir / f"case_{rank:03d}_{gid}_M{bid:02d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        save_path = case_dir / f"{gid}_M{bid:02d}_typical_case_{rank:03d}_3panel"

        meta = {
            "rank": rank,
            "group_id": gid,
            "beam_id": bid,
            "csv_id": cid,
            "confidence": conf,
            "gSNR": gsnr,
            "SNR": snr,
            "DriftRate": drift,
            "Uncorrected_Frequency": uncorr,
            "stitched_freq_start": f0,
            "stitched_freq_end": f1,
            "stitched_time_start": t0s,
            "stitched_time_end": t1s,
            "boundary_freq_mhz": fb,
            "f_vis_start": f_start,
            "f_vis_stop": f_stop,
            "xx_file": str(xx_file) if xx_file else "",
            "yy_file": str(yy_file) if yy_file else "",
            "rendered": "0",
            "out_img": "",
            "err": "",
        }

        if xx_file is None or yy_file is None:
            meta["err"] = "missing_xx_or_yy"
            meta_rows.append(meta)
            continue

        try:
            arr_i_full, tsamp = load_i_slice(xx_file, yy_file, f_start=f_start, f_stop=f_stop)

            # tight time window around stitched span (pad)
            tspan = abs(t1s - t0s)
            tpad = max(tspan * TIME_PAD_FRAC, MIN_TIME_PAD_S)
            t_req0 = min(t0s, t1s) - tpad
            t_req1 = max(t0s, t1s) + tpad

            arr_i, t0_plot, t1_plot = crop_time(arr_i_full, tsamp, t_req0, t_req1)
            vmin, vmax = robust_vmin_vmax(arr_i)

            total_seconds = tsamp * arr_i_full.shape[0]
            l_t0, l_t1 = maybe_convert_time_to_seconds(float(left["time_start"]), float(left["time_end"]), tsamp,
                                                       total_seconds)
            r_t0, r_t1 = maybe_convert_time_to_seconds(float(right["time_start"]), float(right["time_end"]), tsamp,
                                                       total_seconds)
            s_t0, s_t1 = maybe_convert_time_to_seconds(t0s, t1s, tsamp, total_seconds)

            left_box = (float(left["freq_start"]), float(left["freq_end"]), l_t0, l_t1)
            right_box = (float(right["freq_start"]), float(right["freq_end"]), r_t0, r_t1)
            stitched_box = (f0, f1, s_t0, s_t1)

            title_left = f"Left patch | {gid} | M{bid:02d} | conf={conf:.3f} gSNR={gsnr:.1f} SNR={snr:.2f}"
            title_right = f"Right patch | {gid} | M{bid:02d} | boundary centered"
            title_bottom = f"Stitched (summary) | DriftRate={drift:.6g} | UncorrF={uncorr:.6f} MHz"

            plot_case_3panel(
                arr_tf=arr_i,
                f_start=f_start,
                f_stop=f_stop,
                t0=t0_plot,
                t1=t1_plot,
                vmin=vmin,
                vmax=vmax,
                fb=fb,
                title_left=title_left,
                title_right=title_right,
                title_bottom=title_bottom,
                left_box=left_box,
                right_box=right_box,
                stitched_box=stitched_box,
                save_path=save_path,
                dpi=int(args.dpi),
                fmt=str(args.fmt),
            )

            meta["rendered"] = "1"
            meta["out_img"] = str(save_path.with_suffix(f".{args.fmt}"))
            meta_rows.append(meta)

        except Exception as e:
            meta["err"] = repr(e)
            meta_rows.append(meta)

    meta_df = pd.DataFrame(meta_rows)
    meta_path = ts_dir / "case_vis_meta.csv"
    meta_df.to_csv(meta_path, index=False)

    print(f"\n[\033[32mInfo\033[0m] Done. Outputs: {ts_dir}")
    print(f"[\033[32mInfo\033[0m] Meta CSV: {meta_path}")


if __name__ == "__main__":
    main()
