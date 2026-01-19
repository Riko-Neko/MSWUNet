# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FAST SETI Event Filter (Main Workflow)

What it does:
1) Scan a folder for CSV files (CSV filename is used as a unique identifier).
2) Print an overview grouped by "group_id" (prefix before "_M" in filename).
3) Apply pipeline filters:
   - Frequency band: 1.05–1.45 GHz (CSV columns are in MHz)
   - confidence >= 0.8
   - gSNR >= 1500
   - SNR >= 10
4) Write outputs into:
   ./filter_workflow/events/<timestamp>_f1050-1450_conf0.8_gSNR1500_SNR10/
   - Per-file filtered CSV: <original_stem>_out.csv
   - One merged summary CSV with extra columns:
       group_id: prefix before "_M"
       beam_id : integer parsed from "_Mxx" (e.g., M01 -> 1)
       csv_id  : original CSV filename
   - One group stats CSV: counts per group at each stage

Stitching directory support:
- If input folder contains "case_summary.csv", treat it as a stitching directory:
  - Only process this single file (case_summary.csv) for threshold filtering.
"""

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# ====== Configs of Pipeline ======
PIPELINE_COLUMNS = [
    "DriftRate", "SNR", "Uncorrected_Frequency", "freq_start", "freq_end", "class_id", "confidence",
    "cell_row", "cell_col", "gSNR", "freq_min", "freq_max", "time_start", "time_end", "mode"
]
DEFAULT_INPUT_FOLDER = ROOT / './data_process/post_process/filter_workflow/init'
# DEFAULT_INPUT_FOLDER = ROOT / './data_process/post_process/filter_workflow/stitching/20260111_175621_f1050-1450_bandonly_overlap0.2_tol7.5e-06_IoU0.5_skipCrowded'
FREQ_MIN_MHZ = 1050.0
FREQ_MAX_MHZ = 1450.0
THRESH_CONFIDENCE = 0.7
THRESH_GSNR = 500.0
THRESH_SNR = 10.0


def parse_group_and_beam(stem: str) -> Tuple[str, Optional[int]]:
    """
    Parse group_id and beam_id from a file stem.
    Expected patterns:
      <group>_M01...
      <group>_M19...
    group_id = part before '_M'
    beam_id  = integer digits right after '_M' (base-10)
    """
    if "_M" not in stem:
        return stem, None

    group_id = stem.split("_M", 1)[0]
    tail = stem.split("_M", 1)[1]
    m = re.match(r"(\d+)", tail)
    if not m:
        return group_id, None
    return group_id, int(m.group(1))


def resolve_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """
    Return the first existing column name in df among candidates.
    Also tries case-insensitive match if exact not found.
    """
    cols = list(df.columns)

    for c in candidates:
        if c in cols:
            return c

    lower_map = {str(col).lower(): col for col in cols}
    for c in candidates:
        key = c.lower()
        if key in lower_map:
            return lower_map[key]

    return None


def ensure_numeric(series: pd.Series) -> pd.Series:
    """Convert to numeric safely (coerce errors to NaN)."""
    return pd.to_numeric(series, errors="coerce")


def build_frequency_mask(df: pd.DataFrame) -> pd.Series:
    """
    Frequency band filter:
    - Prefer using freq_start/freq_end overlap logic if available:
        keep if [min(freq_start,freq_end), max(freq_start,freq_end)] overlaps [FREQ_MIN_MHZ, FREQ_MAX_MHZ]
    - Otherwise fallback to Uncorrected_Frequency as a center frequency:
        keep if center in [FREQ_MIN_MHZ, FREQ_MAX_MHZ]
    """
    col_f0 = resolve_column(df, ["Uncorrected_Frequency", "uncorrected_frequency", "f0", "frequency"])
    col_fs = resolve_column(df, ["freq_start", "FreqStart", "f_start", "start_freq"])
    col_fe = resolve_column(df, ["freq_end", "FreqEnd", "f_end", "end_freq"])

    if col_fs and col_fe:
        fs = ensure_numeric(df[col_fs])
        fe = ensure_numeric(df[col_fe])
        f_lo = pd.concat([fs, fe], axis=1).min(axis=1)
        f_hi = pd.concat([fs, fe], axis=1).max(axis=1)
        # Overlap condition
        return (f_hi >= FREQ_MIN_MHZ) & (f_lo <= FREQ_MAX_MHZ)

    if col_f0:
        f0 = ensure_numeric(df[col_f0])
        return (f0 >= FREQ_MIN_MHZ) & (f0 <= FREQ_MAX_MHZ)

    raise KeyError(
        "No usable frequency columns found. Need either (freq_start & freq_end) or Uncorrected_Frequency."
    )


def build_param_mask(df: pd.DataFrame) -> pd.Series:
    """Threshold filter for confidence, gSNR, SNR."""
    col_conf = resolve_column(df, ["confidence", "conf"])
    col_gsnr = resolve_column(df, ["gSNR", "gsnr"])
    col_snr = resolve_column(df, ["SNR", "snr"])

    missing = [name for name, col in [("confidence", col_conf), ("gSNR", col_gsnr), ("SNR", col_snr)] if col is None]
    if missing:
        raise KeyError(f"Missing required columns: {missing}. Columns present: {list(df.columns)}")

    conf = ensure_numeric(df[col_conf])
    gsnr = ensure_numeric(df[col_gsnr])
    snr = ensure_numeric(df[col_snr])

    return (conf >= THRESH_CONFIDENCE) & (gsnr >= THRESH_GSNR) & (snr >= THRESH_SNR)


def make_output_dir(base: Path, stitching_mode: bool) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"{ts}_f{int(FREQ_MIN_MHZ)}-{int(FREQ_MAX_MHZ)}_conf{THRESH_CONFIDENCE}_gSNR{int(THRESH_GSNR)}_SNR{int(THRESH_SNR)}"
    if stitching_mode:
        tag = f"{tag}_stitching"
    out_dir = base / "data_process" / "post_process" / "filter_workflow" / "events" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def main():
    ap = argparse.ArgumentParser(description="Main event filter workflow (FAST SETI pipeline).")
    ap.add_argument("--input_dir", type=str, default=DEFAULT_INPUT_FOLDER,
                    help="Folder containing CSV files (CSV is directly inside this folder).")
    args = ap.parse_args()

    in_dir = Path(args.input_dir).expanduser().resolve()
    if not in_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {in_dir}")

    # Stitching directory detection
    case_summary_path = in_dir / "case_summary.csv"
    stitching_mode = case_summary_path.is_file()

    if stitching_mode:
        csv_files = [case_summary_path]
    else:
        csv_files = sorted([p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() == ".csv"])

    if not csv_files:
        print(f"[\033[32mInfo\033[0m] No CSV files found in: {in_dir}")
        return

    # Overview grouping
    groups: Dict[str, List[Path]] = {}
    for p in csv_files:
        group_id, _beam = parse_group_and_beam(p.stem)
        groups.setdefault(group_id, []).append(p)

    print("\n==============================")
    print("[\033[32mInfo\033[0m] OVERVIEW")
    print("==============================")
    for group_id in sorted(groups.keys()):
        members = groups[group_id]
        ids = [m.name for m in members]
        print(f"\n[GROUP] {group_id}")
        for name in sorted(ids):
            print(f"  - {name}")
    print("\n[INFO] Total CSV files:", len(csv_files))
    print("[INFO] Total groups   :", len(groups))

    # Prepare output dir
    out_dir = make_output_dir(Path.cwd(), stitching_mode=stitching_mode)
    print("\n==============================")
    print("[\033[32mInfo\033[0m] OUTPUT")
    print("==============================")
    print(f"[\033[32mInfo\033[0m] Output directory: {out_dir}")

    merged_rows = []
    stats_rows = []

    # Process each file
    for p in csv_files:
        csv_id = p.name
        group_id, beam_id = parse_group_and_beam(p.stem)

        df = pd.read_csv(p)
        n_total = len(df)

        # Apply masks
        freq_mask = build_frequency_mask(df)
        df_freq = df[freq_mask].copy()
        n_freq = len(df_freq)

        param_mask = build_param_mask(df_freq)
        df_out = df_freq[param_mask].copy()
        n_out = len(df_out)

        # Write per-file output
        out_file = out_dir / f"{p.stem}_out.csv"
        df_out.to_csv(out_file, index=False)

        # Add to merged summary
        if n_out > 0:
            # In stitching mode, case_summary.csv already contains (group_id, beam_id, csv_id) per-row.
            # Keep original behavior for normal mode; for stitching mode, only add missing meta cols if absent.
            if "csv_id" not in df_out.columns:
                df_out.insert(0, "csv_id", csv_id)
            if "beam_id" not in df_out.columns:
                df_out.insert(0, "beam_id", beam_id if beam_id is not None else pd.NA)
            if "group_id" not in df_out.columns:
                df_out.insert(0, "group_id", group_id)

            merged_rows.append(df_out)

        # Collect stats
        stats_group_id = "stitching" if stitching_mode else group_id
        stats_beam_id = "" if stitching_mode else (beam_id if beam_id is not None else "")

        stats_rows.append(
            {"group_id": stats_group_id, "csv_id": csv_id, "beam_id": stats_beam_id,
             "rows_total": n_total, "rows_in_band": n_freq, "rows_after_thresholds": n_out})

        # print(f"[FILE] {csv_id} | total={n_total} | in_band={n_freq} | kept={n_out}")
        if (len(stats_rows) % 19) == 0:
            print(f"[\033[32mInfo\033[0m] Processed {len(stats_rows)}/{len(csv_files)} files...")

    # Write merged summary (TOTAL) into a subfolder with a parameter-tagged filename
    param_tag = f"f{int(FREQ_MIN_MHZ)}-{int(FREQ_MAX_MHZ)}_conf{THRESH_CONFIDENCE}_gSNR{int(THRESH_GSNR)}_SNR{int(THRESH_SNR)}"
    total_dir = out_dir / "total"
    total_dir.mkdir(parents=True, exist_ok=True)
    total_path = total_dir / f"total_{param_tag}.csv"

    if merged_rows:
        merged_df = pd.concat(merged_rows, ignore_index=True)

        # Ensure PIPELINE_COLUMNS exist (create missing as NA), then reorder columns.
        for col in PIPELINE_COLUMNS:
            if col not in merged_df.columns:
                merged_df[col] = pd.NA

        ordered_cols = PIPELINE_COLUMNS.copy()

        # Keep metadata columns as well (do not change other behavior), but keep PIPELINE_COLUMNS intact first.
        for meta_col in ["group_id", "beam_id", "csv_id"]:
            if meta_col in merged_df.columns:
                ordered_cols.append(meta_col)

        # Append any remaining columns (if any) to avoid accidental drops.
        for col in merged_df.columns:
            if col not in ordered_cols:
                ordered_cols.append(col)

        merged_df = merged_df[ordered_cols]
        merged_df.to_csv(total_path, index=False)
        print(f"\n[\033[32mInfo\033[0m] Total summary written: {total_path}")
    else:
        print("\n[\033[32mInfo\033[0m] No rows survived filters; total summary not created.")

    # Write stats (file-level)
    stats_df = pd.DataFrame(stats_rows)
    stats_file = out_dir / "file_stats.csv"
    stats_df.to_csv(stats_file, index=False)
    print(f"[\033[32mInfo\033[0m] File stats written: {stats_file}")

    # Group-level aggregation
    group_agg = (stats_df.groupby("group_id", as_index=False)[
                     ["rows_total", "rows_in_band", "rows_after_thresholds"]].sum().sort_values("group_id"))
    group_stats_file = out_dir / "group_stats.csv"
    group_agg.to_csv(group_stats_file, index=False)
    print(f"[\033[32mInfo\033[0m] Group stats written: {group_stats_file}")

    print("\n[\033[32mInfo\033[0m] Done!")


if __name__ == "__main__":
    main()
