# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FAST SETI On-Off Veto (Main Workflow)

Rules:
- Only beam_id == 1 is ON-source.
- All other beams are OFF-source.
- Veto criterion: match by Uncorrected_Frequency within tolerance = 3 * Δν
  where Δν = 7.5 Hz, so tolerance = 22.5 Hz = 2.25e-5 MHz.
- No other constraints are applied.

Important:
- Veto is performed within each group_id (same observation group).
  This is a structural grouping, not an additional physical constraint.
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd

# ====== Configs ======
# DEFAULT_INPUT_TOTAL_CSV = (
#     "./filter_workflow/events/20260103_190203_f1050-1450_conf0.8_gSNR1500_SNR10/total/total_f1050-1450_conf0.8_gSNR1500_SNR10.csv")
# DEFAULT_INPUT_TOTAL_CSV = (
#     "./filter_workflow/events/20260103_192202_f1050-1450_conf0.8_gSNR1000_SNR10/total/total_f1050-1450_conf0.8_gSNR1000_SNR10.csv")
DEFAULT_INPUT_TOTAL_CSV = (
    "./filter_workflow/events/20260103_194509_f1050-1450_conf0.8_gSNR500_SNR10/total/total_f1050-1450_conf0.8_gSNR500_SNR10.csv")
OUTPUT_ROOT = "./filter_workflow/candidates"

ON_BEAM_ID = 1

DELTA_NU_HZ = 7.5
TOL_MULTIPLIER = 3.0

# Convert tolerance (Hz) -> MHz: 1 MHz = 1e6 Hz
TOL_HZ = DELTA_NU_HZ * TOL_MULTIPLIER
TOL_MHZ = TOL_HZ * 1e-6  # 22.5 Hz -> 2.25e-5 MHz


def ensure_numeric(series: pd.Series) -> pd.Series:
    """Convert to numeric safely (coerce errors to NaN)."""
    return pd.to_numeric(series, errors="coerce")


def make_output_dir(base: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"{ts}_veto_dnu{DELTA_NU_HZ}Hz_x{int(TOL_MULTIPLIER)}"
    out_dir = base / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def has_match_within_tolerance(off_freqs_sorted: np.ndarray, on_freqs: np.ndarray, tol_mhz: float) -> np.ndarray:
    """
    For each on_freq, determine whether there exists any off_freq such that |off - on| <= tol_mhz.
    off_freqs_sorted must be sorted ascending and finite.
    Returns a boolean array aligned with on_freqs.
    """
    if off_freqs_sorted.size == 0:
        return np.zeros_like(on_freqs, dtype=bool)

    idx = np.searchsorted(off_freqs_sorted, on_freqs, side="left")
    veto = np.zeros_like(on_freqs, dtype=bool)

    # Check candidate neighbor on the right (idx)
    right_ok = (idx < off_freqs_sorted.size)
    if np.any(right_ok):
        d_right = np.abs(off_freqs_sorted[idx[right_ok]] - on_freqs[right_ok])
        veto[right_ok] |= (d_right <= tol_mhz)

    # Check candidate neighbor on the left (idx-1)
    left_ok = (idx > 0)
    if np.any(left_ok):
        d_left = np.abs(off_freqs_sorted[idx[left_ok] - 1] - on_freqs[left_ok])
        veto[left_ok] |= (d_left <= tol_mhz)

    return veto


def main():
    ap = argparse.ArgumentParser(
        description="On-Off veto using frequency tolerance (3x 7.5 Hz) on Uncorrected_Frequency.")
    ap.add_argument(
        "--input_csv",
        type=str,
        default=DEFAULT_INPUT_TOTAL_CSV,
        help="Path to total_*.csv produced by the filter stage (default: edit DEFAULT_INPUT_TOTAL_CSV).",
    )
    args = ap.parse_args()

    in_csv = Path(args.input_csv).expanduser().resolve()
    if not in_csv.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {in_csv}")

    print("\n==============================")
    print("[\033[32mInfo\033[0m] ON-OFF VETO")
    print("==============================")
    print(f"[\033[32mInfo\033[0m] Input CSV: {in_csv}")
    print(f"[\033[32mInfo\033[0m] ON beam_id: {ON_BEAM_ID}")
    print(f"[\033[32mInfo\033[0m] Δν: {DELTA_NU_HZ} Hz | tolerance: {TOL_HZ} Hz = {TOL_MHZ:.8f} MHz")
    print(f"[\033[32mInfo\033[0m] Matching field: Uncorrected_Frequency (MHz)")
    print(f"[\033[32mInfo\033[0m] No other constraints: TRUE")

    df = pd.read_csv(in_csv)

    required_cols = ["group_id", "beam_id", "Uncorrected_Frequency"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}. Columns present: {list(df.columns)}")

    # Normalize types
    df["beam_id"] = ensure_numeric(df["beam_id"])
    df["Uncorrected_Frequency"] = ensure_numeric(df["Uncorrected_Frequency"])

    # Drop rows without valid frequency or beam_id
    before_drop = len(df)
    df = df.dropna(subset=["beam_id", "Uncorrected_Frequency", "group_id"]).copy()
    after_drop = len(df)
    if after_drop != before_drop:
        print(
            f"[\033[32mInfo\033[0m] Dropped {before_drop - after_drop} rows due to NaN in beam_id/frequency/group_id.")

    # Prepare output directory
    out_root = Path(OUTPUT_ROOT).expanduser().resolve()
    out_dir = make_output_dir(out_root)
    print(f"[\033[32mInfo\033[0m] Output directory: {out_dir}")

    # Split ON/OFF
    on_df = df[df["beam_id"] == ON_BEAM_ID].copy()
    off_df = df[df["beam_id"] != ON_BEAM_ID].copy()

    print(f"[\033[32mInfo\033[0m] Rows total: {len(df)} | ON rows: {len(on_df)} | OFF rows: {len(off_df)}")

    # Edge cases
    if len(on_df) == 0:
        print("[\033[32mInfo\033[0m] No ON rows found. Writing empty outputs.")
        (out_dir / "on_candidates.csv").write_text("", encoding="utf-8")
        (out_dir / "on_vetoed.csv").write_text("", encoding="utf-8")
        stats_df = pd.DataFrame([{
            "group_id": "__TOTAL__",
            "on_rows": 0,
            "off_rows": int(len(off_df)),
            "vetoed_on_rows": 0,
            "kept_on_rows": 0,
            "veto_rate": 0.0,
        }])
        stats_df.to_csv(out_dir / "veto_stats.csv", index=False)
        print("\n[\033[32mInfo\033[0m] Done!")
        return

    # Perform veto within each group_id
    veto_flags = np.zeros(len(on_df), dtype=bool)
    on_index = on_df.index.to_numpy()

    stats_rows: List[Dict] = []

    grouped_on = on_df.groupby("group_id", sort=True)
    for gid, on_g in grouped_on:
        off_g = off_df[off_df["group_id"] == gid]

        on_freqs = on_g["Uncorrected_Frequency"].to_numpy(dtype=float)
        off_freqs = off_g["Uncorrected_Frequency"].to_numpy(dtype=float)

        # Keep finite values only
        on_finite_mask = np.isfinite(on_freqs)
        off_freqs = off_freqs[np.isfinite(off_freqs)]
        off_freqs.sort()

        # Compute veto for finite ON freqs
        veto_g = np.zeros_like(on_freqs, dtype=bool)
        if np.any(on_finite_mask):
            veto_g[on_finite_mask] = has_match_within_tolerance(off_freqs, on_freqs[on_finite_mask], TOL_MHZ)

        # Write back to global veto_flags aligned by on_df index
        idx_positions = np.searchsorted(on_index, on_g.index.to_numpy())
        veto_flags[idx_positions] = veto_g

        vetoed_count = int(veto_g.sum())
        on_count = int(len(on_g))
        off_count = int(len(off_g))
        kept_count = on_count - vetoed_count
        veto_rate = (vetoed_count / on_count) if on_count > 0 else 0.0

        stats_rows.append({
            "group_id": gid,
            "on_rows": on_count,
            "off_rows": off_count,
            "vetoed_on_rows": vetoed_count,
            "kept_on_rows": kept_count,
            "veto_rate": veto_rate,
        })

    # Build outputs
    on_df_sorted = on_df.copy()
    on_df_sorted["_veto_flag_tmp_"] = veto_flags  # temporary internal marker

    vetoed_df = on_df_sorted[on_df_sorted["_veto_flag_tmp_"]].drop(columns=["_veto_flag_tmp_"])
    candidates_df = on_df_sorted[~on_df_sorted["_veto_flag_tmp_"]].drop(columns=["_veto_flag_tmp_"])

    candidates_path = out_dir / "on_candidates.csv"
    vetoed_path = out_dir / "on_vetoed.csv"
    candidates_df.to_csv(candidates_path, index=False)
    vetoed_df.to_csv(vetoed_path, index=False)

    # Stats
    stats_df = pd.DataFrame(stats_rows).sort_values("group_id").reset_index(drop=True)

    total_on = int(stats_df["on_rows"].sum())
    total_off = int(stats_df["off_rows"].sum())
    total_vetoed = int(stats_df["vetoed_on_rows"].sum())
    total_kept = int(stats_df["kept_on_rows"].sum())
    total_rate = (total_vetoed / total_on) if total_on > 0 else 0.0

    stats_df = pd.concat([
        stats_df,
        pd.DataFrame([{
            "group_id": "__TOTAL__",
            "on_rows": total_on,
            "off_rows": total_off,
            "vetoed_on_rows": total_vetoed,
            "kept_on_rows": total_kept,
            "veto_rate": total_rate,
        }])
    ], ignore_index=True)

    stats_path = out_dir / "veto_stats.csv"
    stats_df.to_csv(stats_path, index=False)

    print("\n==============================")
    print("[\033[32mInfo\033[0m] RESULTS")
    print("==============================")
    print(f"[\033[32mInfo\033[0m] Candidates (ON kept): {len(candidates_df)} -> {candidates_path}")
    print(f"[\033[32mInfo\033[0m] Vetoed (ON removed): {len(vetoed_df)} -> {vetoed_path}")
    print(f"[\033[32mInfo\033[0m] Stats: {stats_path}")
    print("\n[\033[32mInfo\033[0m] Done!")


if __name__ == "__main__":
    main()
