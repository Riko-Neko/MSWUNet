# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FAST SETI Filter (SNR / Drift Rate / Frequency / Group / Beam)

Rules:
- Filters can be enabled independently: SNR / DR / Frequency / Group / Beam.
- Each dimension supports:
  (a) Range filter: [min, max]
  (b) Tolerance filter: |x - center| <= tol
- Range and tolerance can coexist for the same dimension (intersection).
- Group/Beam filters:
  - Default OFF.
  - When enabled, only rows whose group_id / beam_id are in predefined allow-lists are kept.

Default behavior:
  - Frequency filter ON with range [1050, 1450] MHz
  - SNR filter OFF
  - DR filter OFF
  - Group filter OFF
  - Beam filter OFF

Input/Output:
- Output CSV names follow the example-like split:
  - on_candidates.csv : rows kept after filtering
  - on_vetoed.csv     : rows removed by filtering
- No stats CSV is produced.

Output directory:
OUTPUT_ROOT = ROOT / "./data_process/post_process/analysis_out/{timestamp}_{config}_match"
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# ====== Configs ======
DEFAULT_INPUT_TOTAL_CSV = ROOT / "data_process/post_process/filter_workflow/events/20260108_165002_f1050-1450_conf0.7_gSNR1000_SNR10/total/total_f1050-1450_conf0.7_gSNR1000_SNR10.csv"

# Output dir
OUTPUT_ROOT = ROOT / "data_process/post_process/analysis_out"

COL_GROUP_ID = "group_id"
COL_BEAM_ID = "beam_id"
COL_FREQ_MHZ = "Uncorrected_Frequency"
COL_SNR = "SNR"
COL_DR_HZ_S = "DriftRate"

# Group/Beam allow lists
ALLOW_GROUP_IDS: List[str] = ["K2-155", "K2-18", "GJ-9066", "Ross-128"]
ALLOW_BEAM_IDS: List[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

# Drift-rate tolerance default = df/T
DF_HZ = 7.450580597
T_INT_S = 1183.26
DEFAULT_DR_TOL_HZ_S = DF_HZ / T_INT_S

# Default frequency range (MHz)
DEFAULT_F_MIN = 1050.0
DEFAULT_F_MAX = 1450.0


def ensure_numeric(series: pd.Series) -> pd.Series:
    """Convert to numeric safely (coerce errors to NaN)."""
    return pd.to_numeric(series, errors="coerce")


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def fmt_float(x: float, nd: int = 6) -> str:
    if x is None:
        return "None"
    return f"{x:.{nd}f}".rstrip("0").rstrip(".")


def build_cfg_tag(cfg: Dict) -> str:
    """
    Build a compact config tag for directory naming.
    Example:
      F[r1050-1450]_SNR[off]_DR[off]_G[off]_B[off]
      F[r1050-1450&t1140±0.001]_SNR[r10-50]_DR[t-0.1±0.75]_G[n3]_B[n1]
    """
    parts = []

    # Frequency
    if cfg["freq_enable"]:
        f_parts = []
        if cfg["freq_use_range"]:
            f_parts.append(f"r{fmt_float(cfg['freq_min'], 3)}-{fmt_float(cfg['freq_max'], 3)}")
        if cfg["freq_use_tol"]:
            f_parts.append(f"t{fmt_float(cfg['freq_center'], 6)}±{fmt_float(cfg['freq_tol'], 6)}")
        parts.append("F[" + "&".join(f_parts) + "]")
    else:
        parts.append("F[off]")

    # SNR
    if cfg["snr_enable"]:
        s_parts = []
        if cfg["snr_use_range"]:
            s_parts.append(f"r{fmt_float(cfg['snr_min'], 3)}-{fmt_float(cfg['snr_max'], 3)}")
        if cfg["snr_use_tol"]:
            s_parts.append(f"t{fmt_float(cfg['snr_center'], 3)}±{fmt_float(cfg['snr_tol'], 3)}")
        parts.append("SNR[" + "&".join(s_parts) + "]")
    else:
        parts.append("SNR[off]")

    # DR
    if cfg["dr_enable"]:
        d_parts = []
        if cfg["dr_use_range"]:
            d_parts.append(f"r{fmt_float(cfg['dr_min'], 6)}-{fmt_float(cfg['dr_max'], 6)}")
        if cfg["dr_use_tol"]:
            d_parts.append(f"t{fmt_float(cfg['dr_center'], 6)}±{fmt_float(cfg['dr_tol'], 6)}")
        parts.append("DR[" + "&".join(d_parts) + "]")
    else:
        parts.append("DR[off]")

    # Group/Beam
    if cfg["group_enable"]:
        parts.append(f"G[n{cfg['group_n']}]")
    else:
        parts.append("G[off]")

    if cfg["beam_enable"]:
        parts.append(f"B[n{cfg['beam_n']}]")
    else:
        parts.append("B[off]")

    return "_".join(parts)


def make_output_dir(base: Path, cfg_tag: str) -> Path:
    ts = now_tag()
    out_dir = base / f"{ts}_{cfg_tag}_match"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def apply_range_mask(x: np.ndarray, vmin: Optional[float], vmax: Optional[float]) -> np.ndarray:
    m = np.ones_like(x, dtype=bool)
    if vmin is not None:
        m &= (x >= vmin)
    if vmax is not None:
        m &= (x <= vmax)
    return m


def apply_tol_mask(x: np.ndarray, center: Optional[float], tol: Optional[float]) -> np.ndarray:
    if center is None or tol is None:
        raise ValueError("Tolerance filter requires both center and tol.")
    return np.isfinite(x) & (np.abs(x - center) <= tol)


def main():
    ap = argparse.ArgumentParser(
        description="FAST SETI filter: selectable SNR / DR / Frequency / Group / Beam filters with range/tolerance modes."
    )
    ap.add_argument(
        "--input_csv",
        type=str,
        default=str(DEFAULT_INPUT_TOTAL_CSV),
        help="Path to total_*.csv produced by the filter stage (default: DEFAULT_INPUT_TOTAL_CSV).",
    )

    # ---- Enable switches ----
    ap.add_argument("--enable_freq", action="store_true", default=True,
                    help="Enable frequency filter (default: ON).")
    ap.add_argument("--disable_freq", action="store_true", default=False,
                    help="Disable frequency filter.")

    ap.add_argument("--enable_snr", action="store_true", default=False,
                    help="Enable SNR filter (default: OFF).")

    ap.add_argument("--enable_dr", action="store_true", default=False,
                    help="Enable drift-rate filter (default: OFF).")

    # NEW: group/beam enable switches
    ap.add_argument("--enable_group", action="store_true", default=False,
                    help="Enable group_id allow-list filter (default: OFF).")
    ap.add_argument("--enable_beam", action="store_true", default=False,
                    help="Enable beam_id allow-list filter (default: OFF).")

    # ---- Frequency filter options ----
    ap.add_argument("--freq_use_range", action="store_true", default=True,
                    help="Use frequency range filter (default: ON).")
    ap.add_argument("--freq_min", type=float, default=DEFAULT_F_MIN,
                    help="Frequency range min in MHz (default: 1050).")
    ap.add_argument("--freq_max", type=float, default=DEFAULT_F_MAX,
                    help="Frequency range max in MHz (default: 1450).")

    ap.add_argument("--freq_use_tol", action="store_true", default=False,
                    help="Also apply frequency tolerance filter (default: OFF).")
    ap.add_argument("--freq_center", type=float, default=None,
                    help="Frequency tolerance center in MHz (required if --freq_use_tol).")
    ap.add_argument("--freq_tol", type=float, default=None,
                    help="Frequency tolerance in MHz (required if --freq_use_tol).")

    # ---- SNR filter options ----
    ap.add_argument("--snr_use_range", action="store_true", default=False,
                    help="Use SNR range filter (default: OFF).")
    ap.add_argument("--snr_min", type=float, default=None,
                    help="SNR range min.")
    ap.add_argument("--snr_max", type=float, default=None,
                    help="SNR range max.")

    ap.add_argument("--snr_use_tol", action="store_true", default=False,
                    help="Use SNR tolerance filter (default: OFF).")
    ap.add_argument("--snr_center", type=float, default=None,
                    help="SNR tolerance center (required if --snr_use_tol).")
    ap.add_argument("--snr_tol", type=float, default=None,
                    help="SNR tolerance (required if --snr_use_tol).")

    # ---- Drift-rate filter options ----
    ap.add_argument("--dr_use_tol", action="store_true", default=True,
                    help="Use drift-rate tolerance filter when DR is enabled (default: ON).")
    ap.add_argument("--dr_center", type=float, default=None,
                    help="Drift-rate tolerance center (Hz/s). Required if DR enabled and dr_use_tol.")
    ap.add_argument("--dr_tol", type=float, default=DEFAULT_DR_TOL_HZ_S,
                    help=f"Drift-rate tolerance (Hz/s). Default=df/T={DEFAULT_DR_TOL_HZ_S:.6f}.")

    ap.add_argument("--dr_use_range", action="store_true", default=False,
                    help="Also apply drift-rate range filter (default: OFF).")
    ap.add_argument("--dr_min", type=float, default=None,
                    help="Drift-rate range min (Hz/s).")
    ap.add_argument("--dr_max", type=float, default=None,
                    help="Drift-rate range max (Hz/s).")

    args = ap.parse_args()

    in_csv = Path(args.input_csv).expanduser().resolve()
    if not in_csv.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {in_csv}")

    # Effective enable flags
    freq_enable = bool(args.enable_freq) and (not args.disable_freq)
    snr_enable = bool(args.enable_snr)
    dr_enable = bool(args.enable_dr)
    group_enable = bool(args.enable_group)
    beam_enable = bool(args.enable_beam)

    # Prepare config dict for naming & logic
    cfg = dict(
        freq_enable=freq_enable,
        freq_use_range=bool(args.freq_use_range) if freq_enable else False,
        freq_min=args.freq_min if freq_enable else None,
        freq_max=args.freq_max if freq_enable else None,
        freq_use_tol=bool(args.freq_use_tol) if freq_enable else False,
        freq_center=args.freq_center,
        freq_tol=args.freq_tol,

        snr_enable=snr_enable,
        snr_use_range=bool(args.snr_use_range) if snr_enable else False,
        snr_min=args.snr_min,
        snr_max=args.snr_max,
        snr_use_tol=bool(args.snr_use_tol) if snr_enable else False,
        snr_center=args.snr_center,
        snr_tol=args.snr_tol,

        dr_enable=dr_enable,
        dr_use_tol=bool(args.dr_use_tol) if dr_enable else False,
        dr_center=args.dr_center,
        dr_tol=args.dr_tol,
        dr_use_range=bool(args.dr_use_range) if dr_enable else False,
        dr_min=args.dr_min,
        dr_max=args.dr_max,

        group_enable=group_enable,
        beam_enable=beam_enable,
        group_n=len(ALLOW_GROUP_IDS),
        beam_n=len(ALLOW_BEAM_IDS),
    )

    # Sanity checks for tol requirements
    if cfg["freq_enable"] and cfg["freq_use_tol"]:
        if cfg["freq_center"] is None or cfg["freq_tol"] is None:
            raise ValueError("Frequency tolerance filter requires --freq_center and --freq_tol.")

    if cfg["snr_enable"] and cfg["snr_use_tol"]:
        if cfg["snr_center"] is None or cfg["snr_tol"] is None:
            raise ValueError("SNR tolerance filter requires --snr_center and --snr_tol.")

    if cfg["dr_enable"] and cfg["dr_use_tol"]:
        if cfg["dr_center"] is None or cfg["dr_tol"] is None:
            raise ValueError("DR tolerance filter requires --dr_center (and --dr_tol if not default).")

    # NEW: allow-list sanity
    if cfg["group_enable"] and len(ALLOW_GROUP_IDS) == 0:
        raise ValueError("Group filter enabled but ALLOW_GROUP_IDS is empty. Fill it in Configs.")
    if cfg["beam_enable"] and len(ALLOW_BEAM_IDS) == 0:
        raise ValueError("Beam filter enabled but ALLOW_BEAM_IDS is empty. Fill it in Configs.")

    # ---- Load CSV ----
    print("\n==============================")
    print("[\033[32mInfo\033[0m] FILTER (SNR / DR / FREQ / GROUP / BEAM)")
    print("==============================")
    print(f"[\033[32mInfo\033[0m] Input CSV: {in_csv}")

    df = pd.read_csv(in_csv)

    # Minimal required columns (always)
    required_cols = [COL_GROUP_ID, COL_BEAM_ID, COL_FREQ_MHZ]
    if snr_enable:
        required_cols.append(COL_SNR)
    if dr_enable:
        required_cols.append(COL_DR_HZ_S)

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}. Columns present: {list(df.columns)}")

    # Normalize types
    df[COL_BEAM_ID] = ensure_numeric(df[COL_BEAM_ID])
    df[COL_FREQ_MHZ] = ensure_numeric(df[COL_FREQ_MHZ])
    if snr_enable:
        df[COL_SNR] = ensure_numeric(df[COL_SNR])
    if dr_enable:
        df[COL_DR_HZ_S] = ensure_numeric(df[COL_DR_HZ_S])

    # Drop rows without core structural fields
    before_drop = len(df)
    df = df.dropna(subset=[COL_GROUP_ID, COL_BEAM_ID, COL_FREQ_MHZ]).copy()
    after_drop = len(df)
    if after_drop != before_drop:
        print(f"[\033[32mInfo\033[0m] Dropped {before_drop - after_drop} rows due to NaN in "
              f"{COL_GROUP_ID}/{COL_BEAM_ID}/{COL_FREQ_MHZ}.")

    # ---- Prepare output dir ----
    cfg_tag = build_cfg_tag(cfg)
    out_root = Path(OUTPUT_ROOT).expanduser().resolve()
    out_dir = make_output_dir(out_root, cfg_tag)
    print(f"[\033[32mInfo\033[0m] Output directory: {out_dir}")
    print(f"[\033[32mInfo\033[0m] Config tag: {cfg_tag}")

    # ---- Build mask ----
    keep = np.ones(len(df), dtype=bool)

    # NEW: Group allow-list mask (compare as string for robustness)
    if cfg["group_enable"]:
        gid_str = df[COL_GROUP_ID].astype(str)
        allow_gid = set(str(x) for x in ALLOW_GROUP_IDS)
        keep &= gid_str.isin(allow_gid).to_numpy(dtype=bool)

    # NEW: Beam allow-list mask (compare as int)
    if cfg["beam_enable"]:
        bid = df[COL_BEAM_ID].to_numpy(dtype=float)
        # Convert allow list to float-safe comparison via ints
        allow_bid = set(int(x) for x in ALLOW_BEAM_IDS)
        # bid could be float; cast to int after finite check
        bid_ok = np.isfinite(bid)
        bid_int = np.zeros_like(bid, dtype=int)
        bid_int[bid_ok] = bid[bid_ok].astype(int)
        keep &= (bid_ok & np.isin(bid_int, list(allow_bid)))

    # Frequency mask
    if cfg["freq_enable"]:
        f = df[COL_FREQ_MHZ].to_numpy(dtype=float)
        if cfg["freq_use_range"]:
            keep &= apply_range_mask(f, cfg["freq_min"], cfg["freq_max"])
        if cfg["freq_use_tol"]:
            keep &= apply_tol_mask(f, cfg["freq_center"], cfg["freq_tol"])

    # SNR mask
    if cfg["snr_enable"]:
        s = df[COL_SNR].to_numpy(dtype=float)
        if cfg["snr_use_range"]:
            keep &= apply_range_mask(s, cfg["snr_min"], cfg["snr_max"])
        if cfg["snr_use_tol"]:
            keep &= apply_tol_mask(s, cfg["snr_center"], cfg["snr_tol"])

    # Drift-rate mask
    if cfg["dr_enable"]:
        d = df[COL_DR_HZ_S].to_numpy(dtype=float)
        if cfg["dr_use_range"]:
            keep &= apply_range_mask(d, cfg["dr_min"], cfg["dr_max"])
        if cfg["dr_use_tol"]:
            keep &= apply_tol_mask(d, cfg["dr_center"], cfg["dr_tol"])

    # ---- Output ----
    kept_df = df[keep].copy()
    kept_path = out_dir / "filter_out.csv"
    kept_df.to_csv(kept_path, index=False)

    print("\n==============================")
    print("[\033[32mInfo\033[0m] RESULTS")
    print("==============================")
    print(f"[\033[32mInfo\033[0m] Rows total: {len(df)}")
    print(f"[\033[32mInfo\033[0m] Kept (filter_out.csv): {len(kept_df)} -> {kept_path}")
    print("\n[\033[32mInfo\033[0m] Done!")


if __name__ == "__main__":
    main()
