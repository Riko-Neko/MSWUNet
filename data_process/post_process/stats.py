#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CSV group overview + threshold counting.

Assumptions:
1) All CSV files are directly inside the target folder (no recursion).
2) Each CSV filename (stem) is the unique identifier.
3) Group key = part before the first occurrence of "_M" in the filename stem.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

# ====== Default parameters (edit here or override via CLI) ======

PIPELINE_COLUMNS = ["DriftRate", "SNR", "Uncorrected_Frequency", "freq_start", "freq_end", "class_id", "confidence",
                    "cell_row", "cell_col", "gSNR"    "freq_min", "freq_max", "time_start", "time_end", "mode"]
DEFAULT_FOLDER = './filter_workflow/init'
DEFAULT_THRESHOLDS = {"SNR": 10.0, "gSNR": 2000, "confidence": 0.9}


def list_csv_files(folder: Path) -> List[Path]:
    return sorted([p for p in folder.glob("*.csv") if p.is_file()])


def parse_group_key(stem: str) -> str:
    """
    Group key is the part before '_M' (first occurrence).
    If '_M' not found, group key is the whole stem.
    """
    idx = stem.find("_M")
    return stem[:idx] if idx != -1 else stem


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def count_thresholds_in_df(df: pd.DataFrame, thresholds: Dict[str, float]) -> Dict[str, int]:
    """
    Returns:
      - count_SNR: rows where SNR >= threshold
      - count_gSNR: rows where gSNR >= threshold
      - count_confidence: rows where confidence >= threshold
      - count_all3: rows where all three conditions hold simultaneously
      - n_rows: total rows
    """
    # Column presence check
    missing = [k for k in thresholds.keys() if k not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}. Present columns: {list(df.columns)}")

    snr = safe_numeric(df["SNR"])
    gsnr = safe_numeric(df["gSNR"])
    conf = safe_numeric(df["confidence"])

    c_snr = int((snr >= thresholds["SNR"]).sum())
    c_gsnr = int((gsnr >= thresholds["gSNR"]).sum())
    c_conf = int((conf >= thresholds["confidence"]).sum())
    c_all3 = int(((snr >= thresholds["SNR"]) & (gsnr >= thresholds["gSNR"]) & (conf >= thresholds["confidence"])).sum())

    return {"n_rows": int(len(df)), "count_SNR": c_snr, "count_gSNR": c_gsnr, "count_confidence": c_conf,
            "count_all3": c_all3, }


def main():
    ap = argparse.ArgumentParser(description="Group CSVs by prefix before _M and count threshold-passing rows.")
    ap.add_argument("--folder", type=str, default=DEFAULT_FOLDER,
                    help="Target folder containing CSV files (non-recursive).")
    ap.add_argument("--snr", type=float, default=DEFAULT_THRESHOLDS["SNR"], help="Threshold for SNR (default: 10.0)")
    ap.add_argument("--gsnr", type=float, default=DEFAULT_THRESHOLDS["gSNR"], help="Threshold for gSNR (default: 5.0)")
    ap.add_argument("--conf", type=float, default=DEFAULT_THRESHOLDS["confidence"],
                    help="Threshold for confidence (default: 0.5)")
    args = ap.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    if not folder.exists() or not folder.is_dir():
        raise SystemExit(f"[\033[31mError\033[0m] Folder not found or not a directory: {folder}")

    thresholds = {"SNR": args.snr, "gSNR": args.gsnr, "confidence": args.conf}

    csv_files = list_csv_files(folder)
    if not csv_files:
        print(f"[\033[32mInfo\033[0m] No CSV files found in: {folder}")
        return

    # Build groups: {group_key: [(identifier, path), ...]}
    groups: Dict[str, List[Tuple[str, Path]]] = {}
    for p in csv_files:
        ident = p.stem
        g = parse_group_key(ident)
        groups.setdefault(g, []).append((ident, p))

    print(f"[\033[32mInfo\033[0m]")
    print("=== OVERVIEW ===")
    print(f"Folder: {folder}")
    print(f"CSV files: {len(csv_files)}")
    print(f"Thresholds: SNR>={thresholds['SNR']}, gSNR>={thresholds['gSNR']}, confidence>={thresholds['confidence']}")
    print()

    for g in sorted(groups.keys()):
        idents = [ident for ident, _ in groups[g]]
        print(f"[GROUP] {g}  |  n_files={len(idents)}")
        print("  - " + ", ".join(idents))
    print()

    print("=== PER GROUP ===")
    rows_out = []
    for g in sorted(groups.keys()):
        agg = {
            "group": g,
            "n_files": len(groups[g]),
            "n_rows": 0,
            "count_SNR": 0,
            "count_gSNR": 0,
            "count_confidence": 0,
            "count_all3": 0,
        }

        for ident, path in groups[g]:
            try:
                df = pd.read_csv(path)
                counts = count_thresholds_in_df(df, thresholds)
            except Exception as e:
                print(f"[WARN] Skip {ident} ({path.name}) due to error: {e}")
                continue

            agg["n_rows"] += counts["n_rows"]
            agg["count_SNR"] += counts["count_SNR"]
            agg["count_gSNR"] += counts["count_gSNR"]
            agg["count_confidence"] += counts["count_confidence"]
            agg["count_all3"] += counts["count_all3"]

        rows_out.append(agg)

    out_df = pd.DataFrame(rows_out).sort_values(["group"]).reset_index(drop=True)

    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 200):
        print(out_df.to_string(index=False))

    total = {
        "group": "__TOTAL__",
        "n_files": int(out_df["n_files"].sum()),
        "n_rows": int(out_df["n_rows"].sum()),
        "count_SNR": int(out_df["count_SNR"].sum()),
        "count_gSNR": int(out_df["count_gSNR"].sum()),
        "count_confidence": int(out_df["count_confidence"].sum()),
        "count_all3": int(out_df["count_all3"].sum()),
    }
    print("\n=== TOTAL ===")
    print(pd.DataFrame([total]).to_string(index=False))


if __name__ == "__main__":
    main()
