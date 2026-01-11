# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FAST SETI Cross-Patch Stitching Case Finder (BAND-ONLY + Full Audit + Stats)

Outputs:
- A stats/ folder with CSVs:
    * file_stats.csv     : per csv_id summary (counts, cases, exclusions)
    * group_stats.csv    : per group_id summary
    * case_stats.csv     : per-case stitched freq_start/end (+ patches span, members)
    * link_log.csv       : every attempted cross-patch link (success/fail + why)
    * exclusion_log.csv  : all objects excluded due to IoU suppression / crowded-skip / mismatch
    * case_members.csv   : all objects that actually participate in stitching (with case_id)
- Case-level output:
    * case_summary.csv   : one row per case, PIPELINE_COLUMNS overwritten by stitched freq_start/freq_end,
                           plus ONLY group_id/beam_id/csv_id.

Stitching logic:
1) Map-back:
   - Use (time_start,time_end,freq_start,freq_end) as global coordinates (time-freq boxes).
2) Boundary duplicate merge:
   - For each neighboring patch boundary (same cell_row, col -> col+1), consider detections in the overlap region.
   - Perform IoU-based suppression across the two neighboring patches (no confidence-based keep).
3) Boundary-truncated stitching:
   - Only stitch boundary-truncated segments (touching boundary within TOL_MHZ).
   - Only stitch when drift direction/morphology are consistent (drift sign + class_id).
   - Only stitch when overlap region is sparsely populated; otherwise skip stitching completely (no confidence keep).

Assumptions:
- CSVs always contain the required columns with correct dtypes.
- No NaN exists in required fields.

Note：
- This script runs slowly and is computationally intensive.
"""

import argparse
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any

import pandas as pd

# ====== Configs of Pipeline ======
PIPELINE_COLUMNS = [
    "DriftRate", "SNR", "Uncorrected_Frequency", "freq_start", "freq_end", "class_id", "confidence",
    "cell_row", "cell_col", "gSNR", "freq_min", "freq_max", "time_start", "time_end", "mode"
]
DEFAULT_INPUT_FOLDER = './filter_workflow/init'
FREQ_MIN_MHZ = 1050.0
FREQ_MAX_MHZ = 1450.0

# ====== IoU-based suppression ======
IOU_SUPPRESS_THRESH = 0.5

# ====== Stitching Controls ======
REQUIRE_SAME_DRIFT_SIGN = True
OVERLAP_FRAC = 0.2
# Frequency resolution (Hz) -> tolerance (MHz)
FREQ_RES_HZ = 7.5
TOL_MHZ = (1.0 * FREQ_RES_HZ) * 1e-6  # MHz
# Overlap length requirement between two signal segments (MHz). 0.0 means "any intersection ok"
MIN_SEGMENT_OVERLAP_MHZ = 0.0
# If True, require detection intersects overlap-zone to be considered
STRICT_ZONE_INTERSECTION = True


# ====== Helper structures ======
@dataclass(frozen=True)
class PatchKey:
    row: int
    col: int


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


def ensure_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def build_frequency_mask(df: pd.DataFrame) -> pd.Series:
    """
    Band filter:
    keep if [min(freq_start,freq_end), max(freq_start,freq_end)] overlaps [FREQ_MIN_MHZ, FREQ_MAX_MHZ]
    """
    fs = ensure_numeric(df["freq_start"])
    fe = ensure_numeric(df["freq_end"])
    f_lo = pd.concat([fs, fe], axis=1).min(axis=1)
    f_hi = pd.concat([fs, fe], axis=1).max(axis=1)
    return (f_hi >= FREQ_MIN_MHZ) & (f_lo <= FREQ_MAX_MHZ)


def interval_from_row(row: pd.Series) -> Tuple[float, float]:
    """
    Return (lo, hi) for a signal interval in MHz using freq_start/freq_end.
    """
    fs = float(row["freq_start"])
    fe = float(row["freq_end"])
    return float(min(fs, fe)), float(max(fs, fe))


def time_interval_from_row(row: pd.Series) -> Tuple[float, float]:
    """
    Return (t0, t1) using time_start/time_end (global after map-back).
    """
    t0 = float(row["time_start"])
    t1 = float(row["time_end"])
    return float(min(t0, t1)), float(max(t0, t1))


def interval_overlaps(a_lo: float, a_hi: float, b_lo: float, b_hi: float) -> float:
    return max(0.0, min(a_hi, b_hi) - max(a_lo, b_lo))


def interval_len(a_lo: float, a_hi: float) -> float:
    return max(0.0, a_hi - a_lo)


def freq_iou_1d(a_lo: float, a_hi: float, b_lo: float, b_hi: float) -> float:
    inter = interval_overlaps(a_lo, a_hi, b_lo, b_hi)
    union = interval_len(a_lo, a_hi) + interval_len(b_lo, b_hi) - inter
    return inter / (union + 1e-12)


def in_zone(a_lo: float, a_hi: float, z_lo: float, z_hi: float) -> bool:
    """
    Whether interval intersects zone. Expand zone by tolerance to be robust.
    """
    if not STRICT_ZONE_INTERSECTION:
        return True
    z_lo2 = z_lo - TOL_MHZ
    z_hi2 = z_hi + TOL_MHZ
    return interval_overlaps(a_lo, a_hi, z_lo2, z_hi2) > 0.0 or (a_lo == a_hi and z_lo2 <= a_lo <= z_hi2)


def drift_sign(row: pd.Series) -> int:
    d = float(row["DriftRate"])
    if d > 0:
        return 1
    if d < 0:
        return -1
    return 0


def patch_bounds(patch_df: pd.DataFrame) -> Tuple[float, float]:
    fmin = float(patch_df["freq_min"].iloc[0])
    fmax = float(patch_df["freq_max"].iloc[0])
    return fmin, fmax


def touches_left_boundary(row: pd.Series, fmin: float) -> bool:
    lo, hi = interval_from_row(row)
    return abs(lo - fmin) < TOL_MHZ


def touches_right_boundary(row: pd.Series, fmax: float) -> bool:
    lo, hi = interval_from_row(row)
    return abs(hi - fmax) < TOL_MHZ


def make_output_dir(base: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = (
        f"{ts}_f{int(FREQ_MIN_MHZ)}-{int(FREQ_MAX_MHZ)}"
        f"_bandonly_overlap{OVERLAP_FRAC}"
        f"_tol{TOL_MHZ:.8g}"
        f"_IoU{IOU_SUPPRESS_THRESH}"
        f"_skipCrowded"
    )
    out_dir = base / "filter_workflow" / "stitching" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stats").mkdir(parents=True, exist_ok=True)
    return out_dir


def weighted_mean(vals: List[float], weights: List[float]) -> float:
    sw = sum(weights)
    if sw <= 0:
        return float(vals[0]) if vals else 0.0
    return sum(w * v for v, w in zip(vals, weights)) / (sw + 1e-12)


def weighted_rms(vals: List[float], weights: List[float]) -> float:
    sw = sum(weights)
    if sw <= 0:
        return abs(float(vals[0])) if vals else 0.0
    return (sum(w * (v ** 2) for v, w in zip(vals, weights)) / (sw + 1e-12)) ** 0.5


# ====== Union-Find for cases ======
class UnionFind:
    def __init__(self):
        self.parent: Dict[str, str] = {}
        self.rank: Dict[str, int] = {}

    def add(self, x: str):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x: str) -> str:
        p = self.parent.get(x, x)
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent.get(x, x)

    def union(self, a: str, b: str):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def main():
    ap = argparse.ArgumentParser(description="Cross-patch stitching case finder (band-only + stats).")
    ap.add_argument("--input_dir", type=str, default=DEFAULT_INPUT_FOLDER,
                    help="Folder containing CSV files.")
    args = ap.parse_args()

    in_dir = Path(args.input_dir).expanduser().resolve()
    csv_files = sorted([p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() == ".csv"])
    if not csv_files:
        print(f"[\033[32mInfo\033[0m] No CSV files found in: {in_dir}")
        return

    out_dir = make_output_dir(Path.cwd())
    stats_dir = out_dir / "stats"
    print(f"[\033[32mInfo\033[0m] Output directory: {out_dir}")

    file_stats_rows: List[Dict[str, Any]] = []
    exclusion_log_rows: List[Dict[str, Any]] = []
    link_log_rows: List[Dict[str, Any]] = []
    all_case_members_rows: List[pd.DataFrame] = []
    all_case_summary_rows: List[Dict[str, Any]] = []
    all_case_stats_rows: List[Dict[str, Any]] = []

    for idx, p in enumerate(csv_files, start=1):
        csv_id = p.name
        group_id, beam_id = parse_group_and_beam(p.stem)

        df_raw = pd.read_csv(p)
        n_total = len(df_raw)

        band_mask = build_frequency_mask(df_raw)
        df_band = df_raw[band_mask].copy()
        n_in_band = len(df_band)

        df_band = df_band.reset_index(drop=False).rename(columns={"index": "_orig_index"})
        df_band["_uid"] = df_band.apply(lambda r: f"{csv_id}::{int(r['_orig_index'])}", axis=1)

        patch_lookup: Dict[PatchKey, pd.DataFrame] = {}
        for (r, c), g in df_band.groupby(["cell_row", "cell_col"], as_index=False):
            patch_lookup[PatchKey(int(r), int(c))] = g.copy()

        n_patches = len(patch_lookup)

        zone_cache: Dict[Tuple[PatchKey, str], Tuple[float, float]] = {}
        for pk, g in patch_lookup.items():
            fmin, fmax = patch_bounds(g)
            w = (fmax - fmin) * OVERLAP_FRAC
            zone_cache[(pk, "L")] = (fmin, fmin + w)
            zone_cache[(pk, "R")] = (fmax - w, fmax)

        uid_to_row: Dict[str, pd.Series] = {}
        for _, row in df_band.iterrows():
            uid_to_row[str(row["_uid"])] = row

        active_uids = set(uid_to_row.keys())

        # ====== stats per patch (population in overlap regions) ======
        for pk, g in patch_lookup.items():
            fmin, fmax = patch_bounds(g)
            w = (fmax - fmin) * OVERLAP_FRAC

            zL = zone_cache[(pk, "L")]
            zR = zone_cache[(pk, "R")]

            n_left = 0
            n_right = 0
            for _, row in g.iterrows():
                uid = str(row["_uid"])
                if uid not in active_uids:
                    continue
                lo, hi = interval_from_row(row)
                if in_zone(lo, hi, zL[0], zL[1]):
                    n_left += 1
                if in_zone(lo, hi, zR[0], zR[1]):
                    n_right += 1

        # ====== step 1+2: map-back already implicit; do boundary IoU-based suppression across neighbors ======
        suppressed_count = 0
        excluded_class_count = 0

        patch_keys_sorted = sorted(patch_lookup.keys(), key=lambda x: (x.row, x.col))
        patch_set = set(patch_keys_sorted)

        for pk in patch_keys_sorted:
            nb = PatchKey(pk.row, pk.col + 1)
            if nb not in patch_set:
                continue

            gA = patch_lookup[pk]
            gB = patch_lookup[nb]

            zA = zone_cache[(pk, "R")]  # A right overlap
            zB = zone_cache[(nb, "L")]  # B left overlap

            A_uids = []
            for _, row in gA.iterrows():
                uid = str(row["_uid"])
                if uid not in active_uids:
                    continue
                lo, hi = interval_from_row(row)
                if in_zone(lo, hi, zA[0], zA[1]):
                    A_uids.append(uid)

            B_uids = []
            for _, row in gB.iterrows():
                uid = str(row["_uid"])
                if uid not in active_uids:
                    continue
                lo, hi = interval_from_row(row)
                if in_zone(lo, hi, zB[0], zB[1]):
                    B_uids.append(uid)

            if not A_uids or not B_uids:
                continue

            # Greedy IoU suppression across boundary: if IoU>=thresh (and consistent), drop one deterministically.
            # Deterministic keep rule: keep larger-area box; if tie, keep A-side.
            for a_uid in A_uids:
                if a_uid not in active_uids:
                    continue

                a_row = uid_to_row[a_uid]
                a_cls = int(a_row["class_id"])
                a_sgn = drift_sign(a_row)

                a_lo, a_hi = interval_from_row(a_row)

                best_b = None
                best_iou = 0.0

                for b_uid in B_uids:
                    if b_uid not in active_uids:
                        continue

                    b_row = uid_to_row[b_uid]
                    b_cls = int(b_row["class_id"])
                    if b_cls != a_cls:
                        continue

                    if REQUIRE_SAME_DRIFT_SIGN:
                        b_sgn = drift_sign(b_row)
                        if b_sgn != a_sgn:
                            continue

                    b_lo, b_hi = interval_from_row(b_row)
                    iou = freq_iou_1d(a_lo, a_hi, b_lo, b_hi)

                    if iou > best_iou:
                        best_iou = iou
                        best_b = b_uid

                if best_b is None:
                    continue

                if best_iou >= IOU_SUPPRESS_THRESH:
                    b_row = uid_to_row[best_b]

                    # NMS-like suppression
                    a_conf = float(a_row["confidence"])
                    b_conf = float(b_row["confidence"])

                    if b_conf > a_conf:
                        kept_uid = best_b
                        drop_uid = a_uid
                    else:
                        kept_uid = a_uid
                        drop_uid = best_b

                    if drop_uid in active_uids:
                        active_uids.remove(drop_uid)
                        suppressed_count += 1
                        exclusion_log_rows.append({
                            "group_id": group_id, "beam_id": beam_id if beam_id is not None else "",
                            "csv_id": csv_id,
                            "context": "boundary_iou_suppression",
                            "a_row": pk.row, "a_col": pk.col,
                            "b_row": nb.row, "b_col": nb.col,
                            "class_id": int(uid_to_row[kept_uid]["class_id"]),
                            "kept_uid": kept_uid,
                            "dropped_uid": drop_uid,
                            "iou": best_iou,
                            "reason": "iou_duplicate_suppressed",
                        })

        # ====== step 3: boundary-truncated stitching with crowded-skip ======
        uf = UnionFind()
        for uid in list(active_uids):
            uf.add(uid)

        links_attempted = 0
        links_success = 0

        for pk in patch_keys_sorted:
            nb = PatchKey(pk.row, pk.col + 1)
            if nb not in patch_set:
                continue

            gA = patch_lookup[pk]
            gB = patch_lookup[nb]

            fminA, fmaxA = patch_bounds(gA)
            fminB, fmaxB = patch_bounds(gB)

            zA = zone_cache[(pk, "R")]
            zB = zone_cache[(nb, "L")]

            popA = []
            for _, row in gA.iterrows():
                uid = str(row["_uid"])
                if uid not in active_uids:
                    continue
                lo, hi = interval_from_row(row)
                if in_zone(lo, hi, zA[0], zA[1]):
                    popA.append(uid)

            popB = []
            for _, row in gB.iterrows():
                uid = str(row["_uid"])
                if uid not in active_uids:
                    continue
                lo, hi = interval_from_row(row)
                if in_zone(lo, hi, zB[0], zB[1]):
                    popB.append(uid)

            # overlap region sparsely populated; otherwise skip stitching
            if len(popA) > 1 or len(popB) > 1:
                # Log skip once per boundary
                link_log_rows.append({
                    "group_id": group_id, "beam_id": beam_id if beam_id is not None else "",
                    "csv_id": csv_id,
                    "a_row": pk.row, "a_col": pk.col, "a_side": "R",
                    "b_row": nb.row, "b_col": nb.col, "b_side": "L",
                    "status": "skip",
                    "reason": f"overlap_region_crowded(A={len(popA)},B={len(popB)})",
                })
                # Record all objects involved for audit
                for uid in popA:
                    exclusion_log_rows.append({
                        "group_id": group_id, "beam_id": beam_id if beam_id is not None else "",
                        "csv_id": csv_id,
                        "context": "stitch_skip_crowded",
                        "patch_row": pk.row, "patch_col": pk.col,
                        "side": "R",
                        "uid": uid,
                        "reason": "overlap_region_crowded_skip_stitching",
                    })
                for uid in popB:
                    exclusion_log_rows.append({
                        "group_id": group_id, "beam_id": beam_id if beam_id is not None else "",
                        "csv_id": csv_id,
                        "context": "stitch_skip_crowded",
                        "patch_row": nb.row, "patch_col": nb.col,
                        "side": "L",
                        "uid": uid,
                        "reason": "overlap_region_crowded_skip_stitching",
                    })
                continue

            # Build boundary-truncated heads (after suppression)
            headsA = []
            for uid in popA:
                row = uid_to_row[uid]
                if touches_right_boundary(row, fmaxA):
                    headsA.append(uid)

            headsB = []
            for uid in popB:
                row = uid_to_row[uid]
                if touches_left_boundary(row, fminB):
                    headsB.append(uid)

            if not headsA or not headsB:
                continue

            # Attempt link between (usually unique) heads
            for a_uid in headsA:
                a_row = uid_to_row[a_uid]
                a_cls = int(a_row["class_id"])
                a_sgn = drift_sign(a_row)

                for b_uid in headsB:
                    b_row = uid_to_row[b_uid]
                    b_cls = int(b_row["class_id"])

                    links_attempted += 1

                    # morphology consistency: class_id
                    if b_cls != a_cls:
                        excluded_class_count += 1
                        link_log_rows.append({
                            "group_id": group_id, "beam_id": beam_id if beam_id is not None else "",
                            "csv_id": csv_id,
                            "a_row": pk.row, "a_col": pk.col, "a_side": "R", "class_id": a_cls,
                            "a_uid": a_uid,
                            "b_row": nb.row, "b_col": nb.col, "b_side": "L",
                            "b_uid": b_uid,
                            "status": "fail",
                            "reason": "class_id_mismatch",
                        })
                        continue

                    # drift direction consistency
                    if REQUIRE_SAME_DRIFT_SIGN:
                        b_sgn = drift_sign(b_row)
                        if b_sgn != a_sgn:
                            exclusion_log_rows.append({
                                "group_id": group_id, "beam_id": beam_id if beam_id is not None else "",
                                "csv_id": csv_id,
                                "context": "cross_patch_match",
                                "patch_row": pk.row, "patch_col": pk.col,
                                "neighbor_row": nb.row, "neighbor_col": nb.col,
                                "side": "R->L",
                                "target_class_id": a_cls,
                                "excluded_uid": b_uid,
                                "reason": "drift_sign_mismatch",
                            })
                            link_log_rows.append({
                                "group_id": group_id, "beam_id": beam_id if beam_id is not None else "",
                                "csv_id": csv_id,
                                "a_row": pk.row, "a_col": pk.col, "a_side": "R", "class_id": a_cls,
                                "a_uid": a_uid,
                                "b_row": nb.row, "b_col": nb.col, "b_side": "L",
                                "b_uid": b_uid,
                                "status": "fail",
                                "reason": "drift_sign_mismatch",
                            })
                            continue

                    # segment overlap requirement (freq)
                    a_lo, a_hi = interval_from_row(a_row)
                    b_lo, b_hi = interval_from_row(b_row)
                    ov = interval_overlaps(a_lo, a_hi, b_lo, b_hi)
                    if ov < MIN_SEGMENT_OVERLAP_MHZ:
                        link_log_rows.append({
                            "group_id": group_id, "beam_id": beam_id if beam_id is not None else "",
                            "csv_id": csv_id,
                            "a_row": pk.row, "a_col": pk.col, "a_side": "R", "class_id": a_cls,
                            "a_uid": a_uid,
                            "b_row": nb.row, "b_col": nb.col, "b_side": "L",
                            "b_uid": b_uid,
                            "status": "fail",
                            "reason": f"segment_overlap_too_small({ov:.6g}MHz)",
                        })
                        continue

                    uf.union(a_uid, b_uid)
                    links_success += 1
                    link_log_rows.append({
                        "group_id": group_id, "beam_id": beam_id if beam_id is not None else "",
                        "csv_id": csv_id,
                        "a_row": pk.row, "a_col": pk.col, "a_side": "R", "class_id": a_cls,
                        "a_uid": a_uid,
                        "b_row": nb.row, "b_col": nb.col, "b_side": "L",
                        "b_uid": b_uid,
                        "status": "success",
                        "reason": "",
                        "segment_overlap_mhz": ov,
                    })

        # ====== build cases from stitching unions ======
        comp: Dict[str, List[str]] = {}
        for uid in active_uids:
            root = uf.find(uid)
            comp.setdefault(root, []).append(uid)

        case_components = [members for members in comp.values() if len(members) >= 2]

        case_id = 0
        n_case_members = 0

        for members in sorted(case_components, key=lambda x: len(x), reverse=True):
            class_ids = set()
            intervals = []
            times = []

            # --- aggregation buffers ---
            weights: List[float] = []
            drift_vals: List[float] = []
            snr_vals: List[float] = []
            gsnr_vals: List[float] = []
            conf_vals: List[float] = []
            freq_min_vals: List[float] = []
            freq_max_vals: List[float] = []
            cell_row_vals: List[int] = []
            cell_col_vals: List[int] = []

            for uid in members:
                r = uid_to_row[uid]
                class_ids.add(int(r["class_id"]))

                lo, hi = interval_from_row(r)
                intervals.append((lo, hi))

                t0, t1 = time_interval_from_row(r)
                times.append((t0, t1))

                # weights: use frequency span; clamp by resolution tolerance to avoid zero-weight segments
                w = max(interval_len(lo, hi), TOL_MHZ)
                weights.append(float(w))

                drift_vals.append(float(r["DriftRate"]))
                snr_vals.append(float(r["SNR"]))
                gsnr_vals.append(float(r["gSNR"]))
                conf_vals.append(float(r["confidence"]))

                freq_min_vals.append(float(r["freq_min"]))
                freq_max_vals.append(float(r["freq_max"]))

                cell_row_vals.append(int(r["cell_row"]))
                cell_col_vals.append(int(r["cell_col"]))

            if len(class_ids) != 1:
                continue

            case_id += 1

            stitched_f0 = min([lo for lo, _ in intervals])
            stitched_f1 = max([hi for _, hi in intervals])
            stitched_t0 = min([t0 for t0, _ in times])
            stitched_t1 = max([t1 for _, t1 in times])

            # Representative: deterministic (no confidence keep)
            # Use largest freq-span; if tie, smallest uid lexicographically.
            best_uid = None
            best_w = -1.0
            for uid in members:
                lo, hi = interval_from_row(uid_to_row[uid])
                w = interval_len(lo, hi)
                if w > best_w:
                    best_w = w
                    best_uid = uid
                elif w == best_w and (best_uid is None or uid < best_uid):
                    best_uid = uid

            rep_uid = best_uid if best_uid is not None else members[0]
            rep_row = uid_to_row[rep_uid]

            # ---- case-level metric aggregation (approximation; no raw recomputation) ----
            drift_agg = weighted_mean(drift_vals, weights)
            snr_agg = weighted_rms(snr_vals, weights)      # square-based aggregation
            gsnr_agg = weighted_mean(gsnr_vals, weights)   # representative strength
            conf_agg = max(conf_vals) if conf_vals else float(rep_row["confidence"])

            freq_min_agg = min(freq_min_vals) if freq_min_vals else float(rep_row["freq_min"])
            freq_max_agg = max(freq_max_vals) if freq_max_vals else float(rep_row["freq_max"])

            cell_row_agg = min(cell_row_vals) if cell_row_vals else int(rep_row["cell_row"])
            cell_col_agg = min(cell_col_vals) if cell_col_vals else int(rep_row["cell_col"])

            # Uncorrected_Frequency is the start-time frequency: choose stitched edge by drift sign
            if drift_agg > 0:
                f0_agg = stitched_f0
            elif drift_agg < 0:
                f0_agg = stitched_f1
            else:
                f0_agg = stitched_f0

            mem_df = pd.DataFrame([uid_to_row[u] for u in members]).copy()
            mem_df.insert(0, "case_id", case_id)
            mem_df.insert(0, "csv_id", csv_id)
            mem_df.insert(0, "beam_id", beam_id if beam_id is not None else "")
            mem_df.insert(0, "group_id", group_id)
            mem_df["stitched_freq_start"] = stitched_f0
            mem_df["stitched_freq_end"] = stitched_f1
            mem_df["stitched_time_start"] = stitched_t0
            mem_df["stitched_time_end"] = stitched_t1
            all_case_members_rows.append(mem_df)
            n_case_members += len(mem_df)

            summary = {col: rep_row[col] for col in PIPELINE_COLUMNS}
            # overwrite stitched extents
            summary["freq_start"] = stitched_f0
            summary["freq_end"] = stitched_f1
            summary["time_start"] = stitched_t0
            summary["time_end"] = stitched_t1
            summary["freq_min"] = freq_min_agg
            summary["freq_max"] = freq_max_agg

            # overwrite aggregated metrics
            summary["DriftRate"] = drift_agg
            summary["SNR"] = snr_agg
            summary["gSNR"] = gsnr_agg
            summary["confidence"] = conf_agg
            summary["Uncorrected_Frequency"] = f0_agg

            # overwrite locator (optional but deterministic)
            summary["cell_row"] = cell_row_agg
            summary["cell_col"] = cell_col_agg

            summary["group_id"] = group_id
            summary["beam_id"] = beam_id if beam_id is not None else ""
            summary["csv_id"] = csv_id
            all_case_summary_rows.append(summary)

            all_case_stats_rows.append({
                "group_id": group_id,
                "beam_id": beam_id if beam_id is not None else "",
                "csv_id": csv_id,
                "case_id": case_id,
                "class_id": int(rep_row["class_id"]),
                "stitched_freq_start": stitched_f0,
                "stitched_freq_end": stitched_f1,
                "stitched_time_start": stitched_t0,
                "stitched_time_end": stitched_t1,
                "n_members": len(members),
                "rep_uid": rep_uid,
            })

        # ====== file stats ======
        file_stats_rows.append({
            "group_id": group_id,
            "beam_id": beam_id if beam_id is not None else "",
            "csv_id": csv_id,
            "rows_total": n_total,
            "rows_in_band": n_in_band,
            "n_patches": n_patches,
            "n_cases": case_id,
            "n_case_members": n_case_members,
            "excluded_by_confidence": suppressed_count,  # legacy field name (now used for IoU suppression count)
            "excluded_by_class": excluded_class_count,
            "links_attempted": links_attempted,
            "links_success": links_success,
            "overlap_frac": OVERLAP_FRAC,
            "tol_mhz": TOL_MHZ,
        })

        if (idx % 19) == 0:
            print(f"[\033[32mInfo\033[0m] Processed {idx}/{len(csv_files)} files...")

    # ====== write outputs ======
    if all_case_members_rows:
        case_members_df = pd.concat(all_case_members_rows, ignore_index=True)
        ordered = ["group_id", "beam_id", "csv_id", "case_id"] + PIPELINE_COLUMNS + [
            "stitched_freq_start", "stitched_freq_end", "stitched_time_start", "stitched_time_end", "_uid",
            "_orig_index"
        ]
        case_members_df = case_members_df[ordered]
        case_members_df.to_csv(out_dir / "case_members.csv", index=False)

    if all_case_summary_rows:
        case_summary_df = pd.DataFrame(all_case_summary_rows)
        ordered = PIPELINE_COLUMNS + ["group_id", "beam_id", "csv_id"]
        case_summary_df = case_summary_df[ordered]
        case_summary_df.to_csv(out_dir / "case_summary.csv", index=False)

    pd.DataFrame(file_stats_rows).to_csv(stats_dir / "file_stats.csv", index=False)
    pd.DataFrame(all_case_stats_rows).to_csv(stats_dir / "case_stats.csv", index=False)
    pd.DataFrame(link_log_rows).to_csv(stats_dir / "link_log.csv", index=False)
    pd.DataFrame(exclusion_log_rows).to_csv(stats_dir / "exclusion_log.csv", index=False)

    fs = pd.DataFrame(file_stats_rows)
    grp = fs.groupby("group_id", as_index=False).agg({
        "rows_total": "sum",
        "rows_in_band": "sum",
        "n_patches": "sum",
        "n_cases": "sum",
        "n_case_members": "sum",
        "excluded_by_confidence": "sum",
        "excluded_by_class": "sum",
        "links_attempted": "sum",
        "links_success": "sum",
    }).sort_values("group_id")
    grp.to_csv(stats_dir / "group_stats.csv", index=False)

    print(f"[\033[32mInfo\033[0m] Outputs written to: {out_dir}")
    print("[\033[32mInfo\033[0m] Done!")


if __name__ == "__main__":
    main()
