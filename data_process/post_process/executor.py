# Preset standard column headers for TurboSETI and ML dat files
from data_process.post_process.ML import load_ML_dat
from data_process.post_process.T_SETI import load_seti_dat

SETI_COLUMNS = [
    "Top_Hit_#", "Drift_Rate", "SNR",
    "Uncorrected_Frequency", "Corrected_Frequency",
    "Index", "freq_start", "freq_end",
    "SEFD", "SEFD_freq",
    "Coarse_Channel_Number", "Full_number_of_hits"
]

ML_COLUMNS = [
    "DriftRate", "SNR", "Uncorrected_Frequency",
    "freq_start", "freq_end",
    "cell_row", "cell_col",
    "freq_min", "freq_max",
    "time_start", "time_end"
]
import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd

# Tolerance values for SNR and Drift Rate matching
# SNR_tolerance: float or None, default None (no tolerance, exact match)
# Drift_tolerance: float or None, default None (no tolerance, exact match)
# Set to a value like 0.1 for 10% tolerance, or absolute difference
SNR_tolerance = None
Drift_tolerance = None

def analyze_dataframe(df_ml, df_seti, outdir="analysis_out", overlap_ratio_threshold=0.0):
    os.makedirs(outdir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df_ml = df_ml.rename(columns={"DriftRate": "Drift_Rate"})

    results = {}

    # ---------- 1. 精确频率区间匹配 ----------
    # Cross-merge to find potential overlaps based on frequency ranges
    df_seti['key'] = 1
    df_ml['key'] = 1
    cross = df_seti.merge(df_ml, on='key', suffixes=("_seti", "_ml"))
    del df_seti['key']
    del df_ml['key']

    # Filter for overlapping frequency intervals
    overlapping = cross[
        (cross["freq_start_seti"] <= cross["freq_end_ml"]) &
        (cross["freq_end_seti"] >= cross["freq_start_ml"])
    ]

    # Calculate overlap ratio
    overlapping['overlap_start'] = overlapping[['freq_start_seti', 'freq_start_ml']].max(axis=1)
    overlapping['overlap_end'] = overlapping[['freq_end_seti', 'freq_end_ml']].min(axis=1)
    overlapping['overlap_length'] = overlapping['overlap_end'] - overlapping['overlap_start']
    overlapping['seti_length'] = overlapping['freq_end_seti'] - overlapping['freq_start_seti']
    overlapping['ml_length'] = overlapping['freq_end_ml'] - overlapping['freq_start_ml']
    overlapping['overlap_ratio'] = overlapping['overlap_length'] / overlapping[['seti_length', 'ml_length']].min(axis=1)

    # Apply overlap ratio threshold
    common_exact = overlapping[overlapping['overlap_ratio'] >= overlap_ratio_threshold].copy()

    # Now apply tolerance matching for SNR and Drift_Rate
    if SNR_tolerance is not None:
        common_exact['snr_diff'] = abs(common_exact['SNR_seti'] - common_exact['SNR_ml'])
        snr_matched = common_exact['snr_diff'] <= SNR_tolerance
    else:
        snr_matched = common_exact['SNR_seti'] == common_exact['SNR_ml']

    if Drift_tolerance is not None:
        common_exact['drift_diff'] = abs(common_exact['Drift_Rate_seti'] - common_exact['Drift_Rate_ml'])
        drift_matched = common_exact['drift_diff'] <= Drift_tolerance
    else:
        drift_matched = common_exact['Drift_Rate_seti'] == common_exact['Drift_Rate_ml']

    # Fully matched (frequency overlap + SNR + Drift within tolerance)
    fully_matched = common_exact[snr_matched & drift_matched]

    # Define suffixed columns for SETI and ML
    seti_suffixed_cols = [col + '_seti' if col in df_ml.columns else col for col in df_seti.columns]
    ml_suffixed_cols = [col + '_ml' if col in df_seti.columns else col for col in df_ml.columns]

    # Extract and rename for SETI
    fully_matched_seti = fully_matched[seti_suffixed_cols].rename(
        columns={col + '_seti': col for col in df_seti.columns if col in df_ml.columns}
    ).drop_duplicates()

    # Extract and rename for ML
    fully_matched_ml = fully_matched[ml_suffixed_cols].rename(
        columns={col + '_ml': col for col in df_ml.columns if col in df_seti.columns}
    ).drop_duplicates()

    fully_matched_seti.to_csv(f"{outdir}/common_exact_seti_{timestamp}.csv", index=False)
    fully_matched_ml.to_csv(f"{outdir}/common_exact_ml_{timestamp}.csv", index=False)
    results["common_exact"] = len(fully_matched_seti)

    # Low confidence: frequency overlap but SNR or Drift mismatch
    low_confidence = common_exact[~(snr_matched & drift_matched)]

    # Extract and rename for SETI
    low_confidence_seti = low_confidence[seti_suffixed_cols].rename(
        columns={col + '_seti': col for col in df_seti.columns if col in df_ml.columns}
    ).drop_duplicates()

    # Extract and rename for ML
    low_confidence_ml = low_confidence[ml_suffixed_cols].rename(
        columns={col + '_ml': col for col in df_ml.columns if col in df_seti.columns}
    ).drop_duplicates()

    low_confidence_seti.to_csv(f"{outdir}/low_confidence_seti_{timestamp}.csv", index=False)
    low_confidence_ml.to_csv(f"{outdir}/low_confidence_ml_{timestamp}.csv", index=False)
    results["low_confidence"] = len(low_confidence_seti)

    # ---------- 2. 各自独立信号 ----------
    # ML 独有 (not in frequency matches meeting threshold)
    ml_only_mask = ~df_ml["Uncorrected_Frequency"].isin(common_exact["Uncorrected_Frequency_ml"])
    ml_only = df_ml[ml_only_mask]
    ml_only.to_csv(f"{outdir}/ml_only_{timestamp}.csv", index=False)
    results["ml_only"] = len(ml_only)

    # SETI 独有
    seti_only_mask = ~df_seti["Uncorrected_Frequency"].isin(common_exact["Uncorrected_Frequency_seti"])
    seti_only = df_seti[seti_only_mask]
    seti_only.to_csv(f"{outdir}/seti_only_{timestamp}.csv", index=False)
    results["seti_only"] = len(seti_only)

    # ---------- 3. Patch 尺度匹配 ----------
    overlaps = []
    for _, row_s in df_seti.iterrows():
        mask = (df_ml["freq_min"] <= row_s["freq_end"]) & \
               (df_ml["freq_max"] >= row_s["freq_start"])
        hits = df_ml[mask]
        if not hits.empty:
            for _, row_m in hits.iterrows():
                overlaps.append({
                    "seti_freq_start": row_s["freq_start"],
                    "seti_freq_end": row_s["freq_end"],
                    "ml_freq_min": row_m["freq_min"],
                    "ml_freq_max": row_m["freq_max"],
                    "SNR_seti": row_s["SNR"],
                    "SNR_ml": row_m["SNR"],
                    "cell_row": row_m["cell_row"],
                    "cell_col": row_m["cell_col"],
                })

    patch_overlap = pd.DataFrame(overlaps)
    # Deduplicate ML results by cell_row and cell_col
    patch_overlap_ml_dedup = patch_overlap.drop_duplicates(subset=["ml_freq_min", "ml_freq_max", "cell_row", "cell_col"])
    # Deduplicate SETI by frequency range
    patch_overlap_seti_dedup = patch_overlap.drop_duplicates(subset=["seti_freq_start", "seti_freq_end"])

    patch_overlap_seti_dedup.to_csv(f"{outdir}/patch_overlap_seti_{timestamp}.csv", index=False)
    patch_overlap_ml_dedup.to_csv(f"{outdir}/patch_overlap_ml_{timestamp}.csv", index=False)

    # Unique count after dedup
    unique_count = len(patch_overlap_seti_dedup)  # Based on unique SETI intervals
    results["patch_overlap"] = unique_count

    # ---------- 4. 统计信息 ----------
    with open(f"{outdir}/stats_{timestamp}.txt", "w") as f:
        f.write("=== Dataset Statistics ===\n")
        f.write(f"ML total: {len(df_ml)}\n")
        f.write(f"SETI total: {len(df_seti)}\n\n")
        f.write("=== Results ===\n")
        for k, v in results.items():
            f.write(f"{k}: {v}\n")

    # ---------- 5. 可视化 ----------
    plt.figure()
    df_ml["SNR"].hist(alpha=0.5, label="ML")
    df_seti["SNR"].hist(alpha=0.5, label="SETI")
    plt.legend()
    plt.title("SNR Distribution")
    plt.savefig(f"{outdir}/snr_hist_{timestamp}.png")

    plt.figure()
    plt.scatter(df_ml["Uncorrected_Frequency"], df_ml["SNR"], alpha=0.5, label="ML")
    plt.scatter(df_seti["Uncorrected_Frequency"], df_seti["SNR"], alpha=0.5, label="SETI")
    plt.legend()
    plt.title("Frequency vs SNR")
    plt.xlabel("Frequency")
    plt.ylabel("SNR")
    plt.savefig(f"{outdir}/freq_snr_scatter_{timestamp}.png")

    return results


def summarize_dataframe(df: pd.DataFrame):
    """
    Display a summary overview of a DataFrame
    """
    print(f"\n[\033[32mInfo\033[0m] Running overview of DataFrame with shape {df.shape}")
    print("\n===== Basic Info =====")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    print("\n===== Column Data Types =====")
    print(df.dtypes)

    print("\n===== Missing Values =====")
    missing = df.isnull().sum()
    missing_percent = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({'Missing Count': missing, 'Missing Percent': missing_percent})
    print(missing_df)

    print("\n===== Descriptive Statistics =====")
    print(df.describe(include='all').transpose())

    print("\n===== Unique Values per Column =====\n")
    unique_counts = df.nunique()
    print(unique_counts)


if __name__ == "__main__":
    # mlf = '../../pipeline/log/spliced_blc00010203040506o7o0111213141516o7o0212223242526o7o031323334353637_guppi_58060_26569_HIP17147_0021.gpuspec.0002/hits_20250818_181041.dat'
    setif = '../test_out/truboseti_blis692ns/spliced_blc00010203040506o7o0111213141516o7o0212223242526o7o031323334353637_guppi_58060_26569_HIP17147_0021/spliced_blc00010203040506o7o0111213141516o7o0212223242526o7o031323334353637_guppi_58060_26569_HIP17147_0021.gpuspec.0000.dat'
    mlf = '../../pipeline/log/spliced_blc00010203040506o7o0111213141516o7o0212223242526o7o031323334353637_guppi_58060_26569_HIP17147_0021.gpuspec.0000_chunk30720000_part0/hits_20250819_135738.dat'
    # setif = '../test_out/truboseti_blis692ns/spliced_blc00010203040506o7o0111213141516o7o0212223242526o7o031323334353637_guppi_58060_26569_HIP17147_0021/spliced_blc00010203040506o7o0111213141516o7o0212223242526o7o031323334353637_guppi_58060_26569_HIP17147_0021.gpuspec.0000_chunk30720000_part0.dat'
    df_ml = load_ML_dat(mlf)
    # df_ml = load_ML_csv(mlf)
    df_seti = load_seti_dat(setif)

    summarize_dataframe(df_ml)
    summarize_dataframe(df_seti)
    input("[\033[32mInfo\033[0m] Press to start analysis...")

    analyze_dataframe(df_ml, df_seti)