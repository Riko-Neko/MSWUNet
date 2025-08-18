from pathlib import Path

import pandas as pd


def load_pipeline_dat(path: str):
    """
    Load a .dat file saved by SETIPipelineProcessor or TurboSETI.
    Returns:
        df (pd.DataFrame): hits table data
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")

    # Attempt to read the file header (lines starting with #)
    header_lines = []
    with open(path, "r") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            if not line.startswith("#"):
                f.seek(pos)
                break
            header_lines.append(line.strip())

    # Read table data
    # If the file has a header, use the first line as column names; otherwise, specify default column names
    df = pd.read_csv(path, sep='\t', comment='#', engine='python')

    # If DataFrame has no column names (empty or unnamed), specify default column names
    if df.columns.str.contains("Unnamed").all():
        df.columns = [
            "Top_Hit_#", "Drift_Rate", "SNR", "Uncorrected_Frequency", "Corrected_Frequency",
            "Index", "freq_start", "freq_end", "SEFD", "SEFD_freq", "Coarse_Channel_Number", "Full_number_of_hits"
        ]
    return df


def analyze_hits(file1: str, file2: str, output_dir: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1️⃣ Read dat files
    df1 = load_pipeline_dat(file1)
    df2 = load_pipeline_dat(file2)

    # 2️⃣ Compare to find overlapping signals (common hits)
    common_hits = []
    df1_used = set()
    df2_used = set()

    for i1, row1 in df1.iterrows():
        for i2, row2 in df2.iterrows():
            # Determine if frequency ranges overlap
            if max(row1['freq_start'], row2['freq_start']) <= min(row1['freq_end'], row2['freq_end']):
                common_hits.append({
                    'df1_index': i1,
                    'df2_index': i2,
                    'freq_start': max(row1['freq_start'], row2['freq_start']),
                    'freq_end': min(row1['freq_end'], row2['freq_end']),
                    'SNR_df1': row1['SNR'],
                    'SNR_df2': row2['SNR'],
                    'DriftRate_df1': row1['Drift_Rate'],
                    'DriftRate_df2': row2['Drift_Rate']
                })
                df1_used.add(i1)
                df2_used.add(i2)

    df_common = pd.DataFrame(common_hits)

    # 3️⃣ Single-sided hits
    df1_only = df1.loc[~df1.index.isin(df1_used)]
    df2_only = df2.loc[~df2.index.isin(df2_used)]

    # 4️⃣ Statistical analysis
    def stats(df):
        if df.empty:
            return {'count': 0, 'SNR_mean': None, 'SNR_max': None, 'SNR_min': None,
                    'Drift_mean': None, 'Drift_max': None, 'Drift_min': None}
        return {
            'count': len(df),
            'SNR_mean': df['SNR'].mean(),
            'SNR_max': df['SNR'].max(),
            'SNR_min': df['SNR'].min(),
            'Drift_mean': df['Drift_Rate'].mean(),
            'Drift_max': df['Drift_Rate'].max(),
            'Drift_min': df['Drift_Rate'].min()
        }

    stats_df1_only = stats(df1_only)
    stats_df2_only = stats(df2_only)
    stats_common = stats(df_common[['SNR_df1', 'SNR_df2', 'DriftRate_df1', 'DriftRate_df2']].rename(
        columns={'SNR_df1': 'SNR', 'SNR_df2': 'SNR2', 'DriftRate_df1': 'Drift', 'DriftRate_df2': 'Drift2'}
    ))

    # 5️⃣ Save files
    # Comparison file
    common_file = output_dir / "hits_comparison.dat"
    df_common.to_csv(common_file, sep='\t', index=False)

    # Statistical results file
    stats_file = output_dir / "hits_statistics.dat"
    stats_df = pd.DataFrame([
        {'type': 'df1_only', **stats_df1_only},
        {'type': 'df2_only', **stats_df2_only},
        {'type': 'common', **stats_common}
    ])
    stats_df.to_csv(stats_file, sep='\t', index=False)

    print(f"✅ Analysis completed, results saved to: {output_dir}")
    return df1_only, df2_only, df_common, stats_df


if __name__ == '__main__':
    file1 = "../pipeline/log/seti_pipeline.dat"
    file2 = "../data_process/test_out/truboseti_blis692ns/spliced_blc00010203040506o7o0111213141516o7o0212223242526o7o031323334353637_guppi_58060_26569_HIP17147_0021/spliced_blc00010203040506o7o0111213141516o7o0212223242526o7o031323334353637_guppi_58060_26569_HIP17147_0021.gpuspec.0000.dat"
    output_dir = "./analysis_out"
    analyze_hits(file1, file2, output_dir)
