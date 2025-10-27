import pandas as pd

# Mask模式的标准列
ML_MASK_COLUMNS = [
    "DriftRate", "SNR", "Uncorrected_Frequency",
    "freq_start", "freq_end",
    "cell_row", "cell_col",
    "freq_min", "freq_max",
    "time_start", "time_end"
]

# Detection模式的标准列
ML_DETECTION_COLUMNS = [
    "DriftRate", "SNR", "Uncorrected_Frequency",
    "freq_start", "freq_end",
    "confidence",
    "cell_row", "cell_col",
    "freq_min", "freq_max",
    "time_start", "time_end",
    "mode"
]


def load_ML_dat(filepath: str, mode: str) -> pd.DataFrame:
    """
    Load ML dat file (space-separated, no header).
    """
    df = pd.read_csv(
        filepath,
        delim_whitespace=True,  # space/tab-separated
        names=ML_MASK_COLUMNS if mode == "mask" else ML_DETECTION_COLUMNS,
        header=0,
        comment="#",  # allow comments
        engine="c"  # faster
    )
    return df


def load_ML_csv(filepath: str, mode: str) -> pd.DataFrame:
    """
    Load ML csv file (comma-separated, with header).
    """
    df = pd.read_csv(
        filepath,
        usecols=ML_MASK_COLUMNS if mode == "mask" else ML_DETECTION_COLUMNS
        # only use needed columns (prevent extra fields)
    )
    return df
