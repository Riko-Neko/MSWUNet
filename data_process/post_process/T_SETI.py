import pandas as pd

SETI_COLUMNS = [
    "Top_Hit_#", "Drift_Rate", "SNR",
    "Uncorrected_Frequency", "Corrected_Frequency",
    "Index", "freq_start", "freq_end",
    "SEFD", "SEFD_freq",
    "Coarse_Channel_Number", "Full_number_of_hits"
]


def load_seti_dat(filepath: str) -> pd.DataFrame:
    """
    Load TurboSETI output .dat file (hit table) and return a pandas DataFrame.
    Use preset column headers to avoid relying on the file's own # header line.
    """
    df = pd.read_csv(
        filepath,
        delim_whitespace=True,  # Delimited by whitespace
        comment="#",  # Ignore comments
        names=SETI_COLUMNS,  # Use preset column headers
        engine="c"
    )
    return df
