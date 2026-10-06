"""Chargement et preparation de la consommation electrique (UCI)."""
import pandas as pd

RAW_PATH = "data/raw/household_power_consumption.txt"
TARGET = "Global_active_power"
MIN_COVERAGE = 0.8  # part minimale de minutes valides pour garder un jour


def load_minute(path=RAW_PATH):
    """Lit le fichier brut : une ligne par minute, '?' = valeur manquante."""
    df = pd.read_csv(path, sep=";", na_values="?", low_memory=False,
                     usecols=["Date", "Time", TARGET])
    df.index = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
    return df[[TARGET]].astype("float64").rename_axis("timestamp")


def to_daily(minute_df, min_coverage=MIN_COVERAGE):
    """Energie journaliere (kWh), jours partiels ou trop incomplets en NaN."""
    s = minute_df[TARGET]
    n_valid = s.resample("D").count()
    daily = pd.DataFrame({
        "kwh_somme": s.resample("D").sum(min_count=1) / 60.0,
        "kwh": (s.resample("D").mean() * 24.0).where(n_valid >= min_coverage * 1440),
        "minutes_valides": n_valid,
    })
    # premier et dernier jours : journees partielles dans le fichier brut
    return daily.iloc[1:-1]
