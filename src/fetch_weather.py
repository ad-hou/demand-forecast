"""Telecharge la temperature moyenne journaliere (Open-Meteo, archive)."""
import json
import urllib.request

import pandas as pd

URL = ("https://archive-api.open-meteo.com/v1/archive?latitude=48.78&longitude=2.29"
       "&start_date=2006-12-17&end_date=2010-11-25"
       "&daily=temperature_2m_mean&timezone=Europe%2FParis")


def main():
    with urllib.request.urlopen(URL, timeout=60) as f:
        d = json.load(f)["daily"]
    w = pd.DataFrame({"temp": d["temperature_2m_mean"]}, index=pd.to_datetime(d["time"]))
    w.index.name = "date"
    w.to_parquet("data/weather.parquet")

    k = pd.read_parquet("data/daily.parquet")["kwh"]
    j = k.to_frame().join(w).dropna()
    print("Jours meteo :", len(w), "| manquants :", int(w["temp"].isna().sum()))
    print("Jours communs avec la consommation :", len(j))
    print("Temperature : min %.1f | moyenne %.1f | max %.1f" % (
        w["temp"].min(), w["temp"].mean(), w["temp"].max()))
    print("Correlation temperature / kWh : %.2f" % j["temp"].corr(j["kwh"]))
    for name, months in [("hiver (dec-fev)", [12, 1, 2]), ("ete (juin-aout)", [6, 7, 8])]:
        s = j[j.index.month.isin(months)]
        print("  %s : %.2f" % (name, s["temp"].corr(s["kwh"])))


if __name__ == "__main__":
    main()
