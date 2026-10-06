"""Genere la prevision finale (7 jours) et les metriques lues par l'app."""
import json

import pandas as pd

from src.backtest import HORIZON, score
from src.models import ets_prophet

BEST = "ets_prophet"


def main():
    k = pd.read_parquet("data/daily.parquet")["kwh"]
    history = k.interpolate(limit_area="inside").ffill().bfill()
    pred = ets_prophet(history, HORIZON)

    r = pd.read_parquet("data/backtest_results.parquet")
    models = {name: score(g) for name, g in r.groupby("model")}

    best = r[r["model"] == BEST].dropna(subset=["y", "pred"]).copy()
    best["resid"] = best["y"] - best["pred"]
    q10, q90 = (float(v) for v in best["resid"].quantile([0.1, 0.9]))

    origins = sorted(best["origin"].unique())
    cut = origins[len(origins) // 2]
    lo, hi = best[best["origin"] < cut]["resid"].quantile([0.1, 0.9])
    test = best[best["origin"] >= cut]["resid"]
    holdout = float(((test >= lo) & (test <= hi)).mean())

    dates = pd.date_range(history.index[-1] + pd.Timedelta(days=1), periods=HORIZON)
    fc = pd.DataFrame({"pred": pred, "low": pred + q10, "high": pred + q90}, index=dates)
    fc.index.name = "date"
    fc.to_parquet("data/forecast.parquet")

    metrics = {
        "best_model": BEST,
        "n_windows": int(r["origin"].nunique()),
        "q10": q10,
        "q90": q90,
        "holdout_coverage": holdout,
        "last_date": str(history.index[-1].date()),
        "models": models,
    }
    with open("models/metrics.json", "w", encoding="ascii") as f:
        json.dump(metrics, f, indent=2)

    print(fc.round(2).to_string())
    print(f"Fourchette 80 pct : [{q10:.2f} ; {q90:+.2f}] kWh")
    print(f"Couverture hors echantillon : {holdout * 100:.1f} pct")


if __name__ == "__main__":
    main()
