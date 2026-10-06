"""Les vacances scolaires (zone C) ameliorent-elles la prevision ? Compare chaque modele
avec et sans la variable, sur les memes fenetres, avec un bootstrap sur les fenetres.

Usage : python -m src.run_vacances   (resultats : data/backtest_vacances.parquet)
"""
import numpy as np
import pandas as pd

from src.backtest import rolling_origin
from src.models import ets_prophet_vacances, lightgbm_vacances, prophet_vacances

PAIRS = [("prophet", "prophet_vacances"), ("lightgbm", "lightgbm_vacances"),
         ("ets_prophet", "ets_prophet_vacances")]


def mae_by_window(results, model):
    g = results[results["model"] == model].dropna(subset=["y", "pred"])
    return g.groupby("origin").apply(lambda x: (x["y"] - x["pred"]).abs().mean())


def bootstrap_diff(a, b, n=10000, seed=0):
    """Ecart moyen de MAE (b - a) sur les fenetres, IC 95 % : negatif = b meilleur."""
    d = (b - a).to_numpy()
    rng = np.random.default_rng(seed)
    means = [rng.choice(d, len(d)).mean() for _ in range(n)]
    return d.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def main():
    k = pd.read_parquet("data/daily.parquet")["kwh"]
    runs = []
    for name, f in {"prophet_vacances": prophet_vacances,
                    "lightgbm_vacances": lightgbm_vacances,
                    "ets_prophet_vacances": ets_prophet_vacances}.items():
        runs.append(rolling_origin(k, f).assign(model=name))
        print(name, "ok", flush=True)
    new = pd.concat(runs)
    new.to_parquet("data/backtest_vacances.parquet")
    allres = pd.concat([pd.read_parquet("data/backtest_results.parquet"), new])
    for a, b in PAIRS:
        m, lo, hi = bootstrap_diff(mae_by_window(allres, a), mae_by_window(allres, b))
        print(f"{b} vs {a} : {m:+.3f} kWh/jour (IC 95 % : {lo:+.3f} a {hi:+.3f})")


if __name__ == "__main__":
    main()
