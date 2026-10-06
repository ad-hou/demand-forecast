"""Backtesting a origine glissante (fenetre expansive), sans fuite de donnees."""
import numpy as np
import pandas as pd

from src.metrics import evaluate

INITIAL_TRAIN = 730
HORIZON = 7
STEP = 28


def rolling_origin(series, forecaster, initial=INITIAL_TRAIN, horizon=HORIZON, step=STEP):
    """Pour chaque origine, le modele ne recoit que l'historique AVANT l'origine.

    Les trous de l'historique sont combles par interpolation sur cet historique
    tronque uniquement : aucune valeur posterieure a l'origine n'est utilisee.
    """
    rows = []
    for origin in range(initial, len(series) - horizon + 1, step):
        history = series.iloc[:origin].interpolate(limit_area="inside").ffill().bfill()
        actual = series.iloc[origin:origin + horizon]
        pred = np.asarray(forecaster(history, horizon), dtype=float)
        for h, (date, y, p) in enumerate(zip(actual.index, actual.to_numpy(), pred), 1):
            rows.append({"origin": series.index[origin], "date": date,
                         "h": h, "y": y, "pred": p})
    return pd.DataFrame(rows)


def score(results):
    """Metriques par fenetre (jours exclus ignores), puis moyenne des fenetres."""
    ok = results.dropna(subset=["y", "pred"])
    per_window = [evaluate(g["y"], g["pred"]) for _, g in ok.groupby("origin")]
    return pd.DataFrame(per_window).mean().to_dict()
