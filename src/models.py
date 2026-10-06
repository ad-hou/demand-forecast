"""Modeles de prevision : baselines. Un modele = f(history, horizon) -> array."""
import numpy as np


def naive(history, horizon):
    """Repete la derniere valeur observee."""
    return np.repeat(float(history.iloc[-1]), horizon)


def seasonal_naive(period):
    """Repete les `period` derniers jours (periode 7 = meme jour de semaine)."""
    def forecast(history, horizon):
        last = history.iloc[-period:].to_numpy(dtype=float)
        return np.tile(last, int(np.ceil(horizon / period)))[:horizon]
    return forecast
import warnings

import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")


def ets(history, horizon):
    """Holt-Winters additif, saisonnalite hebdomadaire, sans tendance."""
    fit = ExponentialSmoothing(history.to_numpy(dtype=float), trend=None,
                               seasonal="add", seasonal_periods=7,
                               initialization_method="estimated").fit()
    return np.asarray(fit.forecast(horizon))


def sarima(history, horizon):
    """SARIMA(1,0,1)(1,0,1,7)."""
    fit = SARIMAX(history.to_numpy(dtype=float), order=(1, 0, 1),
                  seasonal_order=(1, 0, 1, 7)).fit(disp=False)
    return np.asarray(fit.forecast(horizon))


def prophet(history, horizon):
    """Prophet : saisonnalites hebdo + annuelle, jours feries francais."""
    import logging

    from prophet import Prophet

    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    logging.getLogger("prophet").setLevel(logging.ERROR)
    df = history.rename("y").rename_axis("ds").reset_index()
    m = Prophet(weekly_seasonality=True, yearly_seasonality=True,
                daily_seasonality=False)
    m.add_country_holidays(country_name="FR")
    m.fit(df)
    future = m.make_future_dataframe(periods=horizon, include_history=False)
    return m.predict(future)["yhat"].to_numpy()


LAGS = (1, 2, 7, 14, 28)


def _features(values, date):
    """Variables d'un jour a partir du passe uniquement (values = historique)."""
    row = {f"lag_{l}": values[-l] for l in LAGS}
    row["mean_7"] = float(np.mean(values[-7:]))
    row["mean_28"] = float(np.mean(values[-28:]))
    row["dow"] = date.dayofweek
    row["month"] = date.month
    return row


def lightgbm(history, horizon):
    """LightGBM, prevision recursive : chaque prediction rejoint l'historique."""
    import lightgbm as lgb
    import pandas as pd

    values = history.to_numpy(dtype=float)
    dates = history.index
    rows, target = [], []
    for i in range(max(LAGS), len(values)):
        rows.append(_features(values[:i], dates[i]))
        target.append(values[i])
    model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
                              num_leaves=15, min_child_samples=20,
                              random_state=42, verbose=-1)
    model.fit(pd.DataFrame(rows), target)

    hist = list(values)
    out = []
    for h in range(1, horizon + 1):
        date = dates[-1] + pd.Timedelta(days=h)
        x = pd.DataFrame([_features(np.asarray(hist), date)])
        p = float(model.predict(x)[0])
        out.append(p)
        hist.append(p)
    return np.asarray(out)
