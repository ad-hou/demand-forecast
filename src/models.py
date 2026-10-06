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


def _prophet(history, horizon, vacances):
    import logging

    from prophet import Prophet

    from src.calendar_fr import school_holiday

    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    logging.getLogger("prophet").setLevel(logging.ERROR)
    df = history.rename("y").rename_axis("ds").reset_index()
    m = Prophet(weekly_seasonality=True, yearly_seasonality=True,
                daily_seasonality=False)
    m.add_country_holidays(country_name="FR")
    if vacances:  # le calendrier scolaire est connu a l'avance : pas de fuite
        df["vacances"] = school_holiday(df["ds"])
        m.add_regressor("vacances")
    m.fit(df)
    future = m.make_future_dataframe(periods=horizon, include_history=False)
    if vacances:
        future["vacances"] = school_holiday(future["ds"])
    return m.predict(future)["yhat"].to_numpy()


def prophet(history, horizon):
    """Prophet : saisonnalites hebdo + annuelle, jours feries francais."""
    return _prophet(history, horizon, vacances=False)


def prophet_vacances(history, horizon):
    """Prophet + vacances scolaires zone C comme variable explicative."""
    return _prophet(history, horizon, vacances=True)


LAGS = (1, 2, 7, 14, 28)


def _features(values, date, vacances=False):
    """Variables d'un jour a partir du passe uniquement (values = historique)."""
    row = {f"lag_{l}": values[-l] for l in LAGS}
    if vacances:
        from src.calendar_fr import school_holiday
        row["vacances"] = float(school_holiday([date])[0])
    row["mean_7"] = float(np.mean(values[-7:]))
    row["mean_28"] = float(np.mean(values[-28:]))
    row["dow"] = date.dayofweek
    row["month"] = date.month
    return row


def _lightgbm(history, horizon, vacances):
    """LightGBM, prevision recursive : chaque prediction rejoint l'historique."""
    import lightgbm as lgb
    import pandas as pd

    values = history.to_numpy(dtype=float)
    dates = history.index
    rows, target = [], []
    for i in range(max(LAGS), len(values)):
        rows.append(_features(values[:i], dates[i], vacances))
        target.append(values[i])
    model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
                              num_leaves=15, min_child_samples=20,
                              random_state=42, verbose=-1)
    model.fit(pd.DataFrame(rows), target)

    hist = list(values)
    out = []
    for h in range(1, horizon + 1):
        date = dates[-1] + pd.Timedelta(days=h)
        x = pd.DataFrame([_features(np.asarray(hist), date, vacances)])
        p = float(model.predict(x)[0])
        out.append(p)
        hist.append(p)
    return np.asarray(out)


def lightgbm(history, horizon):
    return _lightgbm(history, horizon, vacances=False)


def lightgbm_vacances(history, horizon):
    """LightGBM + indicateur de vacances scolaires zone C."""
    return _lightgbm(history, horizon, vacances=True)


def ets_prophet_vacances(history, horizon):
    """Moyenne de ETS et Prophet avec vacances scolaires."""
    return (ets(history, horizon) + prophet_vacances(history, horizon)) / 2


def ets_prophet(history, horizon):
    """Moyenne simple de ETS et Prophet."""
    return (ets(history, horizon) + prophet(history, horizon)) / 2
