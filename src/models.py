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
