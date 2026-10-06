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
