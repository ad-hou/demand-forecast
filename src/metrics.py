"""Metriques de prevision."""
import numpy as np


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def smape(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    denom = np.abs(y) + np.abs(p)
    return float(100 * np.mean(np.where(denom == 0, 0.0, 2 * np.abs(p - y) / denom)))


def evaluate(y, p):
    return {"mae": mae(y, p), "rmse": rmse(y, p), "smape": smape(y, p)}
