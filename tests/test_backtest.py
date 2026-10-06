import numpy as np
import pandas as pd

from src.backtest import rolling_origin, score
from src.models import naive, seasonal_naive


def make_series(n=900, nan_at=()):
    idx = pd.date_range("2008-01-01", periods=n, freq="D")
    s = pd.Series(10.0 + (idx.dayofweek.to_numpy() >= 5) * 5.0, index=idx)
    for i in nan_at:
        s.iloc[i] = np.nan
    return s


def test_no_future_data_in_history():
    s = make_series(nan_at=[800])
    seen = []

    def spy(history, horizon):
        seen.append(history.index[-1])
        return np.zeros(horizon)

    res = rolling_origin(s, spy, initial=730, horizon=7, step=28)
    for origin, last in zip(sorted(res["origin"].unique()), seen):
        assert last < origin


def test_seasonal_naive_7_is_exact_on_weekly_pattern():
    res = rolling_origin(make_series(), seasonal_naive(7))
    assert score(res)["mae"] == 0.0


def test_naive_is_worse_than_seasonal_on_weekly_pattern():
    s = make_series()
    assert score(rolling_origin(s, naive))["mae"] > 0.0


def test_missing_actual_days_are_ignored():
    s = make_series(nan_at=[735])
    res = rolling_origin(s, seasonal_naive(7))
    assert res["y"].isna().sum() >= 1
    assert score(res)["mae"] == 0.0


def test_history_gap_is_filled_without_nan():
    s = make_series(nan_at=[700, 701])
    seen = []

    def spy(history, horizon):
        seen.append(history.isna().sum())
        return np.zeros(horizon)

    rolling_origin(s, spy)
    assert max(seen) == 0
