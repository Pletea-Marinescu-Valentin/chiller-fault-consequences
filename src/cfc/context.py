"""Exogenous operating context and the weekly evaluation folds.

The context z holds only quantities that a plant fault cannot change: outdoor
dry- and wet-bulb temperatures, their recent averages, the hour, the day of the
week and public holidays. Counterfactual (fault-free) models are functions of z.

Every faulty year shares the weather of the fault-free year, so a model fitted
on fault-free week w would know the counterfactual of week w exactly. All
models are therefore cross-fitted over interleaved weekly folds: week w belongs
to fold w mod n_folds and is predicted by a model that never saw that fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

YEAR_START = pd.Timestamp("2018-01-01")  # a Monday; the simulated year is 2018
N_WEEKS = 52  # full weeks; the remaining day, 31 December, is not evaluated

# US federal holidays of 2018, the calendar of the simulated office building
HOLIDAYS = pd.to_datetime(["2018-01-01", "2018-01-15", "2018-02-19", "2018-05-28",
                           "2018-07-04", "2018-09-03", "2018-10-08", "2018-11-12",
                           "2018-11-22", "2018-12-25"])


def week_index(index: pd.DatetimeIndex) -> np.ndarray:
    return ((index - YEAR_START).days // 7).to_numpy()


def folds(index: pd.DatetimeIndex, n_folds: int) -> np.ndarray:
    return week_index(index) % n_folds


def context(df: pd.DataFrame, windows: tuple[int, ...] = (6, 24, 72)) -> pd.DataFrame:
    """Hourly context features from the logged outdoor temperatures and the calendar."""
    z = pd.DataFrame(index=df.index)
    z["oat"] = df["OA_TEMP"]
    z["oawb"] = df["OA_TEMP_WB"]
    for h in windows:
        z[f"oat_{h}h"] = df["OA_TEMP"].rolling(h, min_periods=1).mean()
        z[f"oawb_{h}h"] = df["OA_TEMP_WB"].rolling(h, min_periods=1).mean()
    z["hour"] = df.index.hour
    z["dow"] = df.index.day_of_week
    z["holiday"] = df.index.normalize().isin(HOLIDAYS).astype(int)
    return z


def weekly_sum(values: np.ndarray | pd.Series, index: pd.DatetimeIndex) -> np.ndarray:
    """Sum of hourly values per full week (hourly mean kW -> kWh, K -> K h)."""
    s = pd.Series(np.asarray(values, dtype=float)).groupby(week_index(index)).sum()
    return s.reindex(range(N_WEEKS)).to_numpy()


def weekly_mean(values: np.ndarray | pd.Series, index: pd.DatetimeIndex) -> np.ndarray:
    s = pd.Series(np.asarray(values, dtype=float)).groupby(week_index(index)).mean()
    return s.reindex(range(N_WEEKS)).to_numpy()
