"""Exogenous operating context and the weekly evaluation folds.

The context z holds only quantities that a plant fault cannot change: outdoor
dry- and wet-bulb temperatures, their recent averages, the hour, the day of the
week and public holidays. Counterfactual (fault-free) models are functions of z.

Every faulty run shares the weather of its fault-free run, so a model fitted
on fault-free week w would know the counterfactual of week w exactly. All
models are therefore cross-fitted over interleaved weekly folds: week w belongs
to fold w mod n_folds and is predicted by a model that never saw that fold.
Weeks are counted from the first day of the data; the chiller-plant year starts
on Monday 1 January 2018 and has 52 full weeks (31 December is not evaluated).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

YEAR_START = pd.Timestamp("2018-01-01")
N_WEEKS = 52  # full weeks of the chiller-plant year

# US federal holidays of 2018, the calendar of the simulated buildings
HOLIDAYS = pd.to_datetime(["2018-01-01", "2018-01-15", "2018-02-19", "2018-05-28",
                           "2018-07-04", "2018-09-03", "2018-10-08", "2018-11-12",
                           "2018-11-22", "2018-12-25"])


def week_index(index: pd.DatetimeIndex) -> np.ndarray:
    return ((index - index[0].normalize()).days // 7).to_numpy()


def n_weeks(index: pd.DatetimeIndex) -> int:
    """Number of full weeks between the first and the last day of the data."""
    return ((index[-1].normalize() - index[0].normalize()).days + 1) // 7


def folds(index: pd.DatetimeIndex, n_folds: int) -> np.ndarray:
    return week_index(index) % n_folds


def context(df: pd.DataFrame, windows: tuple[int, ...] = (6, 24, 72)) -> pd.DataFrame:
    """Hourly context features from the logged outdoor temperatures and the calendar."""
    z = pd.DataFrame(index=df.index)
    weather = {name: col for name, col in (("oat", "OA_TEMP"), ("oawb", "OA_TEMP_WB"))
               if col in df.columns}
    for name, col in weather.items():
        z[name] = df[col]
    for h in windows:
        for name, col in weather.items():
            z[f"{name}_{h}h"] = df[col].rolling(h, min_periods=1).mean()
    z["hour"] = df.index.hour
    z["dow"] = df.index.day_of_week
    z["holiday"] = df.index.normalize().isin(HOLIDAYS).astype(int)
    return z


def weekly_sum(values: np.ndarray | pd.Series, index: pd.DatetimeIndex) -> np.ndarray:
    """Sum of hourly values per full week (hourly mean kW -> kWh, K -> K h)."""
    s = pd.Series(np.asarray(values, dtype=float)).groupby(week_index(index)).sum()
    return s.reindex(range(n_weeks(index))).to_numpy()


def weekly_mean(values: np.ndarray | pd.Series, index: pd.DatetimeIndex) -> np.ndarray:
    s = pd.Series(np.asarray(values, dtype=float)).groupby(week_index(index)).mean()
    return s.reindex(range(n_weeks(index))).to_numpy()
