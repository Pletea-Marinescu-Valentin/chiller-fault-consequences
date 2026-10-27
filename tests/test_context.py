import numpy as np
import pandas as pd
import pytest

from cfc.consequences import cost
from cfc.context import N_WEEKS, YEAR_START, context, folds, week_index, weekly_sum


def hourly_year():
    index = pd.date_range(YEAR_START + pd.Timedelta(hours=1), "2018-12-31 23:00", freq="h")
    t = np.arange(len(index))
    return pd.DataFrame({"OA_TEMP": 10 + 10 * np.sin(2 * np.pi * t / 8760),
                         "OA_TEMP_WB": 5 + 8 * np.sin(2 * np.pi * t / 8760)}, index=index)


def test_weeks_start_on_monday_first_of_january():
    df = hourly_year()
    w = week_index(df.index)
    assert w[0] == 0 and w[-1] == 52  # 31 December is the 365th day
    assert (df.index[w == 1][0]) == pd.Timestamp("2018-01-08 00:00")


def test_folds_interleave_weeks():
    df = hourly_year()
    f = folds(df.index, 13)
    w = week_index(df.index)
    assert np.array_equal(f, w % 13)
    assert set(np.unique(w[f == 0])) == {0, 13, 26, 39, 52}


def test_weekly_sum_turns_mean_power_into_energy():
    df = hourly_year()
    e = weekly_sum(np.full(len(df), 2.0), df.index)  # 2 kW for every hour
    assert len(e) == N_WEEKS
    assert e[1] == pytest.approx(2.0 * 168)
    assert e[0] == pytest.approx(2.0 * 167)  # the first hour of the year is not logged


def test_context_uses_only_weather_and_calendar():
    z = context(hourly_year())
    assert {"oat", "oawb", "oat_24h", "hour", "dow", "holiday"} <= set(z.columns)
    assert z.loc["2018-07-04", "holiday"].eq(1).all()
    assert not z.isna().any().any()


def test_cost_prices_only_a_service_deficit():
    d_energy = np.array([100.0, -50.0, 10.0])
    d_service = np.array([2.0, 3.0, -4.0])
    assert np.allclose(cost(d_energy, d_service, 10.0), [120.0, -20.0, 10.0])
    assert np.allclose(cost(d_energy, d_service, 0.0), d_energy)
