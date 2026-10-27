import numpy as np
import pandas as pd
import pytest

from cfc.datasets.lbnl_chiller import (BASIC_POINTS, CASES, FAULT_TYPES, FAULTY, POWER_POINTS,
                                       _to_si, supply_excess)


def test_case_inventory():
    assert len(CASES) == 24 and len(FAULTY) == 23
    assert {CASES[c].fault for c in FAULTY} == set(FAULT_TYPES)
    assert len({c.file for c in CASES.values()}) == 24
    assert all(0 < CASES[c].level <= 1 for c in FAULTY)
    undocumented = [c for c in FAULTY if not FAULT_TYPES[CASES[c].fault].documented]
    assert sorted(undocumented) == ["chiller_fouling_65", "chiller_fouling_95"]
    assert CASES["dp_bias_-20"].file.endswith("pressure_bias_-020.csv")
    assert CASES["dp_bias_+10"].file.endswith("pressure_bias_010.csv")


def test_diagnosed_points_are_basic_points():
    points = [t.point for t in FAULT_TYPES.values() if t.point]
    assert set(points) <= set(BASIC_POINTS)
    assert not set(POWER_POINTS) & set(BASIC_POINTS)


def raw_frame():
    cols = {"OA_TEMP": [50.0], "OA_TEMP_WB": [68.0], "CWL_SEC_SW_TEMP": [56.0],
            "CWL_SEC_RW_TEMP": [44.6], "CWL_PRI_SW_TEMPSPT": [44.6], "CWL_SEC_CW_FLOW": [100.0],
            "CWL_SEC_DP": [100.0], "CWL_SEC_DPSPT": [100.0], "CWL_SEC_LOAD": [5000.0]}
    return pd.DataFrame(cols, index=pd.DatetimeIndex(["2018-01-01 01:00"]))


def test_si_conversion_swaps_the_mislabelled_pairs():
    df = _to_si(raw_frame())
    assert df.OA_TEMP.iloc[0] == pytest.approx(20.0)       # was labelled wet bulb (68 F)
    assert df.OA_TEMP_WB.iloc[0] == pytest.approx(10.0)    # was labelled dry bulb (50 F)
    assert df.CWL_SEC_SW_TEMP.iloc[0] == pytest.approx(7.0)  # supply is the colder column
    assert df.CWL_SEC_RW_TEMP.iloc[0] == pytest.approx(13.333, abs=1e-3)
    assert df.CWL_SEC_CW_FLOW.iloc[0] == pytest.approx(6.30902)
    assert df.CWL_SEC_DP.iloc[0] == pytest.approx(24.9089)
    assert df.CWL_SEC_LOAD.iloc[0] == pytest.approx(5.0)
    assert supply_excess(df).iloc[0] == pytest.approx(0.0)
