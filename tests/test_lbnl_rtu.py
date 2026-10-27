import pandas as pd
import pytest

from cfc.datasets.lbnl_rtu import BASIC_POINTS, CASES, FAULT_TYPES, FAULTY, _to_si
from cfc.virtual_meter import RTU_COMPONENTS


def test_case_inventory():
    assert len(CASES) == 25 and len(FAULTY) == 24
    assert {CASES[c].fault for c in FAULTY} == set(FAULT_TYPES)
    assert len({c.file for c in CASES.values()}) == 25
    assert all(0 < CASES[c].level <= 1 for c in FAULTY)
    assert CASES["liquid_line_1"].file == "RTU_sim_liquidpipe01bar.csv"
    assert CASES["suction_line_9"].level == pytest.approx(1.0)


def test_meter_inputs_are_basic_points():
    inputs = {p for members in RTU_COMPONENTS.values() for m in members for p in m.inputs}
    assert inputs <= set(BASIC_POINTS)


def test_si_conversion():
    raw = pd.DataFrame({"RTU_OA_TEMP": [68.0], "RTU_OA_HUM": [50.0], "ZA_TEMP": [77.0],
                        "RTU_TOT_WATT": [2000.0], "RTU_SA_FLOW": [1000.0],
                        "RTU_REFG_SUCT_PRES": [1.5e6]})
    df = _to_si(raw)
    assert "RTU_OA_HUM" not in df.columns
    assert df.OA_TEMP.iloc[0] == pytest.approx(20.0)
    assert df.ZA_TEMP.iloc[0] == pytest.approx(25.0)
    assert df.RTU_TOT_POW.iloc[0] == pytest.approx(2.0)
    assert df.RTU_SA_FLOW.iloc[0] == pytest.approx(471.947)
    assert df.RTU_REFG_SUCT_PRES.iloc[0] == pytest.approx(1500.0)
