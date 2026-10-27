import numpy as np
import pandas as pd
import pytest

from cfc.context import YEAR_START, folds
from cfc.virtual_meter import VirtualMeter, diagnosed_inputs, input_points

PARAMS = {"n_estimators": 200, "learning_rate": 0.1, "num_leaves": 15, "min_child_samples": 10}


def synthetic_plant(n_hours=26 * 168, seed=0):
    """Component powers as known functions of their own inputs, plus a decoy sensor."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(YEAR_START, periods=n_hours, freq="h")
    load = np.clip(0.5 + 0.4 * np.sin(2 * np.pi * np.arange(n_hours) / 24)
                   + 0.1 * rng.standard_normal(n_hours), 0.05, 1.0)
    d = pd.DataFrame(index=idx)
    for i in (1, 2, 3):
        on = (load > (i - 1) / 3).astype(float)
        d[f"CHL_STA_{i}"] = on
        d[f"CT_STA_{i}"] = on
        d[f"CHL_SW_TEMP_{i}"] = 7 + rng.normal(0, 0.3, n_hours)
        d[f"CHL_RW_TEMP_{i}"] = d[f"CHL_SW_TEMP_{i}"] + 5 * load * on
        d[f"CHL_RWCD_TEMP_{i}"] = 25 + 5 * load + rng.normal(0, 0.5, n_hours)
        d[f"CHL_SWCD_TEMP_{i}"] = d[f"CHL_RWCD_TEMP_{i}"] + 4 * load * on
        d[f"CHL_POW_{i}"] = on * (20 * (d[f"CHL_RW_TEMP_{i}"] - d[f"CHL_SW_TEMP_{i}"])
                                  + 2 * (d[f"CHL_SWCD_TEMP_{i}"] - d[f"CHL_SW_TEMP_{i}"]))
        d[f"CT_FAN_SPD_{i}"] = on * np.clip(load + rng.normal(0, 0.05, n_hours), 0, 1)
        d[f"CT_POW_{i}"] = 10 * d[f"CT_FAN_SPD_{i}"] ** 3
        d[f"CWL_PRI_PM_POW_{i}"] = 15 * on
        d[f"CT_SW_TEMP_{i}"] = d[f"CHL_RWCD_TEMP_{i}"] - 0.1  # decoy: not a meter input
    for i in (1, 2):
        d[f"CWL_SEC_PM_STA_{i}"] = 1.0 if i == 1 else (load > 0.6).astype(float)
        d[f"CWL_SEC_PM_SPD_{i}"] = d[f"CWL_SEC_PM_STA_{i}"] * (0.3 + 0.7 * load)
        d[f"CWL_SEC_PM_POW_{i}"] = 50 * d[f"CWL_SEC_PM_SPD_{i}"] ** 3
    d["TWV_CTRL"] = np.clip(0.6 - 0.5 * load, 0, 1)
    cond = 8 * (d.CHL_STA_1 + d.CHL_STA_2 + d.CHL_STA_3) + 20 * d.TWV_CTRL
    for i in (1, 2, 3):
        d[f"CDWL_PM_POW_{i}"] = cond / 3
    d["CWL_PRI_SW_TEMP"] = d.CHL_SW_TEMP_1
    power_cols = [c for c in d.columns if "_POW_" in c]
    d["PLANT_POW"] = d[power_cols].sum(axis=1)
    return d


def test_meter_recovers_plant_power_out_of_fold():
    d = synthetic_plant()
    fold = folds(d.index, 13)
    vm = VirtualMeter(PARAMS, 13).fit(d, fold)
    pred = vm.predict(d, fold)
    assert np.mean(np.abs(pred - d.PLANT_POW)) / d.PLANT_POW.mean() < 0.05


def test_meter_ignores_points_that_drive_no_component():
    d = synthetic_plant()
    fold = folds(d.index, 13)
    vm = VirtualMeter(PARAMS, 13).fit(d, fold)
    biased = d.copy()
    biased["CT_SW_TEMP_1"] -= 2.0  # a tower sensor bias feeds no component model
    assert np.allclose(vm.predict(biased, fold), vm.predict(d, fold))
    assert "CT_SW_TEMP_1" not in input_points()


def test_diagnosis_substitutes_or_drops_the_named_point():
    d = synthetic_plant(n_hours=200)
    frame, drop = diagnosed_inputs(d.assign(CHL_SW_TEMP_1=d.CHL_SW_TEMP_1 + 2), "CHL_SW_TEMP_1")
    assert drop == () and np.allclose(frame.CHL_SW_TEMP_1, d.CWL_PRI_SW_TEMP)
    assert diagnosed_inputs(d, "TWV_CTRL")[1] == ("TWV_CTRL",)
    assert diagnosed_inputs(d, "CWL_SEC_DP")[1] == ()
    assert diagnosed_inputs(d, None)[1] == ()


def test_dropped_input_is_not_used():
    d = synthetic_plant()
    fold = folds(d.index, 13)
    vm = VirtualMeter(PARAMS, 13, drop=("TWV_CTRL",)).fit(d, fold)
    stuck = d.assign(TWV_CTRL=1.0)
    assert np.allclose(vm.predict(stuck, fold), vm.predict(d, fold))


def test_envelope_flags_inputs_outside_the_fault_free_range():
    d = synthetic_plant()
    fold = folds(d.index, 13)
    vm = VirtualMeter(PARAMS, 13).fit(d, fold)
    assert not vm.outside(d).any()
    hot = d.copy()
    hot.iloc[:10, hot.columns.get_loc("CHL_SWCD_TEMP_1")] += 40.0
    out = vm.outside(hot)
    assert out[:10].all() and not out[10:].any()
