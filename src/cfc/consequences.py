"""Ground-truth consequences from the paired simulations.

Each faulty year and the fault-free year share weather and schedules, so the
consequence of a fault in week w is the paired difference

    dE(w) = sum_{t in w} [P_f(t) - P_0(t)]      energy penalty [kWh]
    dS(w) = sum_{t in w} [X_f(t) - X_0(t)]      service deficit [K h]

with P the plant electric power and X the excess of the chilled-water supply
temperature over its setpoint. A maintenance planner values one week of the
fault at

    C_kappa(w) = dE(w) + kappa * max(dS(w), 0),

where kappa [kWh/(K h)] prices one degree-hour of degraded supply. A fault that
improves the supply earns no credit; one that saves energy keeps its negative
dE.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .context import N_WEEKS, weekly_sum
from .datasets.lbnl_chiller import COMPONENT_POWER, FAULTY, load_case


def cost(d_energy: np.ndarray, d_service: np.ndarray, kappa: float) -> np.ndarray:
    return np.asarray(d_energy) + kappa * np.clip(np.asarray(d_service), 0.0, None)


def weekly_truth(cases: list[str] | None = None) -> pd.DataFrame:
    """One row per (case, week): energy penalty, service deficit and per-component penalty."""
    cases = FAULTY if cases is None else cases
    ff = load_case("fault_free")
    rows = []
    for case in cases:
        d = load_case(case)
        if not d.index.equals(ff.index):
            raise ValueError(f"{case}: time index differs from the fault-free year")
        out = {"case": np.repeat(case, N_WEEKS), "week": np.arange(N_WEEKS),
               "d_energy": weekly_sum(d.PLANT_POW - ff.PLANT_POW, d.index),
               "d_service": weekly_sum(d.SUP_EXCESS - ff.SUP_EXCESS, d.index)}
        for comp, cols in COMPONENT_POWER.items():
            out[f"d_{comp.replace(' ', '_')}"] = weekly_sum(
                d[cols].sum(axis=1) - ff[cols].sum(axis=1), d.index)
        rows.append(pd.DataFrame(out))
    return pd.concat(rows, ignore_index=True)
