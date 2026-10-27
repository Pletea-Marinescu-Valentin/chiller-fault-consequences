"""Ground-truth consequences from the paired simulations.

Each faulty run and the fault-free run share weather and schedules, so the
consequence of a fault in week w is the paired difference

    dE(w) = sum_{t in w} [P_f(t) - P_0(t)]      energy penalty [kWh]
    dS(w) = sum_{t in w} [X_f(t) - X_0(t)]      service deficit [K h]

with P the electric power and X the temperature excess that measures service:
the chilled-water supply above its setpoint for the chiller plant, the zone
above its cooling setpoint for the rooftop unit. A maintenance planner values
one week of the fault at

    C_kappa(w) = dE(w) + kappa * max(dS(w), 0),

where kappa [kWh/(K h)] prices one degree-hour of degraded service. A fault
that improves the service earns no credit; one that saves energy keeps its
negative dE.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from .context import n_weeks, weekly_sum
from .datasets import lbnl_chiller


def cost(d_energy: np.ndarray, d_service: np.ndarray, kappa: float) -> np.ndarray:
    return np.asarray(d_energy) + kappa * np.clip(np.asarray(d_service), 0.0, None)


def weekly_truth(cases: list[str] | None = None,
                 load: Callable[[str], pd.DataFrame] = lbnl_chiller.load_case,
                 baseline: str = "fault_free",
                 components: dict[str, list[str]] = lbnl_chiller.COMPONENT_POWER) -> pd.DataFrame:
    """One row per (case, week): energy penalty, service deficit and per-component penalty.

    `load` returns hourly frames with PLANT_POW [kW] and SUP_EXCESS [K].
    """
    cases = lbnl_chiller.FAULTY if cases is None else cases
    ff = load(baseline)
    weeks = n_weeks(ff.index)
    rows = []
    for case in cases:
        d = load(case)
        if not d.index.equals(ff.index):
            raise ValueError(f"{case}: time index differs from the fault-free run")
        out = {"case": np.repeat(case, weeks), "week": np.arange(weeks),
               "d_energy": weekly_sum(d.PLANT_POW - ff.PLANT_POW, d.index),
               "d_service": weekly_sum(d.SUP_EXCESS - ff.SUP_EXCESS, d.index)}
        for comp, cols in components.items():
            out[f"d_{comp.replace(' ', '_')}"] = weekly_sum(
                d[cols].sum(axis=1) - ff[cols].sum(axis=1), d.index)
        rows.append(pd.DataFrame(out))
    return pd.concat(rows, ignore_index=True)
