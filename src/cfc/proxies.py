"""Priority proxies a planner could use instead of an estimated consequence.

* documented intensity, relative to the largest intensity of the same type;
* a fault-type priority: the mean consequence of the other simulated cases of
  the same type, as a priority list compiled from a fault library would give;
  a type with no other case gets the mean of all other cases;
* the anomaly magnitude of cfc.estimators.AnomalyScore.
"""

from __future__ import annotations

import pandas as pd

from .datasets.lbnl_chiller import CASES


def intensity_level(case: pd.Series, cases: dict = CASES) -> pd.Series:
    return case.map(lambda c: cases[c].level)


def type_priority(case_consequence: pd.Series, cases: dict = CASES) -> pd.Series:
    """Leave-one-case-out mean consequence of the case's fault type, indexed by case."""
    fault = pd.Series({c: cases[c].fault for c in case_consequence.index})
    prior = {}
    for c in case_consequence.index:
        same = (fault == fault[c]) & (fault.index != c)
        others = case_consequence[same] if same.any() else case_consequence.drop(c)
        prior[c] = others.mean()
    return pd.Series(prior)
