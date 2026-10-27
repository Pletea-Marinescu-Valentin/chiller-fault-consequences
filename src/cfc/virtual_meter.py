"""Virtual plant power meter from the inventory's basic BAS points.

Plant power is the sum of component models, each fed only with the points that
drive that component: a chiller's power from its status and water
temperatures, a tower fan's from its speed, a pump's from its status and speed.
Identical units (three chillers, three towers, three primary pumps, two
secondary pumps) share one pooled model. The condenser pumps share one group
model on the chiller statuses and the bypass-valve command.

A black-box map from all basic points to plant power reads seasonal
correlations instead (a slow tower fan "means" a light load) and inverts the
sign of the penalty of a tower sensor bias; the component structure avoids that.

The models are fitted on the fault-free year only. Because a fault is assumed
diagnosed, the meter uses the diagnosis: a faulty sensor with a redundant header
sensor is replaced by it, and an actuator command that no longer reflects the
actuator is removed from the inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .estimators import CrossFitted


@dataclass(frozen=True)
class Member:
    targets: tuple[str, ...]  # power columns summed into one target
    inputs: tuple[str, ...]


def _pooled(power: str, inputs: tuple[str, ...], units: tuple[int, ...]) -> list[Member]:
    return [Member((power.format(i=i),), tuple(p.format(i=i) for p in inputs)) for i in units]


COMPONENTS = {
    "chillers": _pooled("CHL_POW_{i}", ("CHL_STA_{i}", "CHL_RW_TEMP_{i}", "CHL_SW_TEMP_{i}",
                                        "CHL_SWCD_TEMP_{i}", "CHL_RWCD_TEMP_{i}"), (1, 2, 3)),
    "towers": _pooled("CT_POW_{i}", ("CT_STA_{i}", "CT_FAN_SPD_{i}"), (1, 2, 3)),
    "secondary pumps": _pooled("CWL_SEC_PM_POW_{i}", ("CWL_SEC_PM_STA_{i}", "CWL_SEC_PM_SPD_{i}"),
                               (1, 2)),
    "primary pumps": _pooled("CWL_PRI_PM_POW_{i}", ("CHL_STA_{i}",), (1, 2, 3)),
    "condenser pumps": [Member(("CDWL_PM_POW_1", "CDWL_PM_POW_2", "CDWL_PM_POW_3"),
                               ("CHL_STA_1", "CHL_STA_2", "CHL_STA_3", "TWV_CTRL"))],
}

# Rooftop unit: compressor power from the stage and the air temperatures around the
# coils, supply-fan power from the stage (the fan speed follows the stage).
RTU_COMPONENTS = {
    "compressor": [Member(("RTU_COMP_POW",), ("RTU_STG_STA", "OA_TEMP", "RTU_SA_TEMP",
                                               "RTU_RA_TEMP"))],
    "fan": [Member(("RTU_SA_FAN_POW",), ("RTU_STG_STA",))],
}

# Redundant sensors downstream of a unit sensor: the primary supply header
# carries chiller 1's leaving water whenever chiller 1 runs alone.
SUBSTITUTES = {"CHL_SW_TEMP_1": "CWL_PRI_SW_TEMP"}


def input_points() -> list[str]:
    return sorted({p for members in COMPONENTS.values() for m in members for p in m.inputs})


def diagnosed_inputs(frame: pd.DataFrame, point: str | None) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Frame to feed the meter and inputs to drop, given the diagnosed point."""
    if point in SUBSTITUTES:
        frame = frame.copy()
        frame[point] = frame[SUBSTITUTES[point]]
        return frame, ()
    if point in input_points():
        return frame, (point,)
    return frame, ()


class VirtualMeter:
    """Sum of cross-fitted component models; `components` defaults to the chiller plant."""

    def __init__(self, params: dict, n_folds: int, seed: int = 0, drop: tuple[str, ...] = (),
                 components: dict[str, list[Member]] | None = None):
        self.params, self.n_folds, self.seed, self.drop = params, n_folds, seed, tuple(drop)
        self.components = COMPONENTS if components is None else components

    def _inputs(self, member: Member) -> list[str]:
        return [p for p in member.inputs if p not in self.drop]

    def fit(self, ff: pd.DataFrame, fold: np.ndarray) -> "VirtualMeter":
        self.models = {}
        for name, members in self.components.items():
            X = np.vstack([ff[self._inputs(m)].to_numpy() for m in members])
            y = np.concatenate([ff[list(m.targets)].sum(axis=1).to_numpy() for m in members])
            self.models[name] = CrossFitted(self.params, self.n_folds, self.seed).fit(
                X, y, np.tile(fold, len(members)))
        used = sorted({p for ms in self.components.values() for m in ms for p in self._inputs(m)})
        lo, hi = ff[used].min(), ff[used].max()
        self.envelope = (lo, hi)
        return self

    def component_power(self, frame: pd.DataFrame, fold: np.ndarray) -> pd.DataFrame:
        out = {}
        for name, members in self.components.items():
            out[name] = sum(self.models[name].predict(frame[self._inputs(m)].to_numpy(), fold)
                            for m in members)
        return pd.DataFrame(out, index=frame.index)

    def predict(self, frame: pd.DataFrame, fold: np.ndarray) -> np.ndarray:
        return self.component_power(frame, fold).sum(axis=1).to_numpy()

    def outside(self, frame: pd.DataFrame, margin: float = 0.05) -> np.ndarray:
        """Hours in which any input leaves the fault-free range widened by `margin`."""
        lo, hi = self.envelope
        span = (hi - lo).replace(0.0, 1.0)
        x = frame[lo.index]
        return ((x < lo - margin * span) | (x > hi + margin * span)).any(axis=1).to_numpy()
