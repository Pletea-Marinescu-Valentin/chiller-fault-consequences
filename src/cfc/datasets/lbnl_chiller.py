"""Loader for the LBNL FDD data set "Chiller plant", doi:10.25984/1881324.

The archive holds one CSV per simulated year (EnergyPlus-Modelica, TMY Chicago,
1-min samples): one fault-free year and one year per fault type and intensity,
all with the same weather and schedules. Point names follow the data inventory
(Granderson et al., LBNL, 2022). The loader converts units to SI and fixes two
pairs of swapped labels, both verified against the plant's own equations:

* OA_TEMP and OA_TEMP_WB: the column labelled wet bulb exceeds the one
  labelled dry bulb in 99 % of the minutes, the tower setpoint
  max(T_wb + 8 F, 60 F) is reproduced only by the column labelled OA_TEMP,
  and the chilled-water reset only by the column labelled OA_TEMP_WB;
* CWL_SEC_SW_TEMP and CWL_SEC_RW_TEMP: the logged load CWL_SEC_LOAD equals
  rho c_p V (T_return - T_supply) only with the two columns exchanged.

The archive also holds two chiller-fouling years that the inventory does not
list (chiller_fouling_065 and _095).
"""

from __future__ import annotations

import zipfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
ARCHIVE = REPO / "data" / "raw" / "lbnl_chiller" / "LBNL_FDD_Data_Sets_Chiller_Plant.zip"
CACHE = REPO / "data" / "interim" / "lbnl_chiller"

GPM_TO_LPS = 0.0630902
INH2O_TO_KPA = 0.249089
DT_HOURS = 1.0 / 60.0


@dataclass(frozen=True)
class Case:
    name: str
    file: str
    fault: str | None
    intensity: float  # in the unit of FAULT_TYPES[fault].unit; 0 for the fault-free year
    level: float  # documented intensity relative to the largest of its type, in (0, 1]


@dataclass(frozen=True)
class FaultType:
    label: str
    component: str
    unit: str
    point: str | None = None  # BAS point the diagnosis names: the faulty sensor or actuator
    documented: bool = True


FAULT_TYPES = {
    "ct_bias": FaultType("Tower leaving-water sensor bias", "cooling tower 1", "K", "CT_SW_TEMP_1"),
    "chiller_bias": FaultType("Chiller leaving-water sensor bias", "chiller 1", "K",
                              "CHL_SW_TEMP_1"),
    "dp_bias": FaultType("Secondary-loop DP sensor bias", "secondary loop", "%", "CWL_SEC_DP"),
    "valve_leak": FaultType("Tower bypass valve leakage", "three-way valve", "%", "TWV_CTRL"),
    "valve_stuck": FaultType("Tower bypass valve stuck", "three-way valve", "%", "TWV_CTRL"),
    "ct_fouling": FaultType("Tower fouling", "cooling tower 1", "% UA"),
    "chiller_fouling": FaultType("Chiller fouling", "chiller 1", "% UA", documented=False),
    "ct_pi": FaultType("Tower PI mistuning", "cooling tower", "-"),
}


def _cases() -> dict[str, Case]:
    rows = [("fault_free", "ChillerPlant.csv", None, 0.0, 0.0)]
    for b in (-2, -1, 1, 2):
        rows.append((f"ct_bias_{b:+d}", f"ChillerPlant_coolingtower_bias_{b}.csv", "ct_bias",
                     float(b), abs(b) / 2))
        rows.append((f"chiller_bias_{b:+d}", f"ChillerPlant_chiller_bias_{b}.csv",
                     "chiller_bias", float(b), abs(b) / 2))
    for b in (-20, -10, 10, 20):
        sign = "-" if b < 0 else ""
        rows.append((f"dp_bias_{b:+d}",
                     f"ChillerPlant_secondary_chilled_water_pressure_bias_{sign}{abs(b):03d}.csv",
                     "dp_bias", float(b), abs(b) / 20))
    for p in (25, 50, 75):
        rows.append((f"valve_leak_{p}", f"ChillerPlant_bypass_leakage_{p:03d}.csv",
                     "valve_leak", float(p), p / 75))
    for p in (50, 75):
        rows.append((f"valve_stuck_{p}", f"ChillerPlant_bypass_stuck_{p:03d}.csv",
                     "valve_stuck", float(p), p / 75))
    for ua in (95, 80, 65):
        rows.append((f"ct_fouling_{ua}", f"ChillerPlant_coolingtower_fouling_{ua:03d}.csv",
                     "ct_fouling", float(ua), (100 - ua) / 35))
    for ua in (95, 65):
        rows.append((f"chiller_fouling_{ua}", f"ChillerPlant_chiller_fouling_{ua:03d}.csv",
                     "chiller_fouling", float(ua), (100 - ua) / 35))
    rows.append(("ct_pi", "ChillerPlant_coolingtower_PI.csv", "ct_pi", 1.0, 1.0))
    return {r[0]: Case(*r) for r in rows}


CASES = _cases()
FAULTY = [c for c in CASES if CASES[c].fault is not None]

COMPONENT_POWER = {
    "chillers": ["CHL_POW_1", "CHL_POW_2", "CHL_POW_3"],
    "towers": ["CT_POW_1", "CT_POW_2", "CT_POW_3"],
    "condenser pumps": ["CDWL_PM_POW_1", "CDWL_PM_POW_2", "CDWL_PM_POW_3"],
    "primary pumps": ["CWL_PRI_PM_POW_1", "CWL_PRI_PM_POW_2", "CWL_PRI_PM_POW_3"],
    "secondary pumps": ["CWL_SEC_PM_POW_1", "CWL_SEC_PM_POW_2"],
}
POWER_POINTS = [p for pts in COMPONENT_POWER.values() for p in pts]

# "Basic point" column of the inventory: points a typical BAS trends.
BASIC_POINTS = [
    "CWL_SEC_RW_TEMP", "CWL_SEC_SW_TEMP", "TWV_CTRL", "CT_SW_TEMPSPT", "CDWL_SW_TEMP",
    "CDWL_RW_TEMP", "CWL_PRI_RW_TEMP", "CWL_PRI_SW_TEMP", "CWL_PRI_SW_TEMPSPT", "OA_TEMP",
    "OA_TEMP_WB", "CWL_SEC_DPSPT", "CWL_SEC_DP",
    *[f"{p}_{i}" for p in ("CT_STA", "CT_RW_TEMP", "CT_SW_TEMP", "CT_FAN_SPD",
                           "CT_FAN_SPD_CTRL") for i in (1, 2, 3)],
    *[f"{p}_{i}" for p in ("CHL_STA", "CHL_RW_TEMP", "CHL_SW_TEMP", "CHL_SWCD_TEMP",
                           "CHL_RWCD_TEMP") for i in (1, 2, 3)],
    *[f"{p}_{i}" for p in ("CWL_SEC_PM_SPD", "CWL_SEC_PM_STA") for i in (1, 2)],
]
WEATHER = ["OA_TEMP", "OA_TEMP_WB"]

SWAPPED = {"OA_TEMP": "OA_TEMP_WB", "OA_TEMP_WB": "OA_TEMP",
           "CWL_SEC_SW_TEMP": "CWL_SEC_RW_TEMP", "CWL_SEC_RW_TEMP": "CWL_SEC_SW_TEMP"}


def _to_si(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.rename(columns=SWAPPED)
    for col in df.columns:
        if "TEMP" in col:
            df[col] = (df[col] - 32.0) / 1.8
        elif "FLOW" in col:
            df[col] = df[col] * GPM_TO_LPS
        elif col in ("CWL_SEC_DP", "CWL_SEC_DPSPT"):
            df[col] = df[col] * INH2O_TO_KPA
    df["CWL_SEC_LOAD"] = df["CWL_SEC_LOAD"] / 1000.0  # W -> kW
    return df


def read_raw(case: str, archive: Path = ARCHIVE) -> pd.DataFrame:
    """Read one case straight from the archive, with the original labels and units."""
    with zipfile.ZipFile(archive) as zf, zf.open(CASES[case].file) as fh:
        raw = pd.read_csv(fh, parse_dates=["Datetime"], index_col="Datetime")
    raw.index.name = "time"
    return raw


def read_csv(case: str, archive: Path = ARCHIVE) -> pd.DataFrame:
    """Read one case straight from the archive, in SI units with the labels fixed."""
    return _to_si(read_raw(case, archive))


def _cache_path(case: str, hourly: bool = False) -> Path:
    return CACHE / ("hourly" if hourly else "minute") / f"{case}.parquet"


def plant_power(df: pd.DataFrame) -> pd.Series:
    """Total electric power of chillers, tower fans and pumps [kW]."""
    return df[POWER_POINTS].sum(axis=1)


def component_power(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({k: df[v].sum(axis=1) for k, v in COMPONENT_POWER.items()})


def supply_excess(df: pd.DataFrame) -> pd.Series:
    """Chilled water delivered to the secondary loop above the plant setpoint [K], >= 0."""
    return (df["CWL_SEC_SW_TEMP"] - df["CWL_PRI_SW_TEMPSPT"]).clip(lower=0.0)


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, compression="zstd")
    tmp.replace(path)


def _build(case: str) -> str:
    """Cache one case at 1-min resolution and as hourly means.

    The hourly file adds PLANT_POW [kW] and SUP_EXCESS [K], both averaged from
    1-min values, so that their hourly means times 1 h are energy and K h.
    """
    df = read_csv(case)
    _write(df, _cache_path(case))
    df["PLANT_POW"] = plant_power(df)
    df["SUP_EXCESS"] = supply_excess(df)
    _write(df.resample("h").mean(), _cache_path(case, hourly=True))
    return case


def build_cache(workers: int = 4) -> None:
    """Convert every CSV of the archive to parquet files under data/interim/lbnl_chiller/."""
    todo = [c for c in CASES
            if not (_cache_path(c).exists() and _cache_path(c, hourly=True).exists())]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for case in pool.map(_build, todo):
            print(f"lbnl_chiller/{case}: cached", flush=True)


def load_case(case: str, points: list[str] | None = None, hourly: bool = True) -> pd.DataFrame:
    """Load one simulated year as hourly means (default) or at 1-min resolution.

    Hourly means of power in kW equal the energy in kWh of that hour.
    """
    path = _cache_path(case, hourly)
    if not path.exists():
        _build(case)
    return pd.read_parquet(path, columns=points)


def case_table() -> pd.DataFrame:
    rows = []
    for c in CASES.values():
        ft = FAULT_TYPES.get(c.fault)
        rows.append({"case": c.name, "fault": c.fault, "intensity": c.intensity,
                     "level": c.level, "label": ft.label if ft else "Fault-free",
                     "documented": ft.documented if ft else True})
    return pd.DataFrame(rows).set_index("case")


def week_of_year(index: pd.DatetimeIndex) -> np.ndarray:
    """Week number 0..51 counted from the first sample (the year starts on a Monday)."""
    return ((index - index[0].normalize()).days // 7).to_numpy()
