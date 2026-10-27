"""Loader for the simulated rooftop unit of the LBNL FDD data sets "RTU", doi:10.25984/1881324.

NREL simulated a 5-ton rooftop unit (two-stage scroll compressor, Modelica)
serving a single-zone small office (EnergyPlus, Spawn co-simulation) with
TMY3 San Francisco weather from 20 July to 28 October, once fault-free and
once per fault type and intensity, at 1-min samples. The zone cooling setpoint
is 23 degC at all times. The loader converts units to SI and adds hourly
PLANT_POW [kW], the total RTU power, and SUP_EXCESS [K], here the zone air
temperature above its setpoint, so the chiller-plant pipeline applies unchanged.

RTU_OA_HUM differs between runs by up to 43 percentage points while the
outdoor temperature agrees within 0.004 F, so it is not used as weather.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from .lbnl_chiller import Case, FaultType

REPO = Path(__file__).resolve().parents[3]
ARCHIVE = REPO / "data" / "raw" / "lbnl_rtu" / "LBNL_FDD_Data_Sets_RTU.zip"
CACHE = REPO / "data" / "interim" / "lbnl_rtu"
FOLDER = "Simulated RTU/"
CFM_TO_LPS = 0.471947

FAULT_TYPES = {
    "cond_fouling": FaultType("Condenser fouling", "condenser", "% airflow"),
    "evap_fouling": FaultType("Evaporator fouling", "evaporator", "% airflow"),
    "liquid_line": FaultType("Liquid-line restriction", "liquid line", "bar"),
    "suction_line": FaultType("Suction-line restriction", "suction line", "bar"),
    "overcharge": FaultType("Refrigerant overcharge", "refrigerant circuit", "%"),
    "undercharge": FaultType("Refrigerant undercharge", "refrigerant circuit", "%"),
}


def _cases() -> dict[str, Case]:
    rows = [("fault_free", "RTU_sim_baseline.csv", None, 0.0, 0.0)]
    for p in (10, 20, 30, 40, 50):
        rows.append((f"cond_fouling_{p}", f"RTU_sim_condfouling{p}.csv", "cond_fouling", p, p / 50))
        rows.append((f"evap_fouling_{p}", f"RTU_sim_evapfouling{p}.csv", "evap_fouling", p, p / 50))
    for b in (1, 4, 8, 10):
        rows.append((f"liquid_line_{b}", f"RTU_sim_liquidpipe{b:02d}bar.csv", "liquid_line", b, b / 10))
    for b in (1, 3, 6, 9):
        rows.append((f"suction_line_{b}", f"RTU_sim_suctionpipe{b:02d}bar.csv", "suction_line", b,
                     b / 9))
    for p in (10, 15, 20):
        rows.append((f"overcharge_{p}", f"RTU_sim_overcharge{p}.csv", "overcharge", p, p / 20))
        rows.append((f"undercharge_{p}", f"RTU_sim_undercharge{p}.csv", "undercharge", p, p / 20))
    return {r[0]: Case(r[0], r[1], r[2], float(r[3]), float(r[4])) for r in rows}


CASES = _cases()
FAULTY = [c for c in CASES if CASES[c].fault is not None]

COMPONENT_POWER = {"compressor": ["RTU_COMP_POW"], "fan": ["RTU_SA_FAN_POW"]}
# "Basic point" column of the inventory
BASIC_POINTS = ["RTU_SA_TEMP", "RTU_RA_TEMP", "OA_TEMP", "RTU_STG_STA", "ZA_TEMP", "ZA_TEMP_SPT"]


def _to_si(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.drop(columns=["RTU_OA_HUM"]).rename(columns={
        "RTU_OA_TEMP": "OA_TEMP", "RTU_TOT_WATT": "RTU_TOT_POW", "RTU_COMP_WATT": "RTU_COMP_POW",
        "RTU_SA_FAN_WATT": "RTU_SA_FAN_POW"})
    for col in df.columns:
        if "TEMP" in col:
            df[col] = (df[col] - 32.0) / 1.8
        elif "FLOW" in col:
            df[col] = df[col] * CFM_TO_LPS
        elif "PRES" in col:
            df[col] = df[col] / 1000.0  # Pa -> kPa
        elif col.endswith("_POW") or "CAPA" in col:
            df[col] = df[col] / 1000.0  # W -> kW
    return df


def read_csv(case: str, archive: Path = ARCHIVE) -> pd.DataFrame:
    with zipfile.ZipFile(archive) as zf, zf.open(FOLDER + CASES[case].file) as fh:
        raw = pd.read_csv(fh, parse_dates=["Datetime"], index_col="Datetime")
    raw.index.name = "time"
    return _to_si(raw)


def _cache_path(case: str, hourly: bool = True) -> Path:
    return CACHE / ("hourly" if hourly else "minute") / f"{case}.parquet"


def _build(case: str) -> None:
    df = read_csv(case)
    df["PLANT_POW"] = df["RTU_TOT_POW"]
    df["SUP_EXCESS"] = (df["ZA_TEMP"] - df["ZA_TEMP_SPT"]).clip(lower=0.0)
    for hourly, frame in ((False, df), (True, df.resample("h").mean())):
        path = _cache_path(case, hourly)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, compression="zstd")


def load_case(case: str, points: list[str] | None = None, hourly: bool = True) -> pd.DataFrame:
    """Load one simulated run as hourly means (default) or at 1-min resolution."""
    if not _cache_path(case, hourly).exists():
        _build(case)
    df = pd.read_parquet(_cache_path(case, hourly), columns=points)
    # The last sample (28 October, 00:00) opens a day that is not simulated
    return df[df.index < df.index[-1].normalize()] if hourly else df
