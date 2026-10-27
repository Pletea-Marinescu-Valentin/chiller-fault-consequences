"""Data audit: inventory, swapped labels, shared weather, near duplicates, plant regimes."""

import zipfile

import numpy as np
import pandas as pd

from _common import TABLES, fmt_int, load_config, write_numbers, write_results
from cfc.context import N_WEEKS, weekly_mean, weekly_sum
from cfc.datasets.lbnl_chiller import (ARCHIVE, CASES, FAULT_TYPES, FAULTY, WEATHER, load_case,
                                       read_raw)

NAME = "tab_audit"
RHO_CP = 998.0 * 4186.0  # J/(m3 K)
GPM_TO_M3S = 0.0630902e-3


def label_checks(cfg: dict) -> dict:
    """Evidence for the two swapped label pairs, on the raw fault-free file (deg F)."""
    raw = read_raw("fault_free")
    out = {"wb_above_db_pct": 100.0 * float((raw.OA_TEMP_WB > raw.OA_TEMP).mean())}
    for label in ("OA_TEMP", "OA_TEMP_WB"):
        tower = np.maximum(raw[label] + cfg["tower_approach_f"], cfg["tower_min_f"])
        out[f"tower_q95_k_{label}"] = float((tower - raw.CT_SW_TEMPSPT).abs().quantile(0.95) / 1.8)
        r = cfg["reset_f"]
        slope = (r["chw_max"] - r["chw_min"]) / (r["oat_min"] - r["oat_max"])
        reset = np.clip(r["chw_min"] + slope * (raw[label] - r["oat_max"]), r["chw_min"], r["chw_max"])
        out[f"reset_q95_k_{label}"] = float((reset - raw.CWL_PRI_SW_TEMPSPT).abs().quantile(0.95) / 1.8)
    loaded = raw.CWL_SEC_LOAD > 1e4
    flow = raw.CWL_SEC_CW_FLOW * GPM_TO_M3S
    as_labelled = RHO_CP * flow * (raw.CWL_SEC_RW_TEMP - raw.CWL_SEC_SW_TEMP) / 1.8
    out["load_corr_as_labelled"] = float(np.corrcoef(raw.CWL_SEC_LOAD[loaded], as_labelled[loaded])[0, 1])
    out["load_ratio_exchanged"] = float((raw.CWL_SEC_LOAD[loaded] / -as_labelled[loaded]).median())
    out["reset_min_f"] = float(raw.CWL_PRI_SW_TEMPSPT.min())
    out["reset_max_f"] = float(raw.CWL_PRI_SW_TEMPSPT.max())
    out["chiller1_on_pct"] = 100.0 * float((raw.CHL_STA_1 > 0.5).mean())
    return out


def inventory() -> pd.DataFrame:
    ff = load_case("fault_free", WEATHER, hourly=False)
    rows = []
    for case, meta in CASES.items():
        d = load_case(case, hourly=False)
        rows.append({"case": case, "file": meta.file, "fault": meta.fault,
                     "documented": FAULT_TYPES[meta.fault].documented if meta.fault else True,
                     "rows": len(d), "start": str(d.index[0]), "end": str(d.index[-1]),
                     "missing": int(d.isna().sum().sum()),
                     "weather_max_diff_k": float((d[WEATHER] - ff[WEATHER]).abs().max().max())})
    return pd.DataFrame(rows)


def duplicates(tol: float) -> list[tuple[str, str]]:
    ff = load_case("fault_free", ["PLANT_POW", "SUP_EXCESS"])
    annual = {}
    for case in FAULTY:
        d = load_case(case, ["PLANT_POW", "SUP_EXCESS"])
        annual[case] = np.array([(d.PLANT_POW - ff.PLANT_POW).sum(), (d.SUP_EXCESS - ff.SUP_EXCESS).sum()])
    pairs = []
    for i, a in enumerate(FAULTY):
        for b in FAULTY[i + 1:]:
            if CASES[a].fault == CASES[b].fault:
                continue
            rel = np.abs(annual[a] - annual[b]) / np.maximum(np.abs(annual[a]), 1.0)
            if np.all(rel < tol):
                pairs.append((a, b))
    return pairs


def regimes(cfg: dict) -> dict:
    ff = load_case("fault_free")
    flow = weekly_mean(ff.CWL_SEC_CW_FLOW, ff.index)
    cooling = weekly_sum(ff.CWL_SEC_LOAD, ff.index)
    energy = weekly_sum(ff.PLANT_POW, ff.index)
    idle = flow < cfg["no_flow_lps"]
    no_load_flowing = (cooling < 1000.0) & ~idle  # < 1 MWh of cooling in the week
    weeks = np.flatnonzero(idle)
    return {"idle_weeks": weeks.tolist(), "idle_energy_mwh": float(energy[idle].mean() / 1000),
            "no_load_flowing_weeks": int(no_load_flowing.sum()),
            "no_load_flowing_energy_mwh": float(energy[no_load_flowing].mean() / 1000),
            "no_load_flowing_flow_lps": float(flow[no_load_flowing].mean())}


def main():
    cfg = load_config(NAME)
    with zipfile.ZipFile(ARCHIVE) as zf:
        n_files = len([n for n in zf.namelist() if n.endswith(".csv")])
    inv = inventory()
    checks = label_checks(cfg)
    pairs = duplicates(cfg["duplicate_tolerance"])
    reg = regimes(cfg)
    write_results(NAME, {"config": cfg, "files": n_files, "checks": checks,
                         "duplicates": pairs, "regimes": reg})
    TABLES.mkdir(parents=True, exist_ok=True)
    inv.to_csv(TABLES / f"{NAME}.csv", index=False)
    print(inv.to_string(index=False))
    print(checks, pairs, reg, sep="\n")

    numbers = {
        "files": f"{n_files}",
        "faulty cases": f"{len(FAULTY)}",
        "fault types": f"{len(FAULT_TYPES)}",
        "undocumented cases": f"{int((~inv.documented).sum())}",
        "rows": fmt_int(inv.rows.iloc[0]),
        "missing": f"{int(inv.missing.sum())}",
        "weather max diff": f"{inv.weather_max_diff_k.max():.2f}",
        "wb above db pct": f"{checks['wb_above_db_pct']:.1f}",
        "tower err true": f"{checks['tower_q95_k_OA_TEMP']:.1f}",
        "tower err labelled": f"{checks['tower_q95_k_OA_TEMP_WB']:.1f}",
        "reset err true": f"{checks['reset_q95_k_OA_TEMP_WB']:.1f}",
        "reset err labelled": f"{checks['reset_q95_k_OA_TEMP']:.1f}",
        "load corr labelled": f"{checks['load_corr_as_labelled']:.3f}".replace("-", "$-$"),
        "reset min": f"{(checks['reset_min_f'] - 32) / 1.8:.1f}",
        "reset max": f"{(checks['reset_max_f'] - 32) / 1.8:.1f}",
        "chiller one on pct": f"{checks['chiller1_on_pct']:.0f}",
        "duplicate pairs": f"{len(pairs)}",
        "idle weeks": f"{len(reg['idle_weeks'])}",
        "idle first week": f"{reg['idle_weeks'][0] + 1}" if reg["idle_weeks"] else "--",
        "idle last week": f"{reg['idle_weeks'][-1] + 1}" if reg["idle_weeks"] else "--",
        "idle energy": f"{reg['idle_energy_mwh']:.1f}",
        "no load flowing weeks": f"{reg['no_load_flowing_weeks']}",
        "no load flowing energy": f"{reg['no_load_flowing_energy_mwh']:.1f}",
        "no load flowing flow": f"{reg['no_load_flowing_flow_lps']:.0f}",
        "weeks": f"{N_WEEKS}",
    }
    write_numbers(NAME, numbers)
    for k, v in numbers.items():
        print(f"{k:26s} {v}")


if __name__ == "__main__":
    main()
