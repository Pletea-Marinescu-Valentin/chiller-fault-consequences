"""Table: weekly consequence estimates from BAS data, with and without power meters."""

import time

import numpy as np
import pandas as pd

from _common import (GENERATED, TABLES, fmt_float, fmt_int, load_config, write_numbers,
                     write_results)
from cfc.context import N_WEEKS, folds, weekly_mean, weekly_sum
from cfc.datasets.lbnl_chiller import BASIC_POINTS, CASES, FAULT_TYPES, FAULTY, load_case
from cfc.estimators import AnomalyScore, counterfactual, penalty
from cfc.virtual_meter import VirtualMeter, diagnosed_inputs
from fig_consequences import SHORT

NAME = "tab_estimation"
ESTIMATES = "estimates_weekly"


def main():
    cfg = load_config(NAME)
    t0 = time.time()
    ff = load_case("fault_free")
    frames = {c: load_case(c) for c in FAULTY}
    fold = folds(ff.index, cfg["n_folds"])
    n, seed = cfg["n_folds"], cfg["seed"]

    # Metered: plant power meter and fault-free metered history
    base = counterfactual(ff, ff.PLANT_POW.to_numpy(), fold, cfg["counterfactual"], n, seed)
    cf_metered = penalty(ff.PLANT_POW.to_numpy(), base, ff, fold)
    # Service: supply-temperature excess, basic sensors in both settings
    base_x = counterfactual(ff, ff.SUP_EXCESS.to_numpy(), fold, cfg["counterfactual"], n, seed)
    cf_service = penalty(ff.SUP_EXCESS.to_numpy(), base_x, ff, fold)
    # Unmetered: virtual meter, one variant per set of inputs removed by a diagnosis
    variants = {}
    for c in [None, *FAULTY]:
        point = FAULT_TYPES[CASES[c].fault].point if c else None
        _, drop = diagnosed_inputs(ff, point)
        if drop not in variants:
            vm = VirtualMeter(cfg["virtual_meter"], n, seed, drop).fit(ff, fold)
            v0 = vm.predict(ff, fold)
            base_v = counterfactual(ff, v0, fold, cfg["counterfactual"], n, seed)
            variants[drop] = (vm, v0, base_v)
    vm0, v0, base_v0 = variants[()]
    cf_unmetered = penalty(v0, base_v0, ff, fold)
    vm_error_ff = weekly_sum(v0 - ff.PLANT_POW.to_numpy(), ff.index)
    print(f"models fitted in {time.time() - t0:.0f} s")

    points = [p for p in BASIC_POINTS if p not in cfg["anomaly_exclude"] and ff[p].std() > 1e-6]
    anomaly = AnomalyScore(points, cfg["anomaly"], n, seed).fit(ff, fold)
    anomaly_ff = anomaly.weekly(ff, fold)
    print(f"anomaly models fitted in {time.time() - t0:.0f} s ({len(points)} points)")

    rows = []
    for c in FAULTY:
        d = frames[c]
        point = FAULT_TYPES[CASES[c].fault].point
        frame, drop = diagnosed_inputs(d, point)
        vm, _, base_v = variants[drop]
        virtual = vm.predict(frame, fold)
        out = vm.outside(frame, cfg["envelope_margin"])
        rows.append(pd.DataFrame({
            "case": c, "fault": CASES[c].fault, "level": CASES[c].level, "week": np.arange(N_WEEKS),
            "d_energy": weekly_sum(d.PLANT_POW - ff.PLANT_POW, d.index),
            "d_service": weekly_sum(d.SUP_EXCESS - ff.SUP_EXCESS, d.index),
            "e_metered": penalty(d.PLANT_POW.to_numpy(), base, d, fold),
            "e_unmetered": penalty(virtual, base_v, d, fold),
            "s_hat": penalty(d.SUP_EXCESS.to_numpy(), base_x, d, fold),
            "anomaly": anomaly.weekly(d, fold),
            "vm_error": weekly_sum(virtual - d.PLANT_POW.to_numpy(), d.index),
            "outside": weekly_mean(out, d.index)}))
    est = pd.concat(rows, ignore_index=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    est.to_csv(TABLES / f"{ESTIMATES}.csv", index=False, float_format="%.4f")
    print(f"estimates done in {time.time() - t0:.0f} s")

    est["err_m"] = est.e_metered - est.d_energy
    est["err_u"] = est.e_unmetered - est.d_energy
    est["err_s"] = est.s_hat - est.d_service
    est["out_week"] = est.outside > cfg["envelope_week"]
    q95_m = np.quantile(np.abs(cf_metered), 0.95)
    by_type = est.groupby("fault").agg(
        true_abs=("d_energy", lambda s: s.abs().mean()), mae_m=("err_m", lambda s: s.abs().mean()),
        mae_u=("err_u", lambda s: s.abs().mean()), bias_u=("err_u", "mean"),
        vm_bias=("vm_error", "mean"), service_abs=("d_service", lambda s: s.abs().mean()),
        mae_s=("err_s", lambda s: s.abs().mean()), out=("out_week", "mean"),
        resolvable=("d_energy", lambda s: (s.abs() > q95_m).mean()))
    by_case = est.groupby("case").agg(mae_u=("err_u", lambda s: s.abs().mean()),
                                      vm_bias=("vm_error", "mean"), out=("out_week", "mean"))
    print(by_type.round(2).to_string())
    print(by_case.round(1).to_string())
    # The metered error is the counterfactual error of the fault-free year in the same week
    corr_m = np.corrcoef(est.err_m, np.tile(cf_metered, len(FAULTY)))[0, 1]

    write_results(NAME, {
        "config": cfg, "points_anomaly": points,
        "counterfactual_error_kwh": {"metered": cf_metered.tolist(), "unmetered": cf_unmetered.tolist()},
        "service_counterfactual_error_kh": cf_service.tolist(),
        "virtual_meter_error_ff_kwh": vm_error_ff.tolist(), "anomaly_ff": anomaly_ff.tolist(),
        "by_type": by_type.reset_index().to_dict("records"),
        "by_case": by_case.reset_index().to_dict("records")})

    order = ["chiller_bias", "dp_bias", "ct_bias", "ct_fouling", "chiller_fouling", "ct_pi",
             "valve_leak", "valve_stuck"]
    lines = ["\\begin{tabular}{lrrrrrr}", "\\hline",
             "Fault & $|\\Delta E|$ & \\multicolumn{2}{c}{MAE $\\Delta\\hat E$} & $|\\Delta S|$ & MAE "
             "& Out \\\\",
             " & & meter & virtual & & $\\Delta\\hat S$ & [\\%] \\\\",
             " & [kWh] & [kWh] & [kWh] & [K\\,h] & [K\\,h] & \\\\", "\\hline"]
    for f in order:
        r = by_type.loc[f]
        name = SHORT[f] + ("$^\\ast$" if not FAULT_TYPES[f].documented else "")
        lines.append(f"{name} & {fmt_int(r.true_abs)} & {fmt_int(r.mae_m)} & {fmt_int(r.mae_u)} & "
                     f"{fmt_int(r.service_abs)} & {fmt_float(r.mae_s)} & {100 * r.out:.0f} \\\\")
    lines.append(f"Fault-free & 0 & {fmt_int(np.abs(cf_metered).mean())} & "
                 f"{fmt_int(np.abs(cf_unmetered).mean())} & 0 & {fmt_float(np.abs(cf_service).mean())}"
                 f" & 0 \\\\")
    lines += ["\\hline", "\\end{tabular}"]
    GENERATED.mkdir(parents=True, exist_ok=True)
    (GENERATED / f"{NAME}_table.tex").write_text("\n".join(lines) + "\n")

    ff_week = weekly_sum(ff.PLANT_POW, ff.index)
    numbers = {
        "folds": f"{n}",
        "cf metered mae": fmt_int(np.abs(cf_metered).mean()),
        "cf metered pct": f"{100 * np.abs(cf_metered).mean() / ff_week.mean():.0f}",
        "cf metered qnf": fmt_int(q95_m),
        "cf metered max": fmt_int(np.abs(cf_metered).max()),
        "cf unmetered mae": fmt_int(np.abs(cf_unmetered).mean()),
        "metered error corr": f"{corr_m:.3f}",
        "vm ff mae": fmt_int(np.abs(vm_error_ff).mean()),
        "vm ff pct": fmt_float(100 * np.abs(vm_error_ff).mean() / ff_week.mean()),
        "service cf mae": fmt_float(np.abs(cf_service).mean()),
        "service cf qnf": fmt_float(np.quantile(np.abs(cf_service), 0.95)),
        "anomaly ff median": f"{np.median(anomaly_ff):.2f}",
        "anomaly points": f"{len(points)}",
        "resolvable pct": f"{100 * (est.d_energy.abs() > q95_m).mean():.0f}",
        "mae service all": fmt_float(est.err_s.abs().mean()),
        "out weeks pct": f"{100 * est.out_week.mean():.0f}",
    }
    for key, case in {"vm bias chiller minus two": "chiller_bias_-2",
                      "vm bias chiller fouling severe": "chiller_fouling_65",
                      "vm bias stuck large": "valve_stuck_75",
                      "vm bias tower minus two": "ct_bias_-2",
                      "vm bias dp minus twenty": "dp_bias_-20"}.items():
        numbers[key] = fmt_int(by_case.loc[case, "vm_bias"])
    for f in order:
        key = f.replace("_", " ")
        numbers[f"{key} resolvable pct"] = f"{100 * by_type.loc[f, 'resolvable']:.0f}"
        numbers[f"{key} out pct"] = f"{100 * by_type.loc[f, 'out']:.0f}"
    write_numbers(NAME, numbers)
    for k, v in numbers.items():
        print(f"{k:36s} {v}")


if __name__ == "__main__":
    main()
