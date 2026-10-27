"""Table: replication on the simulated rooftop unit, from ground truth to repair priority."""

import time

import numpy as np
import pandas as pd

from _common import (GENERATED, TABLES, fmt_float, fmt_int, load_config, write_numbers,
                     write_results)
from cfc.consequences import cost, weekly_truth
from cfc.context import folds, n_weeks, weekly_mean, weekly_sum
from cfc.datasets import lbnl_rtu
from cfc.estimators import AnomalyScore, counterfactual, penalty
from cfc.virtual_meter import RTU_COMPONENTS, VirtualMeter
from fig_ranking import METHODS, PROXIES, evaluate
from scipy.stats import kendalltau

NAME = "tab_rtu"
MULTIPLE_WORD = {0: "zero", 1: "ref", 10: "high"}


def main():
    cfg = load_config(NAME)
    t0 = time.time()
    ff = lbnl_rtu.load_case("fault_free")
    frames = {c: lbnl_rtu.load_case(c) for c in lbnl_rtu.FAULTY}
    fold = folds(ff.index, cfg["n_folds"])
    n, seed, weeks = cfg["n_folds"], cfg["seed"], n_weeks(ff.index)
    weather_diff = max(float((d.OA_TEMP - ff.OA_TEMP).abs().max()) for d in frames.values())

    truth = weekly_truth(lbnl_rtu.FAULTY, lbnl_rtu.load_case, "fault_free",
                         lbnl_rtu.COMPONENT_POWER)
    base = counterfactual(ff, ff.PLANT_POW.to_numpy(), fold, cfg["counterfactual"], n, seed)
    cf_metered = penalty(ff.PLANT_POW.to_numpy(), base, ff, fold)
    base_x = counterfactual(ff, ff.SUP_EXCESS.to_numpy(), fold, cfg["counterfactual"], n, seed)
    cf_service = penalty(ff.SUP_EXCESS.to_numpy(), base_x, ff, fold)
    vm = VirtualMeter(cfg["virtual_meter"], n, seed, components=RTU_COMPONENTS).fit(ff, fold)
    v0 = vm.predict(ff, fold)
    base_v = counterfactual(ff, v0, fold, cfg["counterfactual"], n, seed)
    cf_unmetered = penalty(v0, base_v, ff, fold)
    vm_error_ff = weekly_sum(v0 - ff.PLANT_POW.to_numpy(), ff.index)
    anomaly = AnomalyScore(cfg["anomaly_points"], cfg["anomaly"], n, seed).fit(ff, fold)

    rows = []
    for c, d in frames.items():
        virtual = vm.predict(d, fold)
        rows.append(pd.DataFrame({
            "case": c, "fault": lbnl_rtu.CASES[c].fault, "level": lbnl_rtu.CASES[c].level,
            "week": np.arange(weeks),
            "d_energy": weekly_sum(d.PLANT_POW - ff.PLANT_POW, d.index),
            "d_service": weekly_sum(d.SUP_EXCESS - ff.SUP_EXCESS, d.index),
            "e_metered": penalty(d.PLANT_POW.to_numpy(), base, d, fold),
            "e_unmetered": penalty(virtual, base_v, d, fold),
            "s_hat": penalty(d.SUP_EXCESS.to_numpy(), base_x, d, fold),
            "anomaly": anomaly.weekly(d, fold),
            "vm_error": weekly_sum(virtual - d.PLANT_POW.to_numpy(), d.index),
            "outside": weekly_mean(vm.outside(d, cfg["envelope_margin"]), d.index)}))
    est = pd.concat(rows, ignore_index=True)
    assert np.allclose(est.d_energy, truth.d_energy) and np.allclose(est.d_service, truth.d_service)
    TABLES.mkdir(parents=True, exist_ok=True)
    est.to_csv(TABLES / "rtu_estimates_weekly.csv", index=False, float_format="%.4f")
    print(f"estimates done in {time.time() - t0:.0f} s")

    # kappa anchor: electricity to remove 1 K h from the mean supply air stream at nominal COP
    m_cp = ff.RTU_SA_FLOW.mean() / 1000 * 1.2 * 1.006  # kW/K
    anchor = m_cp / cfg["nominal_cop"]
    multiples = cfg["kappa_multiples"]
    res = evaluate(est, [m * anchor for m in multiples], cfg["portfolio_size"], cfg["capacity"],
                   cfg["portfolios"], seed, cfg["envelope_week"], cases=lbnl_rtu.CASES)
    res["multiple"] = res.kappa.map(dict(zip([m * anchor for m in multiples], multiples)))
    res.to_csv(TABLES / f"{NAME}.csv", index=False, float_format="%.4f")

    annual = truth.groupby("case")[["d_energy", "d_service"]].sum()
    annual["level"] = [lbnl_rtu.CASES[c].level for c in annual.index]
    ref_cost = pd.Series(cost(truth.d_energy, truth.d_service, anchor)).groupby(truth.case).sum()
    tau_level_energy = kendalltau(annual.level, annual.d_energy).statistic
    tau_level_cost = kendalltau(annual.level, ref_cost.loc[annual.index]).statistic
    est["err_m"] = est.e_metered - est.d_energy
    est["err_u"] = est.e_unmetered - est.d_energy
    by_type = est.groupby("fault").agg(true_abs=("d_energy", lambda s: s.abs().mean()),
                                       mae_u=("err_u", lambda s: s.abs().mean()),
                                       vm_bias=("vm_error", "mean"),
                                       out=("outside", lambda s: (s > cfg["envelope_week"]).mean()))
    ff_week = weekly_sum(ff.PLANT_POW, ff.index)
    write_results(NAME, {"config": cfg, "anchor": anchor, "weeks": weeks,
                         "weather_max_diff_k": weather_diff, "annual": annual.reset_index().to_dict("records"),
                         "by_type": by_type.reset_index().to_dict("records"),
                         "counterfactual_error_kwh": {"metered": cf_metered.tolist(),
                                                      "unmetered": cf_unmetered.tolist()},
                         "results": res.to_dict("records")})
    print(annual.round(1).to_string())
    print(by_type.round(2).to_string())
    print(res[res.multiple.isin(MULTIPLE_WORD)].round(3).to_string(index=False))

    numbers = {
        "weeks": f"{weeks}", "faulty cases": f"{len(lbnl_rtu.FAULTY)}",
        "fault types": f"{len(lbnl_rtu.FAULT_TYPES)}", "folds": f"{n}",
        "weather max diff": f"{weather_diff:.3f}",
        "ff energy": fmt_int(ff_week.sum()),
        "ff excess": fmt_int(weekly_sum(ff.SUP_EXCESS, ff.index).sum()),
        "ff mean excess": fmt_float(ff.SUP_EXCESS.mean(), 1),
        "anchor": fmt_float(anchor, 2),
        "max penalty pct": f"{100 * annual.d_energy.max() / ff_week.sum():.0f}",
        "max saving pct": f"{-100 * annual.d_energy.min() / ff_week.sum():.0f}",
        "negative energy cases": f"{int((annual.d_energy < 0).sum())}",
        "negative service cases": f"{int((annual.d_service < 0).sum())}",
        "tau level energy": fmt_float(tau_level_energy, 2),
        "tau level cost": fmt_float(tau_level_cost, 2),
        "cf metered pct": f"{100 * np.abs(cf_metered).mean() / ff_week.mean():.0f}",
        "cf unmetered pct": f"{100 * np.abs(cf_unmetered).mean() / ff_week.mean():.0f}",
        "vm ff pct": fmt_float(100 * np.abs(vm_error_ff).mean() / ff_week.mean()),
        "service cf mae": fmt_float(np.abs(cf_service).mean()),
    }
    for m_val, word in MULTIPLE_WORD.items():
        sub = res[res.multiple == m_val].set_index("method")
        for m in (*METHODS, "unmetered_abstain"):
            numbers[f"value {m.replace('_', ' ')} {word}"] = fmt_float(sub.loc[m, "value"], 2)
        numbers[f"best proxy value {word}"] = fmt_float(max(sub.loc[m, "value"] for m in PROXIES), 2)
    write_numbers(NAME, numbers)
    for k, v in numbers.items():
        print(f"{k:34s} {v}")


if __name__ == "__main__":
    main()
