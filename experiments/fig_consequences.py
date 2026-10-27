"""Fig. and Table: annual energy penalty and service deficit of every fault (ground truth)."""

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from matplotlib.ticker import FuncFormatter

from _common import (BLUE, GENERATED, INK_SECONDARY, ORANGE, TABLES, TEXT_WIDTH, figure,
                     fmt_float, fmt_int, load_config, save_figure, write_numbers, write_results)
from cfc.consequences import cost, weekly_truth
from cfc.context import weekly_mean, weekly_sum
from cfc.datasets.lbnl_chiller import CASES, COMPONENT_POWER, FAULT_TYPES, load_case

NAME = "fig_consequences"

SHORT = {"chiller_bias": "Chiller sensor", "dp_bias": "DP sensor", "ct_bias": "Tower sensor",
         "ct_fouling": "Tower fouling", "chiller_fouling": "Chiller fouling",
         "ct_pi": "Tower PI tuning", "valve_leak": "Bypass leak", "valve_stuck": "Bypass stuck"}


def intensity_text(case: str, latex: bool = True) -> str:
    c = CASES[case]
    unit = FAULT_TYPES[c.fault].unit
    if c.fault == "ct_pi":
        return "--"
    if c.fault in ("ct_bias", "chiller_bias", "dp_bias"):
        sign = ("$-$" if latex else "−") if c.intensity < 0 else "+"
        value = f"{sign}{abs(c.intensity):g}"
    else:
        value = f"{c.intensity:g}"
    unit = {"%": "\\%" if latex else "%", "% UA": "\\% UA" if latex else "% UA"}.get(unit, unit)
    return f"{value} {unit}"


def ordered_cases(order: list[str]) -> list[str]:
    def key(case):
        c = CASES[case]
        return (order.index(c.fault), c.intensity if "bias" in c.fault else c.level)
    return sorted([c for c in CASES if CASES[c].fault is not None], key=key)


def main():
    cfg = load_config(NAME)
    truth = weekly_truth()
    ff = load_case("fault_free")
    ff_energy = weekly_sum(ff.PLANT_POW, ff.index).sum() / 1000
    ff_cooling = weekly_sum(ff.CWL_SEC_LOAD, ff.index).sum() / 1000
    ff_comp = {k: weekly_sum(ff[v].sum(axis=1), ff.index).sum() / 1000
               for k, v in COMPONENT_POWER.items()}
    # Physical anchor for kappa: electricity to remove 1 K h of excess at the mean
    # secondary flow of cooling hours and the annual chiller COP.
    cooling_hours = ff.CWL_SEC_LOAD > 1.0
    m_cp = (ff.CWL_SEC_CW_FLOW[cooling_hours].mean() * 0.998 * 4.186)  # kW/K
    cop = ff_cooling / ff_comp["chillers"]
    kappa_anchor = m_cp / cop

    truth["cost"] = cost(truth.d_energy, truth.d_service, cfg["kappa"])
    annual = truth.groupby("case")[["d_energy", "d_service", "cost"]].sum()
    annual["weeks_penalty"] = truth.groupby("case").d_energy.apply(lambda s: int((s > 0).sum()))
    annual["level"] = [CASES[c].level for c in annual.index]
    cases = ordered_cases(cfg["order"])
    annual = annual.loc[cases]

    # Rank agreement of documented intensity with the consequence, across all faults
    tau_level_energy = kendalltau(annual.level, annual.d_energy).statistic
    tau_level_cost = kendalltau(annual.level, annual.cost).statistic
    # Stability of the weekly priority order through the year
    taus = {}
    for col in ("d_energy", "cost"):
        wide = truth.pivot(index="week", columns="case", values=col)
        t = [kendalltau(wide.iloc[i], wide.iloc[j]).statistic
             for i in range(len(wide)) for j in range(i + 1, len(wide))]
        taus[col] = np.array(t)
    # Where the chiller-sensor penalty comes from: weeks without cooling load
    no_load = weekly_sum(ff.CWL_SEC_LOAD, ff.index) < cfg["no_load_mwh"] * 1000
    share_no_load = {}
    for case in ("chiller_bias_-1", "chiller_bias_-2"):
        e = truth[truth.case == case].d_energy.to_numpy()
        share_no_load[case] = e[no_load].sum() / e.sum()
    # The chiller sensor bias decides the regime of the secondary loop in weeks without load
    winter_flow = {case: float(weekly_mean(load_case(case, ["CWL_SEC_CW_FLOW"]).CWL_SEC_CW_FLOW,
                                           ff.index)[no_load].mean())
                   for case in ("fault_free", "chiller_bias_-2", "chiller_bias_+2")}

    payload = {"config": cfg, "fault_free": {"energy_mwh": ff_energy, "cooling_mwh": ff_cooling,
                                             "components_mwh": ff_comp},
               "kappa_anchor": kappa_anchor, "annual": annual.reset_index().to_dict("records"),
               "tau_level_energy": tau_level_energy, "tau_level_cost": tau_level_cost,
               "weekly_tau_median": {k: float(np.median(v)) for k, v in taus.items()},
               "share_no_load": share_no_load, "winter_flow_lps": winter_flow}
    write_results(NAME, payload)
    TABLES.mkdir(parents=True, exist_ok=True)
    truth.to_csv(TABLES / "consequences_weekly.csv", index=False, float_format="%.3f")
    annual.to_csv(TABLES / f"{NAME}.csv", float_format="%.3f")
    print(annual.round(1).to_string())

    # LaTeX table body
    lines = ["\\begin{tabular}{llrrrr}", "\\hline",
             "Fault & Intensity & $\\Delta E$ & $\\Delta E$ & $\\Delta S$ & Weeks \\\\",
             " & & [MWh] & [\\%] & [K\\,h] & $\\Delta E>0$ \\\\", "\\hline"]
    last = None
    for case, r in annual.iterrows():
        fault = CASES[case].fault
        name = SHORT[fault] + ("$^\\ast$" if not FAULT_TYPES[fault].documented else "")
        lines.append(f"{name if fault != last else ''} & {intensity_text(case)} & "
                     f"{fmt_int(r.d_energy / 1000)} & "
                     f"{fmt_float(100 * r.d_energy / 1000 / ff_energy)} & "
                     f"{fmt_int(r.d_service)} & {int(r.weeks_penalty)} \\\\")
        last = fault
    lines += ["\\hline", "\\end{tabular}"]
    GENERATED.mkdir(parents=True, exist_ok=True)
    (GENERATED / f"{NAME}_table.tex").write_text("\n".join(lines) + "\n")

    # Figure: two bar panels sharing the fault axis
    labels = [f"{SHORT[CASES[c].fault]} {intensity_text(c, latex=False)}".replace(" --", "")
              for c in cases]
    y = np.arange(len(cases))[::-1]
    fig, (ax1, ax2) = figure(TEXT_WIDTH, 2.9, ncols=2, sharey=True,
                             gridspec_kw={"wspace": 0.06})
    ax1.barh(y, annual.d_energy / 1000, height=0.7, color=BLUE, linewidth=0)
    ax2.barh(y, annual.d_service, height=0.7, color=ORANGE, linewidth=0)
    ax2.set_xscale("symlog", linthresh=100)
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}".replace("-", "−")))
    for ax in (ax1, ax2):
        ax.axvline(0, color=INK_SECONDARY, linewidth=0.6)
        ax.grid(axis="y", visible=False)
        ax.set_ylim(-0.6, len(cases) - 0.4)
    ax1.set_yticks(y, labels)
    ax1.tick_params(axis="y", length=0)
    ax1.set_xlabel("(a) Energy penalty $\\Delta E$ (MWh per year)")
    ax2.set_xlabel("(b) Service deficit $\\Delta S$ (K h per year, symmetric log)")
    groups = pd.Series([CASES[c].fault for c in cases])
    for b in np.flatnonzero(groups.to_numpy()[1:] != groups.to_numpy()[:-1]):
        for ax in (ax1, ax2):
            ax.axhline(y[b] - 0.5, color="#c3c2b7", linewidth=0.4)
    save_figure(fig, NAME)

    numbers = {
        "ff energy": fmt_int(ff_energy),
        "ff cooling": fmt_int(ff_cooling),
        "ff chillers pct": f"{100 * ff_comp['chillers'] / ff_energy:.0f}",
        "ff pumps pct": f"{100 * (ff_comp['condenser pumps'] + ff_comp['primary pumps'] + ff_comp['secondary pumps']) / ff_energy:.0f}",
        "ff secondary pct": f"{100 * ff_comp['secondary pumps'] / ff_energy:.0f}",
        "ff towers pct": f"{100 * ff_comp['towers'] / ff_energy:.1f}",
        "ff plant cop": f"{ff_cooling / ff_energy:.2f}",
        "chiller cop": f"{cop:.1f}",
        "kappa": f"{cfg['kappa']}",
        "kappa anchor": f"{kappa_anchor:.0f}",
        "max penalty": fmt_int(annual.d_energy.max() / 1000),
        "max penalty pct": f"{100 * annual.d_energy.max() / 1000 / ff_energy:.0f}",
        "max saving": fmt_int(-annual.d_energy.min() / 1000),
        "max saving pct": f"{-100 * annual.d_energy.min() / 1000 / ff_energy:.0f}",
        "min abs penalty": f"{annual.d_energy.abs().min() / 1000:.1f}",
        "negative energy cases": f"{int((annual.d_energy < 0).sum())}",
        "saving with deficit cases": f"{int(((annual.d_energy < 0) & (annual.d_service > 0)).sum())}",
        "tau level energy": fmt_float(tau_level_energy, 2),
        "tau level cost": fmt_float(tau_level_cost, 2),
        "weekly tau energy median": f"{np.median(taus['d_energy']):.2f}",
        "weekly tau energy low": fmt_float(np.quantile(taus["d_energy"], 0.1), 2),
        "weekly tau cost median": f"{np.median(taus['cost']):.2f}",
        "weekly tau cost low": fmt_float(np.quantile(taus["cost"], 0.1), 2),
        "no load share chiller one": f"{100 * share_no_load['chiller_bias_-1']:.0f}",
        "no load share chiller two": f"{100 * share_no_load['chiller_bias_-2']:.0f}",
        "no load weeks": f"{int(no_load.sum())}",
        "winter flow ff": f"{winter_flow['fault_free']:.0f}",
        "winter flow chiller minus two": f"{winter_flow['chiller_bias_-2']:.0f}",
        "winter flow chiller plus two": f"{winter_flow['chiller_bias_+2']:.1f}",
    }
    pick = {"chiller minus one": "chiller_bias_-1", "chiller minus two": "chiller_bias_-2",
            "chiller plus one": "chiller_bias_+1", "chiller plus two": "chiller_bias_+2",
            "dp minus twenty": "dp_bias_-20", "dp minus ten": "dp_bias_-10",
            "dp plus ten": "dp_bias_+10", "dp plus twenty": "dp_bias_+20",
            "tower fouling mild": "ct_fouling_95", "tower fouling severe": "ct_fouling_65",
            "tower bias minus two": "ct_bias_-2", "tower bias plus two": "ct_bias_+2",
            "leak small": "valve_leak_25", "leak large": "valve_leak_75",
            "stuck large": "valve_stuck_75", "chiller fouling severe": "chiller_fouling_65",
            "pi": "ct_pi"}
    for key, case in pick.items():
        r = annual.loc[case]
        numbers[f"{key} energy"] = fmt_int(r.d_energy / 1000)
        numbers[f"{key} service"] = fmt_int(r.d_service)
    write_numbers(NAME, numbers)
    for k, v in numbers.items():
        print(f"{k:34s} {v}")


if __name__ == "__main__":
    main()
