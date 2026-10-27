"""Fig. and Table: repair priority by estimated consequence versus priority proxies."""

import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, NullFormatter

from _common import (AQUA, BLUE, GENERATED, GREEN, INK_MUTED, MAGENTA, ORANGE, TABLES, YELLOW,
                     figure, fmt_float, fmt_int, load_config, save_figure, write_numbers,
                     write_results)
from cfc.consequences import cost
from cfc.datasets.lbnl_chiller import CASES
from cfc.proxies import intensity_level, type_priority
from cfc.ranking import decision_value, kendall, sample_portfolios, within_group_kendall
from tab_estimation import ESTIMATES

NAME = "fig_ranking"
METHODS = {
    "metered": ("Estimate, plant meter", BLUE, "-", "o"),
    "unmetered": ("Estimate, virtual meter", ORANGE, "--", "s"),
    "service": ("Service deficit only", AQUA, (0, (1, 1)), "x"),
    "anomaly": ("Anomaly magnitude", YELLOW, "-.", "^"),
    "type": ("Fault-type priority", MAGENTA, ":", "D"),
    "intensity": ("Documented intensity", GREEN, (0, (5, 1, 1, 1)), "v"),
}
PROXIES = ("anomaly", "type", "intensity")
KAPPA_WORD = {0: "zero", 10: "low", 70: "ref", 700: "high"}


def scores(est: pd.DataFrame, kappa: float, envelope_week: float, cases: dict = CASES
           ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    truth = cost(est.d_energy, est.d_service, kappa)
    annual = pd.Series(truth, index=est.index).groupby(est.case).sum()
    abstain = est.e_unmetered.where(est.outside <= envelope_week, 0.0)
    return truth, {
        "metered": cost(est.e_metered, est.s_hat, kappa),
        "unmetered": cost(est.e_unmetered, est.s_hat, kappa),
        "unmetered_abstain": cost(abstain, est.s_hat, kappa),
        "service": cost(np.zeros(len(est)), est.s_hat, kappa),
        "anomaly": est.anomaly.to_numpy(),
        "type": est.case.map(type_priority(annual, cases)).to_numpy(),
        "intensity": intensity_level(est.case, cases).to_numpy(),
    }


def evaluate(est: pd.DataFrame, kappas, size: int, capacity: int, n: int, seed: int,
             envelope_week: float, same_week: bool = True, cases: dict = CASES) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    case, week = est.case.to_numpy(), est.week.to_numpy()
    mixed = sample_portfolios(case, week, size, n, rng)
    same = sample_portfolios(case, week, size, n, rng, same_week=True) if same_week else None
    rows = []
    for kappa in kappas:
        truth, s = scores(est, kappa, envelope_week, cases)
        for m, score in s.items():
            tie = np.random.default_rng(seed + 1)
            dv = decision_value(score, truth, mixed, capacity, tie)
            row = {"kappa": kappa, "method": m, "value": dv["value"], "regret": dv["regret"]}
            if same_week:
                row.update({"tau": kendall(score, truth),
                            "tau_week": within_group_kendall(score, truth, week),
                            "value_same": decision_value(score, truth, same, capacity, tie)["value"]})
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    cfg = load_config(NAME)
    est = pd.read_csv(TABLES / f"{ESTIMATES}.csv")
    size, cap, n, seed = cfg["portfolio_size"], cfg["capacity"], cfg["portfolios"], cfg["seed"]
    env = load_config("tab_estimation")["envelope_week"]
    res = evaluate(est, cfg["kappas"], size, cap, n, seed, env)
    ref = cfg["reference_kappas"]
    # Robustness: leave one fault type out, and other portfolio sizes and capacities
    loto = []
    for fault in sorted(est.fault.unique()):
        r = evaluate(est[est.fault != fault].reset_index(drop=True), ref, size, cap, n, seed,
                     env, same_week=False)
        loto.append(r.assign(excluded=fault))
    loto = pd.concat(loto, ignore_index=True)
    variants = pd.concat([evaluate(est, ref, s, c, n, seed, env, same_week=False)
                          .assign(size=s, capacity=c) for s, c in cfg["variants"]],
                         ignore_index=True)
    write_results(NAME, {"config": cfg, "results": res.to_dict("records"),
                         "leave_one_type_out": loto.to_dict("records"),
                         "variants": variants.to_dict("records")})
    res.to_csv(TABLES / f"{NAME}.csv", index=False, float_format="%.4f")
    print(res[res.kappa.isin(ref)].round(3).to_string(index=False))
    span = loto.groupby(["kappa", "method"]).value.agg(["min", "max"])
    print(span.round(3).to_string())
    print(variants.round(3).to_string(index=False))

    # Table for the reference prices
    lines = ["\\begin{tabular}{llrrrr}", "\\hline",
             "$\\kappa$ & Priority rule & $\\tau$ & $\\bar\\tau_{\\mathrm{week}}$ & "
             "$V_{\\mathrm{fleet}}$ & $V_{\\mathrm{same}}$ \\\\", "\\hline"]
    for kappa in cfg["table_kappas"]:
        sub = res[res.kappa == kappa].set_index("method")
        for i, m in enumerate(METHODS):
            r = sub.loc[m]
            lines.append(f"{kappa if i == 0 else ''} & {METHODS[m][0]} & {fmt_float(r.tau, 2)} & "
                         f"{fmt_float(r.tau_week, 2)} & {fmt_float(r.value, 2)} & "
                         f"{fmt_float(r.value_same, 2)} \\\\")
        lines.append("\\hline")
    lines.append("\\end{tabular}")
    GENERATED.mkdir(parents=True, exist_ok=True)
    (GENERATED / f"{NAME}_table.tex").write_text("\n".join(lines) + "\n")

    # Figure: decision value of each rule against the service price, fleet portfolios
    fig, ax = figure(height=2.3)
    for m, (label, color, ls, marker) in METHODS.items():
        sub = res[res.method == m].sort_values("kappa")
        ax.plot(sub.kappa.to_numpy(dtype=float), sub.value, color=color, linestyle=ls,
                marker=marker, label=label)
    ax.axhline(0, color=INK_MUTED, linewidth=0.5)
    ax.axhline(1, color=INK_MUTED, linewidth=0.5, linestyle=":")
    ax.set_xscale("symlog", linthresh=1, linscale=0.5)
    ax.set_xlim(-0.15, 1.15 * max(cfg["kappas"]))
    ticks = [0, 1, 10, 70, 700]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.set_xticklabels([str(t) for t in ticks])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Price of the service deficit $\\kappa$ (kWh per K h)")
    ax.set_ylabel("Decision value")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncols=2, columnspacing=1.0)
    save_figure(fig, NAME)

    numbers = {"portfolio size": f"{size}", "capacity": f"{cap}", "portfolios": fmt_int(n)}
    for kappa in ref:
        k = KAPPA_WORD[kappa]
        sub = res[res.kappa == kappa].set_index("method")
        for m in METHODS:
            numbers[f"value {m} {k}"] = fmt_float(sub.loc[m, "value"], 2)
            numbers[f"value same {m} {k}"] = fmt_float(sub.loc[m, "value_same"], 2)
            numbers[f"tau {m} {k}"] = fmt_float(sub.loc[m, "tau"], 2)
            numbers[f"tau week {m} {k}"] = fmt_float(sub.loc[m, "tau_week"], 2)
            numbers[f"regret {m} {k}"] = fmt_int(sub.loc[m, "regret"])
        best = max(PROXIES, key=lambda m: sub.loc[m, "value"])
        numbers[f"best proxy value {k}"] = fmt_float(sub.loc[best, "value"], 2)
        numbers[f"best proxy {k}"] = METHODS[best][0].lower()
        numbers[f"value abstain {k}"] = fmt_float(sub.loc["unmetered_abstain", "value"], 2)
        numbers[f"value same abstain {k}"] = fmt_float(sub.loc["unmetered_abstain", "value_same"], 2)
        for m in ("metered", "unmetered", "service"):
            numbers[f"loto min {m} {k}"] = fmt_float(span.loc[(kappa, m), "min"], 2)
            numbers[f"loto max {m} {k}"] = fmt_float(span.loc[(kappa, m), "max"], 2)
        numbers[f"loto max proxy {k}"] = fmt_float(
            max(span.loc[(kappa, m), "max"] for m in PROXIES), 2)
    for (s, c), g in variants.groupby(["size", "capacity"]):
        words = {5: "five", 10: "ten", 1: "one", 2: "two", 3: "three"}
        for kappa in ref:
            sub = g[g.kappa == kappa].set_index("method")
            for m in ("metered", "unmetered", "service"):
                numbers[f"variant {words[s]} {words[c]} {m} {KAPPA_WORD[kappa]}"] = \
                    fmt_float(sub.loc[m, "value"], 2)
            numbers[f"variant {words[s]} {words[c]} best proxy {KAPPA_WORD[kappa]}"] = \
                fmt_float(max(sub.loc[m, "value"] for m in PROXIES), 2)
    write_numbers(NAME, numbers)
    for k, v in numbers.items():
        print(f"{k:40s} {v}")


if __name__ == "__main__":
    main()
