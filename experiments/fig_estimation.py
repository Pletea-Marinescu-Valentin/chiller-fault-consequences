"""Fig.: estimated against true weekly energy penalty, with a plant meter and a virtual meter."""

import json

import numpy as np
import pandas as pd

from _common import (AQUA, BLUE, INK_MUTED, ORANGE, RESULTS, TABLES, figure, load_config,
                     save_figure, write_numbers, write_results)
from tab_estimation import ESTIMATES

NAME = "fig_estimation"
STYLE = {"chiller": ("Chiller sensor and fouling", BLUE, "o"),
         "bypass": ("Bypass valve", ORANGE, "s"),
         "other": ("Other faults", AQUA, "^")}


def main():
    cfg = load_config(NAME)
    est = pd.read_csv(TABLES / f"{ESTIMATES}.csv")
    group = pd.Series("other", index=est.index)
    for name, faults in cfg["groups"].items():
        group[est.fault.isin(faults)] = name
    cf = json.loads((RESULTS / "tab_estimation.json").read_text())["counterfactual_error_kwh"]
    threshold = np.quantile(np.abs(cf["metered"]), cfg["resolvable_quantile"])
    resolvable = est.d_energy.abs() > threshold
    agree = {m: float((np.sign(est[f"e_{m}"]) == np.sign(est.d_energy))[resolvable].mean())
             for m in ("metered", "unmetered")}

    true = est.d_energy / 1000
    lim = np.array([min(true.min(), (est[["e_metered", "e_unmetered"]] / 1000).min().min()),
                    max(true.max(), (est[["e_metered", "e_unmetered"]] / 1000).max().max())])
    lim = lim + np.array([-1, 1]) * 0.04 * np.ptp(lim)
    fig, axes = figure(height=2.1, ncols=2, sharex=True, sharey=True,
                       gridspec_kw={"wspace": 0.08})
    for ax, col, title in zip(axes, ("e_metered", "e_unmetered"),
                              ("(a) Plant meter", "(b) Virtual meter")):
        ax.plot(lim, lim, color=INK_MUTED, linestyle="--", linewidth=0.6, zorder=1)
        for key, (label, color, marker) in STYLE.items():
            sel = group == key
            ax.scatter(true[sel], est.loc[sel, col] / 1000, s=5, color=color, marker=marker,
                       alpha=0.55, linewidths=0, label=label, zorder=2)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=7.5)
        ax.set_xlabel("True penalty (MWh per week)")
    axes[0].set_ylabel("Estimated penalty (MWh per week)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.1), ncols=3,
               markerscale=2.0, handletextpad=0.2, columnspacing=0.9)
    save_figure(fig, NAME)

    write_results(NAME, {"config": cfg, "threshold_kwh": threshold, "sign_agreement": agree,
                         "resolvable": int(resolvable.sum())})
    numbers = {"resolvable": f"{int(resolvable.sum())}",
               "sign agree metered": f"{100 * agree['metered']:.0f}",
               "sign agree unmetered": f"{100 * agree['unmetered']:.0f}"}
    write_numbers(NAME, numbers)
    print(numbers)


if __name__ == "__main__":
    main()
