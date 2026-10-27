"""Fig.: fault-free weekly plant energy and the weekly energy penalty of selected faults."""

import numpy as np

from _common import (AQUA, BLUE, GRID, INK, INK_MUTED, ORANGE, YELLOW, figure, load_config,
                     save_figure, write_numbers, write_results)
from cfc.consequences import weekly_truth
from cfc.context import N_WEEKS, weekly_mean, weekly_sum
from cfc.datasets.lbnl_chiller import load_case
from fig_consequences import SHORT, intensity_text

NAME = "fig_weekly"
STYLES = [(BLUE, "-", "o"), (ORANGE, "--", "s"), (AQUA, "-.", "^"), (YELLOW, ":", "D")]


def main():
    cfg = load_config(NAME)
    ff = load_case("fault_free")
    energy = weekly_sum(ff.PLANT_POW, ff.index) / 1000
    cooling = weekly_sum(ff.CWL_SEC_LOAD, ff.index) / 1000
    flow = weekly_mean(ff.CWL_SEC_CW_FLOW, ff.index)
    idle = flow < cfg["no_flow_lps"]
    truth = weekly_truth(cfg["cases"])
    weeks = np.arange(N_WEEKS) + 1

    fig, (ax1, ax2) = figure(height=3.3, nrows=2, sharex=True,
                             gridspec_kw={"hspace": 0.12, "height_ratios": [1, 1.4]})
    for ax in (ax1, ax2):
        for w in weeks[idle]:
            ax.axvspan(w - 0.5, w + 0.5, color=GRID, linewidth=0, zorder=0)
    ax1.plot(weeks, energy, color=INK, marker="o", markersize=2.5, label="Plant electricity")
    ax1.plot(weeks, cooling, color=INK_MUTED, linestyle="--", label="Cooling delivered")
    ax1.set_ylabel("Fault-free (MWh)")
    ax1.set_ylim(0, 1.3 * max(energy.max(), cooling.max()))
    ax1.legend(loc="upper left", ncols=2)
    ax1.text(weeks[idle].mean(), 0.45 * ax1.get_ylim()[1], "idle\nloop", ha="center",
             va="center", fontsize=6.5, color=INK_MUTED)
    for case, (color, ls, marker) in zip(cfg["cases"], STYLES):
        d = truth[truth.case == case].sort_values("week")
        fault = case.rsplit("_", 1)[0]
        label = f"{SHORT[fault]} {intensity_text(case, latex=False)}"
        ax2.plot(weeks, d.d_energy / 1000, color=color, linestyle=ls, marker=marker,
                 markersize=2.5, label=label)
    ax2.axhline(0, color=INK_MUTED, linewidth=0.5)
    ax2.set_ylabel("Energy penalty $\\Delta E$ (MWh)")
    ax2.set_xlabel("Week of the year")
    ax2.set_xlim(0.5, N_WEEKS + 0.5)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncols=2, columnspacing=1.0)
    save_figure(fig, NAME)

    write_results(NAME, {"config": cfg, "energy_mwh": energy.tolist(),
                         "cooling_mwh": cooling.tolist(), "idle": idle.tolist()})
    peak = int(np.argmax(cooling))
    numbers = {"peak cooling week": f"{peak + 1}", "peak cooling": f"{cooling[peak]:.0f}",
               "peak energy": f"{energy.max():.0f}", "min energy": f"{energy.min():.1f}"}
    write_numbers(NAME, numbers)
    print(numbers)


if __name__ == "__main__":
    main()
