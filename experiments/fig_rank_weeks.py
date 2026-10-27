"""Fig.: weekly repair-priority rank of every fault under the true priced consequence."""

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from _common import INK_SECONDARY, TABLES, figure, load_config, save_figure, write_numbers
from cfc.consequences import cost
from cfc.datasets.lbnl_chiller import CASES
from fig_consequences import SHORT, intensity_text, ordered_cases

NAME = "fig_rank_weeks"


def main():
    cfg = load_config(NAME)
    ref = load_config("fig_consequences")
    truth = pd.read_csv(TABLES / "consequences_weekly.csv")
    truth["cost"] = cost(truth.d_energy, truth.d_service, ref["kappa"])
    wide = truth.pivot(index="case", columns="week", values="cost")
    rank = wide.rank(axis=0, ascending=False, method="average")  # 1 = largest consequence
    cases = ordered_cases(ref["order"])
    rank = rank.loc[cases]

    labels = [f"{SHORT[CASES[c].fault]} {intensity_text(c, latex=False)}".replace(" --", "")
              for c in cases]
    cmap = LinearSegmentedColormap.from_list("ramp", cfg["ramp"][::-1])  # dark = repair first
    fig, ax = figure(height=3.5)
    fig.subplots_adjust(left=0.37, right=0.86, bottom=0.11, top=0.99)  # labels inside the column
    im = ax.imshow(rank.to_numpy(), aspect="auto", cmap=cmap, vmin=1, vmax=len(cases),
                   interpolation="nearest", extent=(0.5, rank.shape[1] + 0.5, len(cases) - 0.5, -0.5))
    ax.set_yticks(range(len(cases)), labels, fontsize=7)
    ax.tick_params(axis="y", length=0)
    ax.set_xticks([1, 10, 20, 30, 40, 52])
    ax.set_xlabel("Week of the year")
    ax.grid(False)
    groups = pd.Series([CASES[c].fault for c in cases]).to_numpy()
    for b in np.flatnonzero(groups[1:] != groups[:-1]):
        ax.axhline(b + 0.5, color="white", linewidth=0.8)
    bar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02, aspect=45, ticks=[1, 6, 12, 18, 23])
    bar.set_label("Priority rank (1: repair first)", fontsize=7)
    bar.ax.tick_params(labelsize=7, length=2, color=INK_SECONDARY)
    bar.outline.set_linewidth(0.4)
    bar.ax.invert_yaxis()
    save_figure(fig, NAME)

    spread = rank.max(axis=1) - rank.min(axis=1)
    first = rank.idxmin(axis=0)
    numbers = {"faults ranked first": f"{first.nunique()}",
               "median rank range": f"{spread.median():.0f}",
               "kappa": f"{ref['kappa']}"}
    write_numbers(NAME, numbers)
    print(numbers, first.value_counts().to_dict())


if __name__ == "__main__":
    main()
