"""Repair priority: rank agreement and the decision value of a ranking.

A portfolio is a set of diagnosed faults, each in its own plant and week. A
planner who can repair `capacity` of them repairs the top-scored ones and
avoids their consequences. The decision value of a scoring rule is the share
of the improvement over a random choice that the oracle (true consequence)
achieves, summed over many random portfolios:

    value = sum(V_rule - V_random) / sum(V_oracle - V_random),

1 for a perfect ranking, 0 for random choice, negative for a ranking worse than
random. Ties in a score are broken at random.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau


def kendall(score: np.ndarray, truth: np.ndarray) -> float:
    """Kendall tau-b; NaN when the score is constant (it ranks nothing)."""
    if np.ptp(score) == 0:
        return float("nan")
    return float(kendalltau(score, truth).statistic)


def within_group_kendall(score: np.ndarray, truth: np.ndarray, group: np.ndarray) -> float:
    """Mean Kendall tau-b over groups, e.g. over weeks of identical plants."""
    taus = []
    for g in np.unique(group):
        sel = group == g
        if np.ptp(truth[sel]) > 0 and np.ptp(score[sel]) > 0:
            taus.append(kendall(score[sel], truth[sel]))
    return float(np.mean(taus)) if taus else float("nan")


def sample_portfolios(case: np.ndarray, week: np.ndarray, size: int, n: int,
                      rng: np.random.Generator, same_week: bool = False) -> np.ndarray:
    """Row indices (n, size) of portfolios with distinct cases and, unless same_week,
    distinct weeks; with same_week all members share one week (identical plants)."""
    cases, weeks = np.unique(case), np.unique(week)
    lookup = {(c, w): i for i, (c, w) in enumerate(zip(case, week))}
    pick_c = np.argsort(rng.random((n, len(cases))), axis=1)[:, :size]
    if same_week:
        pick_w = np.repeat(rng.integers(len(weeks), size=n)[:, None], size, axis=1)
    else:
        pick_w = np.argsort(rng.random((n, len(weeks))), axis=1)[:, :size]
    return np.vectorize(lambda c, w: lookup[(cases[c], weeks[w])])(pick_c, pick_w)


def decision_value(score: np.ndarray, truth: np.ndarray, portfolios: np.ndarray, capacity: int,
                   rng: np.random.Generator) -> dict[str, float]:
    s, t = score[portfolios], truth[portfolios]
    order = np.lexsort((rng.random(s.shape), -s), axis=1)  # by score, ties at random
    chosen = np.take_along_axis(t, order[:, :capacity], axis=1).sum(axis=1)
    best = -np.sort(-t, axis=1)[:, :capacity].sum(axis=1)
    random = capacity * t.mean(axis=1)
    gain = (best - random).sum()
    return {"value": float((chosen - random).sum() / gain) if gain > 0 else float("nan"),
            "regret": float((best - chosen).mean())}
