"""Consequence estimators that use only what a BAS records during the faulty week.

The energy penalty of week w is estimated as observed minus counterfactual,

    dE_hat(w) = sum_{t in w} [P_obs(t) - B(z_t)],

where B is the fault-free plant power predicted from the exogenous context z.
With a plant power meter P_obs is the metered power and B is fitted on metered
fault-free history; without meters both are replaced by the virtual meter of
cfc.virtual_meter. The service deficit uses the supply-temperature excess,
which basic sensors measure in both cases:

    dS_hat(w) = sum_{t in w} [X(t) - B_X(z_t)].

Every model is fitted on the fault-free year only and cross-fitted over the
weekly folds of cfc.context.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from .context import context, weekly_mean, weekly_sum


class CrossFitted:
    """One gradient-boosted model per fold, each fitted on the other folds.

    predict() uses for every row the model that did not see that row's fold,
    so fault-free predictions are out of sample and a faulty week is predicted
    by a model that never saw the same week of the fault-free year.
    """

    def __init__(self, params: dict, n_folds: int, seed: int = 0):
        self.params, self.n_folds, self.seed = params, n_folds, seed

    def fit(self, X: np.ndarray, y: np.ndarray, fold: np.ndarray) -> "CrossFitted":
        self.models = [lgb.LGBMRegressor(**self.params, random_state=self.seed, verbose=-1)
                       .fit(X[fold != f], y[fold != f]) for f in range(self.n_folds)]
        return self

    def predict(self, X: np.ndarray, fold: np.ndarray) -> np.ndarray:
        out = np.empty(len(X))
        for f, model in enumerate(self.models):
            sel = fold == f
            if sel.any():
                out[sel] = model.predict(X[sel])
        return out


def counterfactual(ff: pd.DataFrame, target: np.ndarray, fold: np.ndarray, params: dict,
                   n_folds: int, seed: int = 0) -> CrossFitted:
    """Fault-free model target(z) of a plant quantity on the exogenous context."""
    return CrossFitted(params, n_folds, seed).fit(context(ff).to_numpy(), np.asarray(target), fold)


def penalty(observed: np.ndarray, model: CrossFitted, frame: pd.DataFrame,
            fold: np.ndarray) -> np.ndarray:
    """Weekly sums of observed minus counterfactual (kWh for power, K h for the excess)."""
    return weekly_sum(np.asarray(observed) - model.predict(context(frame).to_numpy(), fold),
                      frame.index)


class AnomalyScore:
    """Magnitude of the departure of the BAS points from their fault-free context models.

    Each point j gets a counterfactual model; its out-of-fold residual on the
    fault-free year sets the scale sigma_j. The hourly score is the root mean
    square of the standardized residuals and the weekly score its mean, i.e.
    how abnormal the plant looks, regardless of what the anomaly costs.
    """

    def __init__(self, points: list[str], params: dict, n_folds: int, seed: int = 0,
                 scale_floor: float = 0.05):
        self.points, self.params, self.n_folds, self.seed = points, params, n_folds, seed
        self.scale_floor = scale_floor

    def fit(self, ff: pd.DataFrame, fold: np.ndarray) -> "AnomalyScore":
        Z = context(ff).to_numpy()
        self.models, self.scale = {}, {}
        for p in self.points:
            m = CrossFitted(self.params, self.n_folds, self.seed).fit(Z, ff[p].to_numpy(), fold)
            resid = ff[p].to_numpy() - m.predict(Z, fold)
            self.models[p] = m
            self.scale[p] = max(resid.std(), self.scale_floor * ff[p].std())
        return self

    def weekly(self, frame: pd.DataFrame, fold: np.ndarray) -> np.ndarray:
        Z = context(frame).to_numpy()
        sq = np.zeros(len(frame))
        for p, m in self.models.items():
            sq += ((frame[p].to_numpy() - m.predict(Z, fold)) / self.scale[p]) ** 2
        return weekly_mean(np.sqrt(sq / len(self.models)), frame.index)
