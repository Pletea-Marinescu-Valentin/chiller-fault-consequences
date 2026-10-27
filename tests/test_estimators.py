import numpy as np
import pandas as pd
import pytest

from cfc.context import YEAR_START, folds
from cfc.estimators import AnomalyScore, CrossFitted

PARAMS = {"n_estimators": 100, "learning_rate": 0.1, "num_leaves": 7, "min_child_samples": 10}


def weather_frame(n_hours=26 * 168, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.date_range(YEAR_START, periods=n_hours, freq="h")
    t = np.arange(n_hours)
    oat = 10 + 8 * np.sin(2 * np.pi * t / 24) + 5 * np.sin(2 * np.pi * t / (24 * 30))
    df = pd.DataFrame({"OA_TEMP": oat, "OA_TEMP_WB": oat - 3}, index=index)
    df["P1"] = 2.0 * oat + rng.normal(0, 0.5, n_hours)
    df["P2"] = np.maximum(oat - 12, 0) + rng.normal(0, 0.2, n_hours)
    return df


def test_cross_fitted_predictions_never_see_their_own_fold():
    n = 13 * 50
    fold = np.repeat(np.arange(13), 50)
    X = fold[:, None].astype(float)
    y = 10.0 * fold
    model = CrossFitted(PARAMS | {"min_child_samples": 5}, 13).fit(X, y, fold)
    pred = model.predict(X, fold)
    for f in range(13):  # a model that saw fold f would reproduce 10 f exactly
        assert np.all(np.abs(pred[fold == f] - 10.0 * f) > 1.0)
    assert len(pred) == n


def test_anomaly_score_grows_with_a_shift_in_one_point():
    ff = weather_frame()
    fold = folds(ff.index, 13)
    score = AnomalyScore(["P1", "P2"], PARAMS, 13).fit(ff, fold)
    base = np.nanmean(score.weekly(ff, fold))
    shifted = ff.copy()
    shifted["P1"] += 5 * score.scale["P1"]
    assert base < 1.5
    assert np.nanmean(score.weekly(shifted, fold)) > 3 * base


def test_anomaly_scale_has_a_floor():
    ff = weather_frame()
    ff["P2"] = 0.5 * ff["OA_TEMP"]  # predicted almost exactly from the context
    score = AnomalyScore(["P2"], PARAMS, 13, scale_floor=0.05).fit(ff, folds(ff.index, 13))
    assert score.scale["P2"] >= 0.05 * ff["P2"].std() - 1e-12
