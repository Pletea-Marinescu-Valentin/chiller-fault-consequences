import numpy as np
import pytest

from cfc.ranking import decision_value, kendall, sample_portfolios, within_group_kendall


def instances(n_cases=12, n_weeks=20, seed=0):
    rng = np.random.default_rng(seed)
    case = np.repeat(np.arange(n_cases), n_weeks)
    week = np.tile(np.arange(n_weeks), n_cases)
    truth = rng.normal(size=case.size) * (1 + case)  # cases differ in scale
    return case, week, truth


def test_portfolios_have_distinct_cases_and_weeks():
    case, week, _ = instances()
    p = sample_portfolios(case, week, size=5, n=500, rng=np.random.default_rng(1))
    assert p.shape == (500, 5)
    assert all(len(set(case[row])) == 5 and len(set(week[row])) == 5 for row in p)


def test_same_week_portfolios_share_one_week():
    case, week, _ = instances()
    p = sample_portfolios(case, week, size=4, n=200, rng=np.random.default_rng(2), same_week=True)
    assert all(len(set(week[row])) == 1 and len(set(case[row])) == 4 for row in p)


def test_decision_value_is_one_for_the_oracle_and_negative_when_inverted():
    case, week, truth = instances()
    rng = np.random.default_rng(3)
    p = sample_portfolios(case, week, size=5, n=2000, rng=rng)
    assert decision_value(truth, truth, p, 1, rng)["value"] == pytest.approx(1.0)
    assert decision_value(truth, truth, p, 1, rng)["regret"] == pytest.approx(0.0)
    assert decision_value(-truth, truth, p, 1, rng)["value"] < -0.5


def test_decision_value_of_a_constant_score_is_near_zero():
    case, week, truth = instances()
    rng = np.random.default_rng(4)
    p = sample_portfolios(case, week, size=5, n=20000, rng=rng)
    value = decision_value(np.zeros_like(truth), truth, p, 2, rng)["value"]
    assert abs(value) < 0.05  # ties are broken at random


def test_kendall_is_undefined_for_a_constant_score():
    _, week, truth = instances()
    assert np.isnan(kendall(np.ones_like(truth), truth))
    assert kendall(truth, truth) == pytest.approx(1.0)
    assert within_group_kendall(2 * truth, truth, week) == pytest.approx(1.0)
