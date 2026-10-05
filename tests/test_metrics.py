"""Metrics against hand-computed answers and brute-force definitions."""

import numpy as np
import pytest

from rewind.metrics import auc, calibration, ece, gauc, log_loss, ne


def brute_auc(y, p):
    pos, neg = p[y == 1], p[y == 0]
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return wins / (len(pos) * len(neg))


def test_auc_extremes_and_ties():
    y = np.array([0, 0, 1, 1])
    assert auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert auc(y, np.full(4, 0.5)) == 0.5


@pytest.mark.parametrize("seed", range(5))
def test_auc_equals_pairwise_definition(seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, 400)
    p = np.round(rng.random(400), 2)  # rounding creates ties
    assert auc(y, p) == pytest.approx(brute_auc(y, p))


def test_gauc_by_hand():
    # User A: perfect ranking (AUC 1), 4 impressions. User B: reversed (AUC 0), 2 impressions.
    # User C: all non-clicks, skipped. Weighted: (1*4 + 0*2) / 6.
    y = np.array([0, 0, 1, 1, 1, 0, 0, 0])
    p = np.array([.1, .2, .8, .9, .1, .9, .5, .5])
    g = np.array(["A", "A", "A", "A", "B", "B", "C", "C"])
    assert gauc(y, p, g) == pytest.approx(4 / 6)


@pytest.mark.parametrize("seed", range(3))
def test_gauc_equals_per_user_brute_force(seed):
    rng = np.random.default_rng(seed)
    g = rng.integers(0, 20, 600)
    y = rng.integers(0, 2, 600)
    p = np.round(rng.random(600), 2)
    num = den = 0.0
    for u in np.unique(g):
        m = g == u
        if 0 < y[m].sum() < m.sum():
            num += brute_auc(y[m], p[m]) * m.sum()
            den += m.sum()
    assert gauc(y, p, g) == pytest.approx(num / den)


def test_ne_of_constant_base_rate_is_one_and_perfect_is_near_zero():
    y = np.array([0] * 95 + [1] * 5)
    assert ne(y, np.full(100, 0.05)) == pytest.approx(1.0)
    assert ne(y, y.astype(float)) < 1e-5


def test_log_loss_by_hand():
    assert log_loss(np.array([1, 0]), np.array([0.5, 0.5])) == pytest.approx(np.log(2))


def test_ece_and_calibration_on_known_cases():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.01, 0.2, 400_000)
    y = (rng.random(400_000) < p).astype(int)  # perfectly calibrated by construction
    assert ece(y, p) < 0.002
    assert calibration(y, p) == pytest.approx(1.0, abs=0.01)
    # Predicting 30% too high: the overall ratio says 1.3, and ECE is ~30% of the mean prediction.
    assert calibration(y, 1.3 * p) == pytest.approx(1.3, abs=0.02)
    assert ece(y, 1.3 * p) == pytest.approx(0.3 * p.mean(), rel=0.1)
