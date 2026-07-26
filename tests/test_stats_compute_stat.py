"""Tests for snaplab_tools.stats.compute_stat, including the p-value-free fast path."""
import warnings

import numpy as np
import pytest

from snaplab_tools.stats import compute_stat


@pytest.fixture
def xy():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(200)
    return x, 0.4 * x + rng.standard_normal(200)


@pytest.mark.parametrize('method', ['pearson', 'spearman', 'r2'])
def test_fast_path_matches_the_scipy_path(xy, method):
    """return_p=False must change only the cost, never the statistic."""
    x, y = xy
    slow, _ = compute_stat(x, y, method)
    fast, _ = compute_stat(x, y, method, return_p=False)
    assert fast == pytest.approx(slow, abs=1e-12)


@pytest.mark.parametrize('method', ['pearson', 'spearman', 'r2'])
def test_p_is_nan_when_not_requested(xy, method):
    x, y = xy
    assert np.isnan(compute_stat(x, y, method, return_p=False)[1])
    assert not np.isnan(compute_stat(x, y, method)[1])


def test_spearman_fast_path_handles_ties(xy):
    """Spearman is Pearson on ranks, so tie handling has to match scipy's average ranks."""
    ties = np.array([1., 1., 2., 2., 3., 3., 4., 4.])
    other = np.array([2., 1., 4., 3., 6., 5., 8., 7.])
    slow, _ = compute_stat(ties, other, 'spearman')
    fast, _ = compute_stat(ties, other, 'spearman', return_p=False)
    assert fast == pytest.approx(slow)


def test_fast_path_returns_nan_for_constant_input():
    """A constant input has no variance, so the correlation is undefined -- not 0, not 1."""
    constant = np.ones(50)
    varying = np.random.default_rng(0).standard_normal(50)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        assert np.isnan(compute_stat(constant, varying, 'pearson', return_p=False)[0])


def test_fast_path_clamps_to_the_valid_range():
    """Floating-point error must not push a perfect correlation outside [-1, 1]."""
    x = np.arange(500, dtype=float)
    assert compute_stat(x, x, 'pearson', return_p=False)[0] == 1.0
    assert compute_stat(x, -x, 'pearson', return_p=False)[0] == -1.0


@pytest.mark.parametrize('return_p', [True, False])
def test_too_few_points_is_nan_either_way(return_p):
    x = np.array([1.0, 2.0])
    assert all(np.isnan(v) for v in compute_stat(x, x, 'pearson', return_p=return_p))
