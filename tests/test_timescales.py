"""Tests for autocorrelation and intrinsic-timescale estimation.

The estimators are simple enough that the useful tests are the ones with a known answer: an ACF
built by hand, whose crossing lag and area can be worked out on paper, and an AR(1) process whose
true autocorrelation is ``rho ** k``.
"""
import warnings

import numpy as np
import pytest

from snaplab_tools.timescales import (
    TIMESCALE_METHODS,
    auc,
    compute_acf,
    compute_timescale,
    decay_time,
    lag_one,
    zero_crossing,
)


@pytest.fixture
def acf_matrix():
    """Two ACFs whose crossings are known by construction.

    Row 0 decays with a length constant of 10 lags; row 1 decays four times faster, so every
    estimator must report a shorter timescale for it than for row 0.
    """
    lags = np.arange(60)
    return np.vstack([np.exp(-lags / 10.0) * np.cos(lags / 20.0),
                      np.exp(-lags / 2.5) * np.cos(lags / 5.0)])


# --------------------------------------------------------------------------------------------
# compute_acf
# --------------------------------------------------------------------------------------------
def test_acf_of_ar1_matches_theory():
    """For an AR(1) process the autocorrelation at lag k is rho ** k."""
    rho, n = 0.8, 20000
    rng = np.random.default_rng(0)
    series = np.zeros(n)
    for t in range(1, n):
        series[t] = rho * series[t - 1] + rng.normal()

    acf = compute_acf(series, nlags=5)

    assert np.allclose(acf, rho ** np.arange(6), atol=0.02)


def test_acf_starts_at_one_and_keeps_input_shape():
    rng = np.random.default_rng(0)
    single, stack = rng.normal(size=200), rng.normal(size=(4, 200))

    assert compute_acf(single).shape == (200,)
    assert compute_acf(single)[0] == pytest.approx(1.0)
    assert compute_acf(stack).shape == (4, 200)
    assert np.allclose(compute_acf(stack)[:, 0], 1.0)


def test_fft_and_direct_paths_agree():
    """The FFT path is an optimisation, not a different estimator."""
    series = np.random.default_rng(0).normal(size=(3, 512))

    assert np.allclose(compute_acf(series, use_fft=False),
                       compute_acf(series, use_fft=True), atol=1e-10)


def test_nlags_truncates_to_lag_zero_through_nlags():
    series = np.random.default_rng(0).normal(size=(2, 300))

    assert compute_acf(series, nlags=25).shape == (2, 26)
    with pytest.raises(ValueError):
        compute_acf(series, nlags=-1)


@pytest.mark.parametrize('bad', [np.zeros(100), np.r_[np.nan, np.random.default_rng(0).normal(size=99)]])
def test_undefined_acf_is_all_nan_rather_than_an_exception(bad):
    """A flat or NaN-carrying region must not abort a whole scan."""
    good = np.random.default_rng(1).normal(size=100)

    out = compute_acf(np.vstack([good, bad]))

    assert np.isfinite(out[0]).all()
    assert np.isnan(out[1]).all()


def test_acf_rejects_degenerate_input():
    with pytest.raises(ValueError):
        compute_acf(np.array([1.0]))
    with pytest.raises(ValueError):
        compute_acf(np.zeros((2, 2, 2)))


# --------------------------------------------------------------------------------------------
# Estimators
# --------------------------------------------------------------------------------------------
def test_zero_crossing_finds_the_first_negative_lag():
    acf = np.array([1.0, 0.6, 0.2, -0.1, -0.3, 0.05])

    assert zero_crossing(acf) == 3
    assert zero_crossing(acf, tr=2.0) == 6.0      # reported in seconds


def test_zero_crossing_is_nan_when_the_acf_never_crosses():
    """A scan too short for the ACF to decay has no crossing, and must say so."""
    assert np.isnan(zero_crossing(np.array([1.0, 0.9, 0.8])))
    assert np.isnan(zero_crossing(np.full(10, np.nan)))


def test_decay_time_uses_the_threshold():
    acf = np.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])

    assert decay_time(acf, threshold=0.5) == 3
    assert decay_time(acf, threshold=0.9) == 1
    assert np.isnan(decay_time(acf, threshold=-1.0))


def test_zero_crossing_is_decay_time_at_threshold_zero(acf_matrix):
    assert np.allclose(zero_crossing(acf_matrix, tr=0.8),
                       decay_time(acf_matrix, tr=0.8, threshold=0.0))


def test_auc_integrates_with_the_trapezoidal_rule():
    """Area of a straight line from 1 to 0 over 4 lags is 2 lags, or 2*tr seconds."""
    acf = np.array([1.0, 0.75, 0.5, 0.25, 0.0, -0.25])

    assert auc(acf, max_lag=5) == pytest.approx(2.0)
    assert auc(acf, tr=0.5, max_lag=5) == pytest.approx(1.0)


def test_auc_stops_at_the_threshold_crossing():
    acf = np.array([1.0, 0.5, -0.5, -1.0])

    # Integrates lags 0..1 only: one trapezoid of height (1.0 + 0.5) / 2.
    assert auc(acf) == pytest.approx(0.75)
    assert np.isnan(auc(np.array([1.0, 0.9, 0.8])))     # never crosses, no endpoint


def test_auc_max_lag_overrides_the_threshold():
    """max_lag gives every region an identical window, even one that never crosses."""
    assert np.isfinite(auc(np.array([1.0, 0.9, 0.8]), max_lag=3))


def test_lag_one_reads_the_second_entry():
    assert lag_one(np.array([1.0, 0.42, 0.1])) == pytest.approx(0.42)
    assert lag_one(np.array([1.0, 0.42, 0.1]), tr=99.0) == pytest.approx(0.42)   # tr ignored
    with pytest.raises(ValueError):
        lag_one(np.array([1.0]))


@pytest.mark.parametrize('method', sorted(TIMESCALE_METHODS))
def test_faster_decay_gives_a_shorter_timescale(acf_matrix, method):
    """Whatever the estimator, the faster-decaying ACF must score lower."""
    slow, fast = compute_timescale(acf_matrix, tr=0.72, method=method)

    assert fast < slow


@pytest.mark.parametrize('method', sorted(TIMESCALE_METHODS))
def test_single_and_stacked_acfs_agree(acf_matrix, method):
    """A region computed alone must equal that region computed in a stack."""
    stacked = compute_timescale(acf_matrix, tr=0.72, method=method)
    alone = compute_timescale(acf_matrix[1], tr=0.72, method=method)

    assert np.isscalar(alone) or np.ndim(alone) == 0
    assert alone == pytest.approx(stacked[1])


def test_compute_timescale_rejects_an_unknown_method(acf_matrix):
    with pytest.raises(ValueError, match='unknown timescale method'):
        compute_timescale(acf_matrix, method='not_a_method')


# --------------------------------------------------------------------------------------------
# The units warning
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize('method', sorted(set(TIMESCALE_METHODS) - {'lag_one'}))
def test_default_tr_warns_that_the_result_is_in_trs(acf_matrix, method):
    """Leaving tr at 1.0 returns lags, not seconds, and silently mislabelling those as seconds
    is wrong by a factor of the TR."""
    with pytest.warns(UserWarning, match='in TRs rather than seconds'):
        compute_timescale(acf_matrix, method=method)


@pytest.mark.parametrize('method', sorted(TIMESCALE_METHODS))
def test_a_real_tr_does_not_warn(acf_matrix, method):
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        compute_timescale(acf_matrix, tr=0.72, method=method)


def test_lag_one_does_not_warn(acf_matrix):
    """The lag-1 autocorrelation is a correlation, so a units warning would be meaningless."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        compute_timescale(acf_matrix, method='lag_one')


@pytest.mark.parametrize('method', sorted(TIMESCALE_METHODS))
def test_verbose_false_silences_the_units_warning(acf_matrix, method):
    """For a real 1.0 s acquisition, which is indistinguishable from the default tr."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        compute_timescale(acf_matrix, tr=1.0, method=method, verbose=False)


def test_verbose_false_does_not_change_the_result(acf_matrix):
    """Silencing the warning must not silently change what is computed."""
    with pytest.warns(UserWarning):
        loud = compute_timescale(acf_matrix, method='zero_crossing')
    quiet = compute_timescale(acf_matrix, method='zero_crossing', verbose=False)

    assert np.array_equal(loud, quiet, equal_nan=True)


def test_the_warning_does_not_change_the_result(acf_matrix):
    with pytest.warns(UserWarning):
        warned = compute_timescale(acf_matrix, method='zero_crossing')

    assert np.array_equal(warned, zero_crossing(acf_matrix, tr=1.0), equal_nan=True)


def test_estimators_called_directly_do_not_warn(acf_matrix):
    """The warning belongs to the dispatcher; calling an estimator directly is an explicit choice."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        zero_crossing(acf_matrix)
        auc(acf_matrix)
        decay_time(acf_matrix)


def test_estimators_disagree_with_each_other(acf_matrix):
    """Different summaries of the same curve are different numbers, not interchangeable ones."""
    values = {m: compute_timescale(acf_matrix, tr=0.72, method=m)[0] for m in TIMESCALE_METHODS}

    assert len(set(np.round(list(values.values()), 6))) > 1