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
@pytest.mark.parametrize('method', sorted(set(TIMESCALE_METHODS) - {'lag_one', 'acf_ratio_21'}))
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

# --------------------------------------------------------------------------------------------
# Amplitude-invariant estimators
# --------------------------------------------------------------------------------------------
from snaplab_tools.timescales import (  # noqa: E402  (kept beside the tests that need them)
    acf_ratio_21,
    tau_expfit_free,
    tau_ratio_21,
    zero_crossing_interp,
)
import scipy as sp  # noqa: E402

SIM_N, SIM_TR, SIM_REPS = 405, 0.72, 500
SIM_PHIS = (0.5, 0.7, 0.9)
SIM_SNRS = (np.inf, 4.0, 1.0, 0.5)          # signal variance / white-noise variance; inf = no noise
#: Fixed before the simulation was run: an invariant estimator's median at every SNR must lie within
#: this fraction of its noise-free median.
INVARIANCE_RTOL = 0.10
#: zero_crossing_interp is deliberately absent: no zero crossing is amplitude-invariant (see the
#: module docstring). tau_ratio_21 is tested through its aggregate, -TR/ln(median ratio).
INVARIANT = ('acf_ratio_21', 'tau_ratio_21', 'tau_expfit_free')
AMPLITUDE_DEPENDENT = ('auc_zero_crossing', 'lag_one')


@pytest.fixture(scope='module')
def simulated_acfs():
    """ACFs of AR(1) series (n=405, TR 0.72) plus white noise, SIM_REPS per (phi, snr)."""
    rng = np.random.default_rng(20261012)
    out = {}
    for phi in SIM_PHIS:
        signal_var = 1.0 / (1.0 - phi ** 2)
        for snr in SIM_SNRS:
            start = rng.standard_normal(SIM_REPS) * np.sqrt(signal_var)
            series = sp.signal.lfilter([1.0], [1.0, -phi], rng.standard_normal((SIM_REPS, SIM_N)),
                                       axis=1, zi=(phi * start)[:, None])[0]
            if np.isfinite(snr):
                series = series + rng.standard_normal((SIM_REPS, SIM_N)) * np.sqrt(signal_var / snr)
            out[phi, snr] = compute_acf(series, use_fft=True)
    return out


def _median(acf, method):
    """The group summary each estimator is meant to be read with. tau_ratio_21 aggregates the ratio
    and converts the aggregate: a median of per-cell tau drops the ratio >= 1 tail and is biased."""
    if method == 'tau_ratio_21':
        return -SIM_TR / np.log(np.nanmedian(acf_ratio_21(acf)))
    return np.nanmedian(TIMESCALE_METHODS[method](acf, SIM_TR))


def _medians(simulated_acfs, method, phi):
    return np.array([_median(simulated_acfs[phi, snr], method) for snr in SIM_SNRS])


@pytest.mark.parametrize('phi', SIM_PHIS)
@pytest.mark.parametrize('method', INVARIANT)
def test_invariant_estimators_do_not_move_with_snr(simulated_acfs, method, phi):
    medians = _medians(simulated_acfs, method, phi)
    relative = medians / medians[0] - 1
    assert np.all(np.abs(relative) <= INVARIANCE_RTOL), (
        f'{method}, phi={phi}: median by SNR {dict(zip(SIM_SNRS, np.round(medians, 3)))}, '
        f'relative to noise-free {np.round(relative, 3)}')


@pytest.mark.parametrize('phi', SIM_PHIS)
@pytest.mark.parametrize('method', AMPLITUDE_DEPENDENT)
def test_amplitude_dependent_estimators_shrink_with_snr(simulated_acfs, method, phi):
    medians = _medians(simulated_acfs, method, phi)
    assert np.all(np.diff(medians) < 0), f'{method}, phi={phi}: {np.round(medians, 3)}'


def test_zero_crossing_interp_is_within_one_tr_below_zero_crossing(simulated_acfs):
    for acf in simulated_acfs.values():
        coarse, fine = zero_crossing(acf, SIM_TR), zero_crossing_interp(acf, SIM_TR)
        assert np.array_equal(np.isnan(coarse), np.isnan(fine))
        gap = (coarse - fine)[np.isfinite(coarse)]
        assert np.all((gap > 0) & (gap <= SIM_TR))


@pytest.mark.parametrize('phi', SIM_PHIS)
def test_tau_ratio_recovers_the_ar1_decay_time(simulated_acfs, phi):
    """Noise-free; the same 10 % tolerance as the invariance test.

    At phi = 0.9 the reference is -TR/ln of the median SAMPLE ratio, not -TR/ln(phi). The gap
    between the two (about 10 % at n = 405) is the finite-sample bias of the ACF itself -- mean
    removal and the biased normalisation pull ACF(2) down more than ACF(1) -- not an error of the
    estimator, which is checked here to be the exact transform of the ratio it is given.
    """
    acf = simulated_acfs[phi, np.inf]
    ratio = acf_ratio_21(acf)
    tau = tau_ratio_21(acf, SIM_TR)
    defined = (ratio > 0) & (ratio < 1)
    assert np.allclose(tau[defined], -SIM_TR / np.log(ratio[defined]))
    reference = (-SIM_TR / np.log(np.nanmedian(ratio))) if phi == 0.9 else (-SIM_TR / np.log(phi))
    assert _median(acf, 'tau_ratio_21') == pytest.approx(reference, rel=INVARIANCE_RTOL)


def test_zero_crossing_interp_on_a_known_line():
    """Crossing between 0.3 at lag 2 and -0.1 at lag 3 is at lag 2.75."""
    acf = np.array([1.0, 0.6, 0.3, -0.1, -0.2])
    assert zero_crossing_interp(acf, tr=2.0) == pytest.approx(5.5)
    assert np.isnan(zero_crossing_interp(np.array([1.0, 0.5, 0.2]), tr=1.0))


def test_ratio_and_tau_are_nan_outside_their_domain():
    acf = np.array([[1.0, 0.5, 0.25],      # ratio 0.5
                    [1.0, -0.1, 0.05],     # ACF(1) <= 0
                    [1.0, 0.2, 0.3],       # ratio >= 1
                    [1.0, 0.2, -0.1]])     # ratio <= 0
    ratio = acf_ratio_21(acf)
    assert ratio[0] == pytest.approx(0.5) and np.isnan(ratio[1])
    tau = tau_ratio_21(acf, tr=0.72)
    assert tau[0] == pytest.approx(-0.72 / np.log(0.5))
    assert np.isnan(tau[1:]).all()


def test_expfit_recovers_an_exact_exponential_whatever_its_amplitude():
    lags = np.arange(40)
    for amplitude in (1.0, 0.4):
        acf = np.r_[1.0, amplitude * np.exp(-lags[1:] * 0.72 / 3.0)]
        assert tau_expfit_free(acf, tr=0.72) == pytest.approx(3.0, rel=1e-4)


def test_expfit_status_codes_and_censoring():
    lags = np.arange(40)
    acf = np.vstack([np.r_[1.0, np.exp(-lags[1:] * 0.72 / 3.0)],     # fits
                     np.r_[1.0, np.full(39, 0.5)],                   # flat: tau runs to the upper bound
                     np.r_[1.0, 0.4, np.nan, np.zeros(37)]])         # NaN inside the window
    tau, amplitude, status = tau_expfit_free(acf, tr=0.72, return_status=True)
    assert status.tolist() == [0, 3, 1]
    assert tau[1] == pytest.approx(1000 * 0.72)       # a bounded fit is censored at the bound, not NaN
    assert np.isnan(tau[2]) and np.isfinite(amplitude[:2]).all()


def test_expfit_window_is_fixed_and_exposed():
    """Only lags 1..max_lag are read: changing lag 5 onwards cannot move a max_lag = 4 fit."""
    lags = np.arange(40)
    base = np.r_[1.0, 0.8 * np.exp(-lags[1:] * 0.72 / 2.0)]
    other = base.copy()
    other[5:] = -0.3
    assert tau_expfit_free(base, tr=0.72) == tau_expfit_free(other, tr=0.72)
    assert tau_expfit_free(base, tr=0.72, max_lag=8) != tau_expfit_free(other, tr=0.72, max_lag=8)
    with pytest.raises(ValueError):
        tau_expfit_free(base, tr=0.72, max_lag=1)


def test_expfit_parallel_and_serial_agree(simulated_acfs):
    acf = simulated_acfs[0.7, 1.0]
    assert np.array_equal(tau_expfit_free(acf, 0.72, n_jobs=1), tau_expfit_free(acf, 0.72, n_jobs=4),
                          equal_nan=True)
