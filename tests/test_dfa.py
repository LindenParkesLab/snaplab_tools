"""Tests for detrended fluctuation analysis (DFA) in snaplab_tools.timescales.

Every test has an answer known from theory: alpha = 0.5 for white noise and 1.0 for 1/f noise, and
an AR(1) process crosses over from alpha = 1.5 (scales much shorter than its time constant, where
the process looks like a random walk) to 0.5 (much longer, where it looks white). For DFA-1 the
squared fluctuation of white noise of variance s2 is s2 * n / 15, and of a random walk with step
variance s2 it is s2 * n**3 / 420, so for AR(1) with coefficient phi the two asymptotes meet at
n = sqrt(420 / 15) / (1 - phi) = sqrt(28) / (1 - phi).
"""
import numpy as np
import pytest
import scipy as sp

from snaplab_tools.timescales import dfa, dfa_exponent, dfa_fluctuation

TOL = 0.05


def white(n, seed):
    return np.random.default_rng(seed).standard_normal(n)


def pink(n, seed):
    """1/f noise by spectral synthesis (power proportional to 1/f)."""
    rng = np.random.default_rng(seed)
    f = np.fft.rfftfreq(n)
    spectrum = rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f))
    spectrum[1:] *= f[1:] ** -0.5
    spectrum[0] = 0
    return np.fft.irfft(spectrum, n)


def ar1(n, phi, seed):
    return sp.signal.lfilter([1.0], [1.0, -phi], np.random.default_rng(seed).standard_normal(n))


def log_scales(lo, hi, num):
    return np.unique(np.geomspace(lo, hi, num).astype(int))


def drop_random(x, fraction, seed):
    """Delete a random `fraction` of samples and concatenate the rest."""
    good = np.ones(len(x), bool)
    good[np.random.default_rng(seed).choice(len(x), int(fraction * len(x)), replace=False)] = False
    return x[good]


# --------------------------------------------------------------------------------------------
# Known exponents
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize('masked', [False, True])
def test_white_noise_alpha_is_one_half(masked):
    x = white(2 ** 16, seed=0)
    if masked:
        x = drop_random(x, 0.10, seed=1)
    alpha, r2 = dfa(x, log_scales(16, 4096, 20))
    assert alpha == pytest.approx(0.5, abs=TOL)
    assert r2 > 0.98


@pytest.mark.parametrize('masked', [False, True])
def test_pink_noise_alpha_is_one(masked):
    x = pink(2 ** 17, seed=0)
    if masked:
        x = drop_random(x, 0.10, seed=1)
    alpha, r2 = dfa(x, log_scales(16, 8192, 20))
    assert alpha == pytest.approx(1.0, abs=TOL)
    assert r2 > 0.98


def test_ar1_crossover():
    phi = np.exp(-1 / 50)
    x = ar1(2 ** 22, phi, seed=0)
    short, long_ = log_scales(5, 6, 2), log_scales(2500, 12500, 6)   # ~0.1 and 50-250 time constants
    F_short, F_long = dfa_fluctuation(x, short), dfa_fluctuation(x, long_)

    assert dfa_exponent(short, F_short)[0] == pytest.approx(1.5, abs=TOL)
    assert dfa_exponent(long_, F_long)[0] == pytest.approx(0.5, abs=TOL)

    # Where lines of slope 1.5 and 0.5 through the two regimes meet: sqrt(28) / (1 - phi).
    c_short = np.mean(np.log10(F_short) - 1.5 * np.log10(short))
    c_long = np.mean(np.log10(F_long) - 0.5 * np.log10(long_))
    assert 10 ** (c_long - c_short) == pytest.approx(np.sqrt(28) / (1 - phi), rel=0.10)


def test_white_noise_fluctuation_matches_theory():
    """F(n)**2 = n / 15 for unit-variance white noise under DFA-1."""
    scales = np.array([50, 100, 200])
    F = dfa_fluctuation(white(2 ** 18, seed=2), scales)
    np.testing.assert_allclose(F ** 2, scales / 15, rtol=0.05)


# --------------------------------------------------------------------------------------------
# Mechanics
# --------------------------------------------------------------------------------------------
def test_shapes_and_counts():
    x = np.random.default_rng(0).standard_normal((3, 1000))
    scales = [10, 100, 1000, 2000]
    F, counts = dfa_fluctuation(x, scales, return_counts=True)
    assert F.shape == (3, 4)
    # 50 % overlap: starts every n // 2 samples, complete segments only.
    np.testing.assert_array_equal(counts, [199, 19, 1, 0])
    assert np.isnan(F[:, 3]).all()
    assert dfa_fluctuation(x[0], scales).shape == (4,)


def test_rows_are_independent():
    x = np.random.default_rng(0).standard_normal((3, 2000))
    scales = [10, 40, 160]
    np.testing.assert_allclose(dfa_fluctuation(x, scales)[1], dfa_fluctuation(x[1], scales))


def test_invariant_to_offset_and_linear_trend():
    """The profile's mean is removed and DFA-1 detrends each segment, so an offset in the series
    does nothing, and a linear trend in the series (quadratic in the profile) needs DFA-2."""
    x = white(5000, seed=3)
    scales = [20, 80, 320]
    np.testing.assert_allclose(dfa_fluctuation(x + 7.0, scales), dfa_fluctuation(x, scales), rtol=1e-9)
    trend = 1e-3 * np.arange(5000)
    np.testing.assert_allclose(dfa_fluctuation(x + trend, scales, order=2), dfa_fluctuation(x, scales, order=2),
                               rtol=1e-6)


def test_good_mask_skips_segments_and_ignores_masked_values():
    x = white(4000, seed=4)
    good = np.ones(4000, bool)
    good[1000:1100] = False
    spoiled = x.copy()
    spoiled[1000:1100] = 1e6                       # masked samples must not matter
    scales = [20, 50, 200]
    F_a, n_a = dfa_fluctuation(x, scales, good=good, return_counts=True)
    F_b, n_b = dfa_fluctuation(spoiled, scales, good=good, return_counts=True)
    np.testing.assert_allclose(F_a, F_b, rtol=1e-9)
    np.testing.assert_array_equal(n_a, n_b)
    assert (n_a < dfa_fluctuation(x, scales, return_counts=True)[1]).all()


def test_counts_pool_recordings():
    """Pooling F**2 by segment counts equals one call over segments of both recordings."""
    rng = np.random.default_rng(5)
    a, b = rng.standard_normal(3000), rng.standard_normal(1800)
    scales = [30, 90]
    Fa, na = dfa_fluctuation(a, scales, return_counts=True)
    Fb, nb = dfa_fluctuation(b, scales, return_counts=True)
    pooled = np.sqrt((Fa ** 2 * na + Fb ** 2 * nb) / (na + nb))
    # Reference: segments computed by hand for each recording.
    for j, n in enumerate(scales):
        sq = []
        for x in (a, b):
            prof = np.cumsum(x - x.mean())
            for s in range(0, len(x) - n + 1, n // 2):
                seg = prof[s:s + n]
                t = np.arange(n)
                sq.append(np.mean((seg - np.polyval(np.polyfit(t, seg, 1), t)) ** 2))
        assert pooled[j] == pytest.approx(np.sqrt(np.mean(sq)), rel=1e-9)


def test_exponent_fit_range_and_nan():
    scales = np.array([10, 20, 40, 80, 160])
    F = np.vstack([scales ** 0.7, scales ** 0.7])
    F[1, 4] = np.nan
    alpha, r2 = dfa_exponent(scales, F)
    assert alpha[0] == pytest.approx(0.7) and r2[0] == pytest.approx(1.0)
    assert np.isnan(alpha[1])
    alpha, _ = dfa_exponent(scales, F, fit_range=(10, 80))
    np.testing.assert_allclose(alpha, 0.7)


def test_invalid_arguments():
    x = white(100, seed=0)
    with pytest.raises(ValueError):
        dfa_fluctuation(x, [2])                    # shorter than order + 2
    with pytest.raises(ValueError):
        dfa_fluctuation(x, [10], overlap=1.0)
    with pytest.raises(ValueError):
        dfa_fluctuation(x, [10], good=np.ones(50, bool))
    with pytest.raises(ValueError):
        dfa_exponent([10, 20, 40], [1.0, 2.0, 3.0], fit_range=(15, 25))
