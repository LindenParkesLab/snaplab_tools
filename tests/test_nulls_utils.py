"""Tests for snaplab_tools.nulls.get_null_p.

Several of these pin down behaviour that used to be a silent wrong answer or an
UnboundLocalError rather than a defined result.
"""
import numpy as np
import pytest

from snaplab_tools.nulls import get_null_p


@pytest.fixture
def null():
    return np.random.default_rng(0).standard_normal(1000)


def test_positive_observed_uses_upper_tail(null):
    assert get_null_p(1.5, null) == pytest.approx(np.mean(null >= 1.5))


def test_negative_observed_uses_lower_tail(null):
    assert get_null_p(-1.5, null) == pytest.approx(np.mean(null <= -1.5))


def test_zero_observed_uses_upper_tail(null):
    assert get_null_p(0.0, null) == pytest.approx(np.mean(null >= 0.0))


def test_abs_tests_magnitude(null):
    assert get_null_p(-1.5, null, abs=True) == pytest.approx(np.mean(np.abs(null) >= 1.5))


def test_p_is_bounded(null):
    for observed in (-10.0, -1.0, 0.0, 1.0, 10.0):
        assert 0.0 <= get_null_p(observed, null) <= 1.0


def test_smallest_never_exceeds_standard(null):
    for observed in (-2.0, -0.5, 0.5, 2.0):
        smallest = get_null_p(observed, null, version='smallest')
        standard = get_null_p(observed, null, version='standard')
        assert smallest <= standard + 1e-12


def test_nan_observed_returns_nan(null):
    """A statistic that could not be computed yields a NaN p-value, not an exception.

    Callers loop over regions; one undefined region should leave a hole in the result vector
    rather than abort the loop.
    """
    assert np.isnan(get_null_p(np.nan, null))


def test_empty_or_all_nan_null_returns_nan():
    assert np.isnan(get_null_p(1.5, np.full(10, np.nan)))
    assert np.isnan(get_null_p(1.5, np.array([])))


def test_non_finite_null_entries_are_dropped_from_the_denominator(null):
    """NaN surrogates must not inflate n -- doing so makes the p-value anti-conservative."""
    contaminated = null.copy()
    contaminated[:200] = np.nan
    finite = contaminated[np.isfinite(contaminated)]

    assert get_null_p(1.5, contaminated) == pytest.approx(np.mean(finite >= 1.5))
    # And it is not the value you would get by counting the NaNs in the denominator.
    assert get_null_p(1.5, contaminated) != pytest.approx(np.sum(finite >= 1.5) / len(contaminated))


def test_unknown_version_raises(null):
    with pytest.raises(ValueError, match="version must be"):
        get_null_p(1.5, null, version='bogus')


def test_accepts_list_input(null):
    assert get_null_p(1.5, list(null)) == pytest.approx(get_null_p(1.5, null))
