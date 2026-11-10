"""Tests for snaplab_tools.stats.icc.

The reference values are the worked example of Shrout & Fleiss (1979, Table 2): six targets rated by
four judges, ICC(1,1) = 0.17, ICC(2,1) = 0.29, ICC(3,1) = 0.71.
"""
import numpy as np
import pytest

from snaplab_tools.stats import icc

SHROUT_FLEISS = np.array([[9, 2, 5, 8],
                          [6, 1, 3, 2],
                          [8, 4, 6, 8],
                          [7, 1, 2, 6],
                          [10, 5, 6, 9],
                          [6, 2, 4, 7]], dtype=float)


@pytest.mark.parametrize('kind, expected', [('ICC(1,1)', 0.17), ('ICC(2,1)', 0.29), ('ICC(3,1)', 0.71)])
def test_shrout_fleiss_example(kind, expected):
    assert icc(SHROUT_FLEISS, kind) == pytest.approx(expected, abs=0.005)


def test_icc31_ignores_a_constant_shift_between_measurements():
    """ICC(3,1) is consistency: adding a constant to one run changes nothing; ICC(2,1) drops."""
    shifted = SHROUT_FLEISS.copy()
    shifted[:, 0] += 10
    assert icc(shifted, 'ICC(3,1)') == pytest.approx(icc(SHROUT_FLEISS, 'ICC(3,1)'))
    assert icc(shifted, 'ICC(2,1)') < icc(SHROUT_FLEISS, 'ICC(2,1)')


def test_simulated_reliability():
    """Subject variance 1 and error variance 1 give ICC = 0.5."""
    rng = np.random.default_rng(0)
    subject = rng.standard_normal((5000, 1))
    data = subject + rng.standard_normal((5000, 3))
    assert icc(data) == pytest.approx(0.5, abs=0.03)


def test_features_and_listwise_deletion():
    data = np.stack([SHROUT_FLEISS, SHROUT_FLEISS[:, ::-1]], axis=2)
    data[0, 1, 1] = np.nan                         # subject 0 dropped for feature 1 only
    out = icc(data)
    assert out.shape == (2,)
    assert out[0] == pytest.approx(icc(SHROUT_FLEISS))
    assert out[1] == pytest.approx(icc(SHROUT_FLEISS[1:, ::-1]))


def test_too_few_subjects_is_nan_and_bad_kind_raises():
    assert np.isnan(icc(np.array([[1.0, 2.0], [np.nan, 3.0]])))
    with pytest.raises(ValueError):
        icc(SHROUT_FLEISS, 'ICC(3,k)')
