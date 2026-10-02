"""Tests for the Butterworth filtering in :mod:`snaplab_tools.signal`.

The defaults are ``method='pad'``, ``padtype='odd'`` and ``padlen = n_timepoints - 1``: scipy's method
and extension, with the longest padding filtfilt accepts. ``method='gust'`` stays available; its tests
pass it explicitly, so none of them depend on the default. Two are worth naming:

- ``TestEdgeArtefact`` is the regression test for the edge defect ``method='gust'`` fixes. Filtering a
  short run with scipy's ``padtype='odd'``, 15-sample padding inflates its intrinsic timescale,
  and inflates it more the shorter the run is, so it biases any comparison between recordings of
  different length. It is measured against a long-series reference, which is the only way to
  separate the edge artefact from the genuine short-run bias.
- ``TestVectorisedMatchesLoop`` pins the vectorised ``filtfilt`` call to the per-region Python
  loop it replaced. They must stay bit-identical; a divergence would silently change every
  cached result computed through this function.
"""
import numpy as np
import pytest
import scipy as sp

import snaplab_tools.signal as snaplab_signal
import snaplab_tools.timescales as snaplab_timescales

TR = 0.8
FS = 1.0 / TR
BAND = dict(lowpass=0.08, highpass=0.01)
#: scipy's own defaults, which this module deliberately does not use.
SCIPY_DEFAULTS = dict(method='pad', padtype='odd', padlen=15)
#: constant padding at maximal length -- another edge treatment, still reachable
CONSTANT_PAD = dict(method='pad', padtype='constant', padlen=None)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def _ar1(n_rep, n_timepoints, rng, phi=0.85):
    """AR(1): a known timescale, stationary, with no onset or offset structure at all."""
    noise = rng.standard_normal((n_rep, n_timepoints))
    series = np.zeros((n_rep, n_timepoints))
    for i in range(1, n_timepoints):
        series[:, i] = phi * series[:, i - 1] + noise[:, i]
    return series


def _int(block):
    acf = np.atleast_2d(snaplab_timescales.compute_acf(np.asarray(block, dtype=float),
                                                       use_fft=True))
    return np.asarray(snaplab_timescales.TIMESCALE_METHODS['auc_zero_crossing'](acf, TR),
                      dtype=float)


class TestDefaults:
    """The defaults, pinned: a change here shifts every cached result filtered through this module."""

    def test_method_default_is_pad(self):
        assert snaplab_signal.DEFAULT_METHOD == 'pad'
        params = __import__('inspect').signature(
            snaplab_signal.apply_frequency_filter).parameters
        assert params['method'].default == 'pad'

    def test_padtype_default_is_odd(self):
        assert snaplab_signal.DEFAULT_PADTYPE == 'odd'
        params = __import__('inspect').signature(
            snaplab_signal.apply_frequency_filter).parameters
        assert params['padtype'].default == 'odd'
        assert params['padlen'].default is None

    def test_the_default_is_odd_padding_at_maximal_length_not_scipys_15(self, rng):
        """Same method and extension as scipy; the padding length is what differs."""
        data = rng.standard_normal((4, 300))
        default = snaplab_signal.apply_frequency_filter(data, FS, **BAND)
        np.testing.assert_array_equal(default, snaplab_signal.apply_frequency_filter(
            data, FS, **BAND, method='pad', padtype='odd', padlen=299))
        assert not np.allclose(default, snaplab_signal.apply_frequency_filter(
            data, FS, **BAND, **SCIPY_DEFAULTS))

    def test_gust_ignores_padding_arguments(self):
        """Under 'gust' nothing is padded, so padtype/padlen must not change the result."""
        rng = np.random.default_rng(5)
        data = rng.standard_normal((4, 200))
        base = snaplab_signal.apply_frequency_filter(data, FS, **BAND, method='gust')
        for kwargs in [dict(padtype='odd', padlen=15), dict(padtype='even', padlen=99)]:
            np.testing.assert_array_equal(
                base, snaplab_signal.apply_frequency_filter(data, FS, **BAND, method='gust', **kwargs))

    def test_default_padlen_is_the_largest_filtfilt_accepts(self):
        assert snaplab_signal.resolve_padlen(478) == 477
        assert snaplab_signal.resolve_padlen(168) == 167
        # filtfilt requires strictly more data than padding, so n-1 is the maximum.
        b, a = sp.signal.butter(2, [0.01 / (FS / 2), 0.08 / (FS / 2)], btype='bandpass')
        x = np.zeros(50)
        sp.signal.filtfilt(b, a, x, padtype='constant', padlen=49)
        with pytest.raises(ValueError):
            sp.signal.filtfilt(b, a, x, padtype='constant', padlen=50)

    def test_explicit_padlen_is_honoured(self):
        assert snaplab_signal.resolve_padlen(478, padlen=15) == 15
        assert snaplab_signal.resolve_padlen(478, padlen=0) == 0

    def test_padlen_too_long_raises_with_a_useful_message(self):
        with pytest.raises(ValueError, match='must be shorter than the data'):
            snaplab_signal.resolve_padlen(100, padlen=100)

    def test_negative_padlen_raises(self):
        with pytest.raises(ValueError, match='non-negative'):
            snaplab_signal.resolve_padlen(100, padlen=-1)

    def test_degenerate_lengths_do_not_crash(self):
        assert snaplab_signal.resolve_padlen(1) == 0
        assert snaplab_signal.resolve_padlen(0) == 0


class TestScipyBehaviourIsStillReachable:
    """Reproducing a pre-fix cache must remain possible, exactly."""

    def test_odd_padlen_15_reproduces_scipys_result(self, rng):
        """scipy's exact result needs padlen=15 too: the default padding length is n_timepoints - 1."""
        data = rng.standard_normal((6, 300))
        b, a = sp.signal.butter(2, [0.01 / (FS / 2), 0.08 / (FS / 2)], btype='bandpass')
        scipy_direct = np.zeros_like(data)
        for i in range(data.shape[0]):
            scipy_direct[i, :] = sp.signal.filtfilt(b, a, data[i, :])
        through_module = snaplab_signal.apply_frequency_filter(
            data, FS, **BAND, **SCIPY_DEFAULTS)
        np.testing.assert_array_equal(scipy_direct, through_module)


class TestVectorisedMatchesLoop:
    """The Python loop over regions was replaced by one axis=-1 call. See the module docstring."""

    @pytest.mark.parametrize('shape', [(40, 478), (9, 168), (5, 1200)])
    def test_gust_float32_is_bit_identical_to_per_region_filtering(self, shape, rng):
        """float32 is what this project actually filters, and there it is exact.

        Scipy's Gustafsson path solves a small system per slice, and the batched form reorders
        those operations, so in float64 it differs from row-by-row filtering by ~3e-14 relative
        (see the float64 test below). That is far below float32 resolution, so the pipeline --
        whose timeseries are float32 throughout -- gets a bit-identical result.
        """
        data = rng.standard_normal(shape).astype(np.float32)
        b, a = sp.signal.butter(2, [0.01 / (FS / 2), 0.08 / (FS / 2)], btype='bandpass')
        loop = np.zeros_like(data)
        for i in range(shape[0]):
            loop[i, :] = sp.signal.filtfilt(b, a, data[i, :], method='gust')
        np.testing.assert_array_equal(
            loop, snaplab_signal.apply_frequency_filter(data, FS, **BAND, method='gust'))

    @pytest.mark.parametrize('shape', [(5, 1200), (3, 478)])
    def test_gust_float64_agrees_to_floating_point_reordering(self, shape, rng):
        """Not bit-identical in float64, and the tolerance is stated rather than assumed."""
        data = rng.standard_normal(shape)
        b, a = sp.signal.butter(2, [0.01 / (FS / 2), 0.08 / (FS / 2)], btype='bandpass')
        loop = np.zeros_like(data)
        for i in range(shape[0]):
            loop[i, :] = sp.signal.filtfilt(b, a, data[i, :], method='gust')
        vectorised = snaplab_signal.apply_frequency_filter(data, FS, **BAND, method='gust')
        np.testing.assert_allclose(loop, vectorised, rtol=1e-11, atol=1e-11)
        # and comfortably below what float32 could represent, which is the pipeline's dtype
        assert np.max(np.abs(loop - vectorised)) < np.finfo(np.float32).eps * np.abs(loop).max()

    @pytest.mark.parametrize('shape,dtype', [((40, 478), np.float32), ((9, 168), np.float32)])
    def test_pad_path_bit_identical_to_per_region_filtering(self, shape, dtype, rng):
        data = rng.standard_normal(shape).astype(dtype)
        b, a = sp.signal.butter(2, [0.01 / (FS / 2), 0.08 / (FS / 2)], btype='bandpass')
        loop = np.zeros_like(data)
        for i in range(shape[0]):
            loop[i, :] = sp.signal.filtfilt(b, a, data[i, :], padtype='constant',
                                            padlen=shape[1] - 1)
        np.testing.assert_array_equal(
            loop, snaplab_signal.apply_frequency_filter(data, FS, **BAND, **CONSTANT_PAD))

    @pytest.mark.parametrize('shape', [(40, 478), (9, 168)])
    def test_default_path_bit_identical_to_per_region_filtering(self, shape, rng):
        """The default (pad, odd, n - 1) on float32 -- the dtype this project filters."""
        data = rng.standard_normal(shape).astype(np.float32)
        b, a = sp.signal.butter(2, [0.01 / (FS / 2), 0.08 / (FS / 2)], btype='bandpass')
        loop = np.zeros_like(data)
        for i in range(shape[0]):
            loop[i, :] = sp.signal.filtfilt(b, a, data[i, :], padtype='odd', padlen=shape[1] - 1)
        np.testing.assert_array_equal(loop, snaplab_signal.apply_frequency_filter(data, FS, **BAND))

    def test_shape_and_dtype_are_preserved(self, rng):
        for dtype in (np.float32, np.float64):
            data = rng.standard_normal((11, 260)).astype(dtype)
            out = snaplab_signal.apply_frequency_filter(data, FS, **BAND)
            assert out.shape == data.shape and out.dtype == data.dtype

    def test_a_transposed_view_is_accepted(self, rng):
        """load_scan_timeseries passes `timeseries.T`, which is not C-contiguous."""
        data = rng.standard_normal((260, 11)).astype(np.float32)
        out = snaplab_signal.apply_frequency_filter(data.T, FS, **BAND)
        assert out.shape == (11, 260) and np.isfinite(out).all()


class TestEdgeArtefact:
    """The regression test for the defect. See the module docstring."""

    @staticmethod
    def _bias(n_timepoints, rng, **filter_kwargs):
        """Whole-run INT bias against the same samples filtered with unlimited context."""
        pad = 2000
        long_series = _ar1(40, n_timepoints + 2 * pad, rng)
        reference = snaplab_signal.apply_frequency_filter(
            long_series, FS, **BAND, method='gust')[:, pad:pad + n_timepoints]
        excerpt = long_series[:, pad:pad + n_timepoints].copy()
        filtered = snaplab_signal.apply_frequency_filter(excerpt, FS, **BAND, **filter_kwargs)
        return float(np.nanmedian(_int(filtered) - _int(reference)))

    @pytest.mark.parametrize('n_timepoints', [168, 478])
    def test_gust_beats_scipys_by_a_wide_margin(self, n_timepoints, rng):
        scipy_bias = self._bias(n_timepoints, np.random.default_rng(1), **SCIPY_DEFAULTS)
        gust_bias = self._bias(n_timepoints, np.random.default_rng(1), method='gust')
        assert scipy_bias > 0.05, 'the artefact this guards should be large under scipy defaults'
        assert abs(gust_bias) < abs(scipy_bias) / 3

    def test_constant_padding_also_beats_scipys(self, rng):
        """The rejected alternative is still far better than scipy's, and stays reachable."""
        scipy_bias = self._bias(168, np.random.default_rng(1), **SCIPY_DEFAULTS)
        pad_bias = self._bias(168, np.random.default_rng(1), **CONSTANT_PAD)
        assert abs(pad_bias) < abs(scipy_bias) / 3

    def test_scipys_bias_grows_as_runs_shorten(self, rng):
        """The property that makes this bias comparisons between runs of different length."""
        short = self._bias(168, np.random.default_rng(2), **SCIPY_DEFAULTS)
        long = self._bias(478, np.random.default_rng(2), **SCIPY_DEFAULTS)
        assert short > long > 0

    def test_the_artefact_is_symmetric(self, rng):
        """filtfilt is zero-phase, so both ends are corrupted -- which is how it is identified."""
        n_timepoints, pad = 478, 2000
        long_series = _ar1(40, n_timepoints + 2 * pad, np.random.default_rng(3))
        reference = snaplab_signal.apply_frequency_filter(
            long_series, FS, **BAND, method='gust')[:, pad:pad + n_timepoints]
        filtered = snaplab_signal.apply_frequency_filter(
            long_series[:, pad:pad + n_timepoints].copy(), FS, **BAND, **SCIPY_DEFAULTS)
        error = np.abs(filtered - reference).mean(axis=0)
        scale = np.abs(reference).mean()
        assert error[0] > 0.5 * scale and error[-1] > 0.5 * scale
        assert error[n_timepoints // 2] < 0.05 * scale


class TestNonFiniteParcels:
    """Parcellations carry unfillable edge-of-mask parcels; Penn_LEAD has 3 of 400.

    ``method='pad'`` tolerates them by propagation. ``'gust'`` solves a linear system and raises
    on the whole array, so without handling this every affected scan filtered with ``'gust'``
    would fail -- and a caller that turns a raise into a skipped scan would lose those subjects
    silently. These pin the behaviour under both methods.
    """

    @staticmethod
    def _data(rng):
        data = rng.standard_normal((4, 300))
        data[1, :] = np.nan          # a wholly non-finite parcel
        data[2, 150] = np.nan        # a single bad sample
        return data

    @pytest.mark.parametrize('method', ['gust', 'pad'])
    def test_non_finite_rows_do_not_raise(self, method, rng):
        out = snaplab_signal.apply_frequency_filter(self._data(rng), FS, **BAND, method=method)
        assert np.isfinite(out[0]).all() and np.isfinite(out[3]).all()
        assert not np.isfinite(out[1]).any() and not np.isfinite(out[2]).any()

    def test_both_methods_agree_on_which_rows_survive(self, rng):
        gust = snaplab_signal.apply_frequency_filter(self._data(rng), FS, **BAND, method='gust')
        pad = snaplab_signal.apply_frequency_filter(self._data(rng), FS, **BAND, method='pad')
        np.testing.assert_array_equal(np.isfinite(gust), np.isfinite(pad))

    def test_the_pad_path_still_matches_historical_propagation(self, rng):
        """Bit-identical to what filtfilt's own NaN propagation produced."""
        data = self._data(rng)
        b, a = sp.signal.butter(2, [0.01 / (FS / 2), 0.08 / (FS / 2)], btype='bandpass')
        historical = np.zeros_like(data)
        for i in range(data.shape[0]):
            historical[i] = sp.signal.filtfilt(b, a, data[i], padtype='constant',
                                               padlen=data.shape[1] - 1)
        got = snaplab_signal.apply_frequency_filter(data, FS, **BAND, **CONSTANT_PAD)
        assert np.array_equal(historical, got, equal_nan=True)

    def test_finite_rows_are_unaffected_by_a_bad_neighbour(self, rng):
        """Filtering around the bad rows must not change the good ones."""
        data = self._data(rng)
        clean = data[[0, 3]].copy()
        out = snaplab_signal.apply_frequency_filter(data, FS, **BAND)
        np.testing.assert_array_equal(
            out[[0, 3]], snaplab_signal.apply_frequency_filter(clean, FS, **BAND))

    def test_an_entirely_non_finite_array_returns_nan_rather_than_raising(self):
        out = snaplab_signal.apply_frequency_filter(np.full((2, 100), np.nan), FS, **BAND)
        assert out.shape == (2, 100) and not np.isfinite(out).any()

    def test_a_one_dimensional_row_is_handled(self, rng):
        assert np.isfinite(
            snaplab_signal.apply_frequency_filter(rng.standard_normal(300), FS, **BAND)).all()
        assert not np.isfinite(
            snaplab_signal.apply_frequency_filter(np.full(300, np.nan), FS, **BAND)).any()


class TestFilterSelection:
    """Unchanged behaviour, pinned so the signature change did not disturb it."""

    def test_bandpass_lowpass_highpass_all_run(self, rng):
        data = rng.standard_normal((4, 400))
        for kwargs in (dict(lowpass=0.08, highpass=0.01), dict(lowpass=0.08),
                       dict(highpass=0.01)):
            out = snaplab_signal.apply_frequency_filter(data, FS, **kwargs)
            assert out.shape == data.shape and np.isfinite(out).all()

    def test_no_cutoff_raises(self, rng):
        with pytest.raises(ValueError, match="At least one of"):
            snaplab_signal.apply_frequency_filter(rng.standard_normal((2, 100)), FS)

    def test_inverted_band_raises(self, rng):
        with pytest.raises(ValueError, match='must be less than'):
            snaplab_signal.apply_frequency_filter(rng.standard_normal((2, 100)), FS,
                                                  lowpass=0.01, highpass=0.08)

    def test_a_lowpass_actually_removes_high_frequencies(self, rng):
        """Sanity that the filter still filters: a fast sinusoid should be attenuated.

        Read away from the ends: odd padding of a sinusoid that stops mid-cycle leaves an edge
        transient, which is TestEdgeArtefact's business, not this test's.
        """
        t = np.arange(600) * TR
        fast = np.sin(2 * np.pi * 0.30 * t)[None, :]
        out = snaplab_signal.apply_frequency_filter(fast, FS, lowpass=0.05)
        assert np.std(out[:, 100:-100]) < 0.05 * np.std(fast)
