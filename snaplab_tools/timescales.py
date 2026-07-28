"""Intrinsic neural timescales from parcellated time series.

Two steps, kept separate on purpose. :func:`compute_acf` turns time series into autocorrelation
functions; everything else reduces an ACF to a single number per region -- the *intrinsic
timescale* (INT).

Estimators
    :func:`zero_crossing` (first lag at which the ACF goes negative), :func:`auc` (area under the
    ACF up to a threshold), :func:`decay_time` (first lag below a threshold), and :func:`lag_one`
    (the lag-1 autocorrelation). :func:`compute_timescale` dispatches by name via
    :data:`TIMESCALE_METHODS`, which is what makes it easy to compute several and compare them.

They disagree, and that is the point: an ACF is a curve, and every INT measure is a different
scalar summary of it. A map built with one estimator is not interchangeable with a map built with
another, so say which you used.

Every estimator accepts either a single ACF ``(n_lags,)`` or a stack ``(n_regions, n_lags)``, and
returns a scalar or a ``(n_regions,)`` vector to match. All of them except :func:`lag_one` take
``tr`` and return **seconds**; pass ``tr=1.0`` (the default) to get answers in lags instead --
:func:`compute_timescale` warns when you do, since a timescale reported in seconds that was
actually computed in TRs is wrong by a factor of the TR.
"""
import warnings

import numpy as np

__all__ = [
    'compute_acf',
    'zero_crossing',
    'auc',
    'decay_time',
    'lag_one',
    'compute_timescale',
    'TIMESCALE_METHODS',
]


def _as_2d(acf):
    """Return (2-D view of `acf`, whether the caller passed a single ACF)."""
    acf = np.asarray(acf, dtype=float)
    if acf.ndim == 1:
        return acf[np.newaxis, :], True
    if acf.ndim != 2:
        raise ValueError(f'acf must be 1-D or 2-D, got {acf.ndim}-D')
    return acf, False


def _unwrap(values, was_1d):
    return float(values[0]) if was_1d else values


def _first_below(acf, threshold):
    """Index of the first lag at which each ACF drops below `threshold`; NaN where it never does.

    The shared core of :func:`zero_crossing` and :func:`decay_time`, which differ only in the
    threshold. NaN entries never compare below the threshold, so an all-NaN ACF yields NaN rather
    than a spurious lag 0.
    """
    below = acf < threshold
    found = below.any(axis=1)
    index = np.argmax(below, axis=1).astype(float)   # argmax gives the first True
    index[~found] = np.nan
    return index


def compute_acf(data, use_fft=False, nlags=None):
    """Autocorrelation function of one or more time series.

    Each series is z-scored, correlated with itself at every non-negative lag, and normalised by
    its lag-0 value, so the ACF starts at 1.

    Parameters
    ----------
    data : (n_timepoints,) or (n_regions, n_timepoints) array-like
        Time series. A single series returns a single ACF.
    use_fft : bool
        Compute via FFT rather than direct correlation. Mathematically identical (agreement to
        ~1e-16) and much faster for long series; the direct path is kept because it is obvious.
    nlags : int or None
        Truncate the result to lags ``0..nlags`` inclusive. ``None`` returns every lag, which for
        a long scan is a large array you rarely need in full -- a timescale is determined by the
        first part of the curve.

    Returns
    -------
    ndarray
        ``(n_lags,)`` or ``(n_regions, n_lags)`` to match the input.

    Notes
    -----
    A series with zero variance, or one containing NaN, has no well-defined ACF and comes back as
    all-NaN rather than raising -- one bad region should not abort a whole scan. Check with
    ``np.isnan(acf).all(axis=-1)``.

    This is the *biased* estimator: every lag is divided by the same lag-0 value rather than by
    the number of overlapping samples, so the ACF is damped towards zero at long lags. That is the
    convention the INT literature uses, and it is what the estimators here assume.
    """
    series = np.asarray(data, dtype=float)
    single = series.ndim == 1
    if single:
        series = series[np.newaxis, :]
    if series.ndim != 2:
        raise ValueError(f'data must be 1-D or 2-D, got {series.ndim}-D')

    n_series, n_timepoints = series.shape
    if n_timepoints < 2:
        raise ValueError('need at least 2 timepoints to compute an autocorrelation')

    out = np.full((n_series, n_timepoints), np.nan)

    # Series that cannot be standardized (flat, or carrying NaN) are left as NaN. np.std returns
    # NaN for a series containing NaN, and NaN > 0 is False, so both cases fall out here.
    std = np.std(series, axis=1)
    valid = std > 0
    if valid.any():
        z = (series[valid] - series[valid].mean(axis=1, keepdims=True)) / std[valid, np.newaxis]

        if use_fft:
            # Zero-pad past 2n-1 so the circular correlation the FFT computes has no wraparound,
            # and to a power of two because that is the length the transform likes.
            size = 2 ** (int(np.log2(2 * n_timepoints - 1)) + 1)
            padded = np.zeros((z.shape[0], size))
            padded[:, :n_timepoints] = z
            spectrum = np.fft.rfft(padded, axis=1)
            acf = np.fft.irfft(spectrum * np.conj(spectrum), axis=1)[:, :n_timepoints]
        else:
            acf = np.array([np.correlate(row, row, mode='full')[n_timepoints - 1:] for row in z])

        out[valid] = acf / acf[:, :1]

    if nlags is not None:
        if nlags < 0:
            raise ValueError(f'nlags must be non-negative, got {nlags}')
        out = out[:, :nlags + 1]

    return out[0] if single else out


def zero_crossing(acf, tr=1.0):
    """First lag at which the ACF goes negative.

    The simplest INT estimator. It makes no assumption about the shape of the
    decay, only that the curve eventually crosses zero.

    Parameters
    ----------
    acf : (n_lags,) or (n_regions, n_lags) array-like
        Autocorrelation function(s), as returned by :func:`compute_acf`.
    tr : float
        Repetition time in seconds. The default of 1.0 returns lags.

    Returns
    -------
    float or (n_regions,) ndarray
        Time of the first negative value. NaN for a region whose ACF never goes negative -- which
        happens when the scan is too short for the ACF to decay, and is a signal that this
        estimator is not usable for that region rather than a value to fill in.
    """
    acf, was_1d = _as_2d(acf)
    return _unwrap(_first_below(acf, 0.0) * tr, was_1d)


def decay_time(acf, tr=1.0, threshold=1.0 / np.e):
    """First lag at which the ACF drops below `threshold`.

    With the default threshold this is the classic decay-time definition: the lag at which the
    autocorrelation has fallen to 1/e of its starting value.

    Parameters
    ----------
    acf : (n_lags,) or (n_regions, n_lags) array-like
        Autocorrelation function(s).
    tr : float
        Repetition time in seconds.
    threshold : float
        ACF value to cross. ``1/e`` by default; 0.5 (a half-life) is the other common choice.

    Returns
    -------
    float or (n_regions,) ndarray
        Time of the first value below `threshold`, or NaN where the ACF never falls that far.
    """
    acf, was_1d = _as_2d(acf)
    return _unwrap(_first_below(acf, threshold) * tr, was_1d)


def auc(acf, tr=1.0, threshold=0.0, max_lag=None):
    """Area under the ACF, integrated up to a threshold crossing or a fixed lag.

    Parameters
    ----------
    acf : (n_lags,) or (n_regions, n_lags) array-like
        Autocorrelation function(s).
    tr : float
        Repetition time in seconds; used as the integration step, so the result is in seconds.
    threshold : float
        Integrate up to the first lag below this value. Ignored when `max_lag` is given.
    max_lag : int or None
        Integrate to this lag index instead, the same for every region. Use it when regions must
        be compared over an identical window.

    Returns
    -------
    float or (n_regions,) ndarray
        Trapezoidal area. NaN for a region whose ACF never crosses `threshold`, when no `max_lag`
        is given -- there is no defensible endpoint in that case.

    Notes
    -----
    Integrated with the trapezoidal rule at ``dx=tr``. A left Riemann sum -- the obvious
    implementation, and the one this replaces -- overestimates the area by about ``tr/2``, which
    is a TR-dependent bias and therefore not comparable across datasets acquired at different TRs.
    """
    acf, was_1d = _as_2d(acf)
    areas = np.full(acf.shape[0], np.nan)

    if max_lag is None:
        endpoints = _first_below(acf, threshold)
    else:
        endpoints = np.full(acf.shape[0], float(max_lag))

    for i, endpoint in enumerate(endpoints):
        if np.isnan(endpoint):
            continue
        areas[i] = np.trapezoid(acf[i, :int(endpoint)], dx=tr)

    return _unwrap(areas, was_1d)


def lag_one(acf, tr=1.0):
    """The lag-1 autocorrelation.

    Parameters
    ----------
    acf : (n_lags,) or (n_regions, n_lags) array-like
        Autocorrelation function(s).
    tr : float
        Accepted for symmetry with the other estimators and ignored -- a correlation is not a
        time, so it does not scale with TR. Two regions with the same lag-1 value in scans of
        different TR do *not* have the same timescale, which is the main reason to prefer one of
        the others when you can.

    Returns
    -------
    float or (n_regions,) ndarray
        The value of the ACF at lag 1.
    """
    acf, was_1d = _as_2d(acf)
    if acf.shape[1] < 2:
        raise ValueError('need at least 2 lags to read the lag-1 autocorrelation')
    return _unwrap(acf[:, 1], was_1d)


#: Estimators whose output is not a time and so does not scale with TR. The units warning in
#: :func:`compute_timescale` would be meaningless for these.
_UNITLESS_METHODS = frozenset({'lag_one'})

#: Named INT estimators, each callable as ``fn(acf, tr)``. Iterate over it to compute several
#: measures from one set of ACFs; the keys are stable and safe to use in output filenames.
TIMESCALE_METHODS = {
    'zero_crossing':     zero_crossing,
    'auc_zero_crossing': lambda acf, tr=1.0: auc(acf, tr, threshold=0.0),
    'auc_half_crossing': lambda acf, tr=1.0: auc(acf, tr, threshold=0.5),
    'decay_time':        decay_time,
    'decay_time_half':   lambda acf, tr=1.0: decay_time(acf, tr, threshold=0.5),
    'lag_one':           lag_one,
}


def compute_timescale(acf, tr=1.0, method='zero_crossing', verbose=True):
    """Compute an intrinsic timescale by name.

    Parameters
    ----------
    acf : (n_lags,) or (n_regions, n_lags) array-like
        Autocorrelation function(s), as returned by :func:`compute_acf`.
    tr : float
        Repetition time in seconds.
    method : str
        A key of :data:`TIMESCALE_METHODS`.
    verbose : bool
        Emit the units warning described below. Set it to False once you have read the warning and
        decided -- because your TR really is 1.0 s, or because lags are the unit you want. It
        silences nothing else, so a False here is a statement about units and not a general mute.

    Returns
    -------
    float or (n_regions,) ndarray

    Warns
    -----
    UserWarning
        When ``tr`` is left at its default of 1.0 and ``verbose`` is True. Every estimator here
        counts lags and multiplies by ``tr``, so ``tr=1.0`` returns TRs, not seconds -- 14 lags
        stay 14. That is a perfectly good unit as long as it is the one you meant, but it is not
        comparable across datasets acquired at different TRs, and a timescale reported in seconds
        that was silently computed in TRs is wrong by a factor of the TR. Pass your acquisition's
        TR to get seconds.

        Not raised for estimators that do not return a time (:func:`lag_one`), or when ``tr`` is
        anything other than 1.0. A genuine 1.0 s acquisition cannot be distinguished from the
        default, so it warns too -- pass ``verbose=False`` to say you meant it.

    Examples
    --------
    >>> acf = compute_acf(timeseries, use_fft=True, nlags=200)
    >>> ints = {m: compute_timescale(acf, tr=0.72, method=m) for m in TIMESCALE_METHODS}
    """
    if method not in TIMESCALE_METHODS:
        raise ValueError(
            f'unknown timescale method {method!r}; available: {sorted(TIMESCALE_METHODS)}'
        )

    if verbose and tr == 1.0 and method not in _UNITLESS_METHODS:
        warnings.warn(
            f"compute_timescale(method={method!r}) was called with tr=1.0, the default, so the "
            f"result is in TRs rather than seconds -- multiplying a lag count by 1.0 leaves it a "
            f"lag count. Pass your acquisition's TR (e.g. tr=0.72) for seconds, or verbose=False "
            f"if your TR really is 1.0 s or lags are the unit you want.",
            UserWarning,
            stacklevel=2,
        )

    return TIMESCALE_METHODS[method](acf, tr)