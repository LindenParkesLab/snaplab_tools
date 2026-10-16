"""Intrinsic neural timescales from parcellated time series.

Two steps, kept separate on purpose. :func:`compute_acf` turns time series into autocorrelation
functions; everything else reduces an ACF to a single number per region -- the *intrinsic
timescale* (INT).

Estimators
    :func:`zero_crossing` (first lag at which the ACF goes negative), :func:`auc` (area under the
    ACF up to a threshold), :func:`decay_time` (first lag below a threshold), and :func:`lag_one`
    (the lag-1 autocorrelation). :func:`compute_timescale` dispatches by name via
    :data:`TIMESCALE_METHODS`, which is what makes it easy to compute several and compare them.

Amplitude-invariant estimators
    Additive white noise multiplies every lag >= 1 of an ACF by the signal fraction
    ``s = var_signal / (var_signal + var_noise)``, so an estimator that reads the ACF's height
    (:func:`auc`, :func:`lag_one`, a fixed threshold) changes with ``s`` even when the dynamics do
    not. :func:`acf_ratio_21` and :func:`tau_ratio_21` (a ratio of lags, in which ``s`` cancels)
    and :func:`tau_expfit_free` (an exponential with a free amplitude over a fixed window of lags)
    read only the curve's shape.

    **Neither zero crossing is amplitude-invariant** -- not :func:`zero_crossing` and not
    :func:`zero_crossing_interp`. A threshold of zero looks scale-free, but a sample ACF crosses
    zero where the signal's decaying correlation meets the estimator's own negative mean-removal
    bias and sampling noise, and neither of those shrinks with ``s``; lower ``s``, earlier crossing.

    :func:`tau_ratio_21` is NaN wherever the ratio is outside (0, 1), and the cells it drops are the
    long-tau tail, increasingly so as noise grows. Aggregate the ratio and convert the aggregate
    (``-tr / ln(median ratio)``); a median of per-cell tau is biased short.

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
import scipy as sp
from joblib import Parallel, delayed, effective_n_jobs

__all__ = [
    'compute_acf',
    'zero_crossing',
    'zero_crossing_interp',
    'auc',
    'decay_time',
    'lag_one',
    'acf_ratio_21',
    'tau_ratio_21',
    'tau_expfit_free',
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


def zero_crossing_interp(acf, tr=1.0):
    """First zero crossing of the ACF, refined by linear interpolation between lags.

    :func:`zero_crossing` reports the first lag ``k`` at which the ACF is negative, so it moves in
    steps of one TR. This places the crossing on the straight line between lags ``k - 1`` (the last
    non-negative value) and ``k``, which gives sub-TR resolution. The result always lies in
    ``[(k - 1) * tr, k * tr)``, i.e. within one TR below :func:`zero_crossing`.

    Parameters
    ----------
    acf : (n_lags,) or (n_regions, n_lags) array-like
        Autocorrelation function(s), as returned by :func:`compute_acf`.
    tr : float
        Repetition time in seconds. The default of 1.0 returns lags.

    Returns
    -------
    float or (n_regions,) ndarray
        Interpolated crossing time. NaN where the ACF never goes negative, as for
        :func:`zero_crossing`.
    """
    acf, was_1d = _as_2d(acf)
    first = _first_below(acf, 0.0)
    out = np.full(acf.shape[0], np.nan)

    out[first == 0] = 0.0      # lag 0 already negative: not a real ACF, but keep zero_crossing's answer
    rows = np.flatnonzero(first >= 1)
    if rows.size:
        k = first[rows].astype(int)
        before, after = acf[rows, k - 1], acf[rows, k]     # before >= 0 > after, so the step is > 0
        out[rows] = (k - 1 + before / (before - after)) * tr
    return _unwrap(out, was_1d)


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


def acf_ratio_21(acf, tr=1.0):
    """The ratio of the lag-2 to the lag-1 autocorrelation, ``ACF(2) / ACF(1)``.

    Scaling every lag >= 1 by the same factor -- which is what additive white noise does to an ACF
    -- cancels in the ratio. For an AR(1) process it equals the AR coefficient.

    Parameters
    ----------
    acf : (n_lags,) or (n_regions, n_lags) array-like
        Autocorrelation function(s).
    tr : float
        Accepted for symmetry with the other estimators and ignored, as in :func:`lag_one`: a ratio
        of correlations is not a time.

    Returns
    -------
    float or (n_regions,) ndarray
        The ratio. NaN where ``ACF(1) <= 0`` (or is NaN), since the ratio's sign and size are then
        not a decay rate.
    """
    acf, was_1d = _as_2d(acf)
    if acf.shape[1] < 3:
        raise ValueError('need at least 3 lags to form ACF(2) / ACF(1)')
    lag1, lag2 = acf[:, 1], acf[:, 2]
    out = np.full(acf.shape[0], np.nan)
    ok = lag1 > 0                                   # NaN compares False, so NaN rows stay NaN
    out[ok] = lag2[ok] / lag1[ok]
    return _unwrap(out, was_1d)


def tau_ratio_21(acf, tr=1.0):
    """Exponential decay time implied by :func:`acf_ratio_21`: ``-tr / ln(ACF(2) / ACF(1))``.

    If ``ACF(k) = A * exp(-k * tr / tau)`` for ``k >= 1``, the ratio of successive lags is
    ``exp(-tr / tau)`` whatever ``A`` is, so this recovers ``tau`` from two lags without reading the
    curve's height. For an AR(1) process with coefficient ``phi`` it is ``-tr / ln(phi)``.

    Parameters
    ----------
    acf : (n_lags,) or (n_regions, n_lags) array-like
        Autocorrelation function(s).
    tr : float
        Repetition time in seconds.

    Returns
    -------
    float or (n_regions,) ndarray
        Decay time in seconds. NaN unless ``0 < ACF(2) / ACF(1) < 1``: a ratio at or above 1 is not
        a decay, and a ratio at or below 0 has no logarithm.
    """
    acf, was_1d = _as_2d(acf)
    ratio = np.atleast_1d(acf_ratio_21(acf))
    out = np.full(acf.shape[0], np.nan)
    ok = (ratio > 0) & (ratio < 1)
    out[ok] = -tr / np.log(ratio[ok])
    return _unwrap(out, was_1d)


#: Limits of :func:`tau_expfit_free`. The tau bounds are multiples of ``tr``.
_EXPFIT_A_BOUNDS = (0.0, 10.0)
_EXPFIT_TAU_BOUNDS = (0.01, 1000.0)
_EXPFIT_MAXFEV = 2000
#: Rows per joblib task; below this a stack is fitted in-process, since spawning workers costs more
#: than the fits. 25 keeps a 400-parcel scan on 16 workers: 70 ms against 297 ms at 200 rows and 568 ms
#: serial, measured on HCP-YA WM residual ACFs (about 1.4 ms per row).
_EXPFIT_ROWS_PER_JOB = 25


def _exp_decay(t, amplitude, tau):
    return amplitude * np.exp(-t / tau)


def _expfit_rows(acf, tr, max_lag):
    """Fit lags 1..max_lag of every row in turn. Returns (tau, amplitude, status), one entry per row."""
    n_rows = acf.shape[0]
    tau, amplitude = np.full(n_rows, np.nan), np.full(n_rows, np.nan)
    status = np.ones(n_rows, dtype=np.int8)                  # 1 until a fit is attempted
    lags = np.arange(1, max_lag + 1)
    tau_start = np.atleast_1d(tau_ratio_21(acf, tr))
    tau_lo, tau_hi = _EXPFIT_TAU_BOUNDS[0] * tr, _EXPFIT_TAU_BOUNDS[1] * tr
    a_lo, a_hi = _EXPFIT_A_BOUNDS

    for i in range(n_rows):
        values = acf[i, lags]
        if not np.isfinite(values).all():
            continue
        tau0 = tau_start[i] if np.isfinite(tau_start[i]) else 2.0 * tr
        tau0 = float(np.clip(tau0, tau_lo * 1.01, tau_hi * 0.99))
        a0 = float(np.clip(values[0] * np.exp(tr / tau0), 1e-3, a_hi * 0.99))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')     # covariance warnings; the estimate is what is kept
                (a_fit, tau_fit), _ = sp.optimize.curve_fit(
                    _exp_decay, lags * tr, values, p0=(a0, tau0),
                    bounds=([a_lo, tau_lo], [a_hi, tau_hi]), maxfev=_EXPFIT_MAXFEV)
        except (RuntimeError, ValueError):
            status[i] = 2
            continue
        if not (np.isfinite(a_fit) and np.isfinite(tau_fit)):
            status[i] = 2
            continue
        amplitude[i] = a_fit
        if tau_fit <= tau_lo * (1 + 1e-6):
            tau[i], status[i] = tau_lo, 3
        elif tau_fit >= tau_hi * (1 - 1e-6):
            tau[i], status[i] = tau_hi, 3
        else:
            tau[i] = tau_fit
            status[i] = 4 if (a_fit <= a_lo + 1e-9 or a_fit >= a_hi * (1 - 1e-6)) else 0
    return tau, amplitude, status


def tau_expfit_free(acf, tr=1.0, max_lag=4, n_jobs=-1, return_status=False):
    """Decay time of an exponential with a free amplitude, ``A * exp(-k * tr / tau)``, fitted to lags 1..K.

    Lag 0 is left out and ``A`` is fitted, so a uniform rescaling of lags >= 1 (additive white
    noise) changes ``A`` and leaves ``tau`` alone. The window is a fixed number of lags rather than
    "until the ACF falls below a threshold": a threshold on the ACF's height is itself an amplitude
    rule, and would shorten the window as the signal fraction falls.

    Parameters
    ----------
    acf : (n_lags,) or (n_regions, n_lags) array-like
        Autocorrelation function(s). Only lags 0..`max_lag` are read.
    tr : float
        Repetition time in seconds.
    max_lag : int
        K, the last lag fitted. At least 2 (two lags for two parameters); 4 by default.
    n_jobs : int
        Workers for :class:`joblib.Parallel`, over blocks of rows. A stack of fewer than a few
        hundred rows is fitted in-process regardless.
    return_status : bool
        Also return the fitted amplitude and a per-row status code.

    Returns
    -------
    tau : float or (n_regions,) ndarray
        Decay time in seconds. A fit that ends on a tau bound returns that bound (``0.01 * tr`` or
        ``1000 * tr``), so a median treats it as censored rather than missing. NaN when a lag in the
        window is non-finite, and when the fit raises or returns a non-finite value.
    amplitude, status : (n_regions,) ndarray
        Only with ``return_status=True``. Status: 0 fitted; 1 non-finite values in the window;
        2 the fit failed (tau NaN); 3 tau ended on a bound (tau set to that bound); 4 ``A`` ended on
        a bound (tau kept).

    Notes
    -----
    :func:`scipy.optimize.curve_fit` (trust-region reflective) with ``A`` in [0, 10], ``tau`` in
    [0.01, 1000] x ``tr``, ``maxfev=2000``. The starting ``tau`` is :func:`tau_ratio_21` where that
    is defined and ``2 * tr`` otherwise; the starting ``A`` puts the curve through ``ACF(1)``.
    """
    acf, was_1d = _as_2d(acf)
    if max_lag < 2:
        raise ValueError(f'max_lag must be at least 2 to fit two parameters, got {max_lag}')
    if acf.shape[1] <= max_lag:
        raise ValueError(f'need lags 0..{max_lag}, got {acf.shape[1]} lags')
    acf = acf[:, :max_lag + 1]
    n_blocks = min(effective_n_jobs(n_jobs), max(1, acf.shape[0] // _EXPFIT_ROWS_PER_JOB))
    if n_blocks == 1:
        tau, amplitude, status = _expfit_rows(acf, tr, max_lag)
    else:
        parts = Parallel(n_jobs=n_blocks)(
            delayed(_expfit_rows)(block, tr, max_lag) for block in np.array_split(acf, n_blocks))
        tau, amplitude, status = (np.concatenate(p) for p in zip(*parts))
    if return_status:
        return _unwrap(tau, was_1d), amplitude, status
    return _unwrap(tau, was_1d)


#: Estimators whose output is not a time and so does not scale with TR. The units warning in
#: :func:`compute_timescale` would be meaningless for these.
_UNITLESS_METHODS = frozenset({'lag_one', 'acf_ratio_21'})

#: Named INT estimators, each callable as ``fn(acf, tr)``. Iterate over it to compute several
#: measures from one set of ACFs; the keys are stable and safe to use in output filenames.
TIMESCALE_METHODS = {
    'zero_crossing':     zero_crossing,
    'auc_zero_crossing': lambda acf, tr=1.0: auc(acf, tr, threshold=0.0),
    'auc_half_crossing': lambda acf, tr=1.0: auc(acf, tr, threshold=0.5),
    'decay_time':        decay_time,
    'decay_time_half':   lambda acf, tr=1.0: decay_time(acf, tr, threshold=0.5),
    'lag_one':           lag_one,
    # Amplitude-invariant (see the module docstring).
    'zero_crossing_interp': zero_crossing_interp,
    'acf_ratio_21':         acf_ratio_21,
    'tau_ratio_21':         tau_ratio_21,
    'tau_expfit_free':      tau_expfit_free,
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