"""Signal-processing utilities.

Currently one function, :func:`apply_frequency_filter`, which applies a zero-phase Butterworth
lowpass, highpass, or bandpass filter to parcellated time series. Filtering to the canonical
resting-state band (roughly 0.01-0.1 Hz) before computing autocorrelation or functional
connectivity in :mod:`snaplab_tools.derivs` is the usual reason to reach for it.

``method``, ``padtype`` and ``padlen`` are all exposed. ``padtype`` and ``padlen`` apply only when
``method='pad'``; scipy ignores them under ``'gust'``. Pass
``method='pad', padtype='odd', padlen=15`` to reproduce scipy's behaviour exactly.
"""
import numpy as np
import scipy as sp

__all__ = ['apply_frequency_filter', 'resolve_padlen', 'DEFAULT_METHOD', 'DEFAULT_PADTYPE']

#: Edge-handling method used by :func:`apply_frequency_filter`: scipy's ``'pad'``, which extends
#: the record at both ends.
DEFAULT_METHOD = 'pad'

#: Padding mode used when ``method='pad'``: ``'odd'``, as scipy. The padding *length* is not
#: scipy's 15 samples but ``n_timepoints - 1`` (:func:`resolve_padlen`). Ignored under
#: ``method='gust'``, which does not pad.
DEFAULT_PADTYPE = 'odd'

def resolve_padlen(n_timepoints, padlen=None):
    """The padding length :func:`apply_frequency_filter` will use for a run of `n_timepoints`.

    Exposed so a caller can record in its own metadata what padding produced a cached result.
    Reading it from here rather than restating ``n - 1`` means a change to the default cannot
    leave a consumer describing files it did not produce.

    Parameters
    ----------
    n_timepoints : int
        Length of the axis being filtered.
    padlen : int or None
        Explicit padding length, or None for the default. The default is ``n_timepoints - 1``,
        the largest value ``scipy.signal.filtfilt`` accepts: it requires the input to be strictly
        longer than the padding.

    Returns
    -------
    int
        Padding length in samples.

    Raises
    ------
    ValueError
        If `padlen` is negative, or is not shorter than `n_timepoints` -- which scipy would
        reject with a less specific message.

    Examples
    --------
    >>> resolve_padlen(478)
    477
    >>> resolve_padlen(478, padlen=15)
    15
    """
    n_timepoints = int(n_timepoints)
    if padlen is None:
        return max(n_timepoints - 1, 0)
    padlen = int(padlen)
    if padlen < 0:
        raise ValueError(f"padlen must be non-negative, got {padlen}.")
    if padlen >= n_timepoints:
        raise ValueError(
            f"padlen ({padlen}) must be shorter than the data ({n_timepoints} samples); "
            "filtfilt requires strictly more data than padding."
        )
    return padlen


def apply_frequency_filter(data, sampling_freq, lowpass=None, highpass=None, order=2,
                           method=DEFAULT_METHOD, padtype=DEFAULT_PADTYPE, padlen=None):
    """Apply a Butterworth frequency filter (lowpass, highpass, or bandpass) to data.

    Uses a zero-phase Butterworth filter (``filtfilt``). Providing both lowpass and highpass
    gives a bandpass filter; at least one must be specified.

    Parameters
    ----------
    data : ndarray
        Data with shape (n_regions, n_timepoints). Filtering is along the last axis.
    sampling_freq : float
        Sampling frequency in Hz.
    lowpass : float or None
        Lowpass cutoff frequency in Hz. Frequencies above this are attenuated.
    highpass : float or None
        Highpass cutoff frequency in Hz. Frequencies below this are attenuated.
    order : int
        Order of the Butterworth filter. Note a bandpass of `order` is ``2 * order`` overall.
    method : {'pad', 'gust'}
        Edge handling. Defaults to :data:`DEFAULT_METHOD` (``'pad'``). ``'gust'`` (Gustafsson's
        method) solves for exact initial conditions rather than extrapolating data that was
        never observed. See the module docstring.
    padtype : {'constant', 'odd', 'even', None}
        Extension used at both ends. **Only applies when ``method='pad'``**; scipy ignores it
        under ``'gust'``. Defaults to :data:`DEFAULT_PADTYPE` (``'odd'``).
    padlen : int or None
        Padding length, again only under ``method='pad'``. None uses :func:`resolve_padlen`,
        i.e. ``n_timepoints - 1``, rather than scipy's 15-sample default.

    Returns
    -------
    ndarray
        Filtered data, same shape and dtype as the input. Any row containing a non-finite
        sample is returned all-NaN rather than raising -- see Notes.

    Notes
    -----
    The output dtype follows the input, as it always has: the result is written into an array
    shaped like `data`. Filtering a float32 array therefore returns float32.

    Rows with non-finite samples are filtered around rather than through. ``method='pad'``
    tolerated them by propagation; ``'gust'`` solves a linear system and would raise on the whole
    array. Both now return an all-NaN row for such a region, so a parcellation with a few
    unfillable parcels behaves the same under either method.
    """
    if lowpass is None and highpass is None:
        raise ValueError("At least one of 'lowpass' or 'highpass' must be specified.")

    nyquist = sampling_freq / 2

    if lowpass is not None and highpass is not None:
        btype = 'bandpass'
        cutoff = [highpass / nyquist, lowpass / nyquist]
        if cutoff[0] >= cutoff[1]:
            raise ValueError(
                f"highpass ({highpass} Hz) must be less than lowpass ({lowpass} Hz) "
                "for a bandpass filter."
            )
    elif lowpass is not None:
        btype = 'low'
        cutoff = lowpass / nyquist
    else:
        btype = 'high'
        cutoff = highpass / nyquist

    # Warn if any cutoff is at or beyond Nyquist
    for label, val in [('lowpass', lowpass), ('highpass', highpass)]:
        if val is not None and val / nyquist >= 1.0:
            print(
                f"Warning: {label} cutoff {val} Hz is >= Nyquist frequency {nyquist} Hz."
            )

    b, a = sp.signal.butter(order, cutoff, btype=btype)

    def _run(block):
        # One vectorised call rather than a Python loop over regions: filtfilt applies the same
        # 1-D operation along `axis`.
        if method == 'gust':
            # padtype/padlen are not passed at all: Gustafsson's method does not pad, and scipy
            # would ignore them. Passing them anyway would imply a padding this never used.
            return sp.signal.filtfilt(b, a, block, axis=-1, method='gust')
        return sp.signal.filtfilt(
            b, a, block, axis=-1, method=method, padtype=padtype,
            padlen=resolve_padlen(np.shape(block)[-1], padlen),
        )

    # Writing into zeros_like keeps the input's dtype and memory order, as the loop did.
    filtered_data = np.zeros_like(data)
    data = np.asarray(data)

    finite_rows = np.isfinite(data).all(axis=-1)
    if finite_rows.ndim == 0:                       # 1-D input: one row, all or nothing
        filtered_data[...] = _run(data) if finite_rows else np.nan
    elif finite_rows.all():
        filtered_data[...] = _run(data)
    else:
        filtered_data[...] = np.nan
        if finite_rows.any():
            filtered_data[finite_rows] = _run(data[finite_rows])

    return filtered_data
