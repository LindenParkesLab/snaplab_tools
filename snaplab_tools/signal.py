"""Signal-processing utilities."""
import numpy as np
from scipy.signal import butter, filtfilt


def apply_frequency_filter(data, sampling_freq, lowpass=None, highpass=None, order=2):
    """Apply a Butterworth frequency filter (lowpass, highpass, or bandpass) to data.

    Uses a zero-phase Butterworth filter (filtfilt), applied per row. Providing both
    lowpass and highpass gives a bandpass filter; at least one must be specified.

    Parameters
    ----------
    data : ndarray
        Data with shape (n_regions, n_timepoints).
    sampling_freq : float
        Sampling frequency in Hz.
    lowpass : float or None
        Lowpass cutoff frequency in Hz. Frequencies above this are attenuated.
    highpass : float or None
        Highpass cutoff frequency in Hz. Frequencies below this are attenuated.
    order : int
        Order of the Butterworth filter.

    Returns
    -------
    ndarray
        Filtered data, same shape as the input.
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

    b, a = butter(order, cutoff, btype=btype)

    n_regions = data.shape[0]
    filtered_data = np.zeros_like(data)
    for region_idx in range(n_regions):
        filtered_data[region_idx, :] = filtfilt(b, a, data[region_idx, :])

    return filtered_data
