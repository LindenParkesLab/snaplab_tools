"""Derived measures computed from regional time series.

:func:`compute_fc` takes a parcellated fMRI time series and reduces it to a Fisher-z functional
connectivity matrix across regions.

Autocorrelation lives in :mod:`snaplab_tools.timescales` instead. This module used to carry its
own ``compute_acf``, which truncated the ACF at its first negative value -- fine for fitting a
decay to the leading edge, but it silently removes the zero crossing that most intrinsic-timescale
estimators are looking for.
"""
import numpy as np

__all__ = ['compute_fc']


def compute_fc(ts):
    """
    Parameters
    ----------
    ts : np.array (n_parcels, n_timepoints)
        time series

    Returns
    -------
    fc : np.array (n_parcels, n_parcels)
        functional connectivity matrix
    """

    fc = np.corrcoef(ts, rowvar=True)
    np.fill_diagonal(fc, np.nan)
    fc = np.arctanh(fc)
    np.fill_diagonal(fc, 1)

    return fc

