"""Derived measures computed from regional time series.

Both functions here take a parcellated fMRI time series and reduce it to something you can
correlate against other brain maps: :func:`compute_acf` gives the autocorrelation function of a
single region (fit :func:`snaplab_tools.utils.exp_decay` to it to get an intrinsic timescale), and
:func:`compute_fc` gives a Fisher-z functional connectivity matrix across regions.
"""
import numpy as np
import scipy as sp
from scipy import signal
from statsmodels.tsa.stattools import acf

__all__ = ['compute_acf', 'compute_fc']


def compute_acf(ts, nlags=None):
    """ Calculate the signal autocorrelation (lagged correlation)

    Parameters
    ----------
    ts : np.array (n_timepoints,)
        time series
    nlags : int (default=None)
        Number of lags to compute acf over.

    Returns
    -------
    ac : np.array (n_timepoints,)
        Time lagged (auto)correlation.
    ac_min : int
        Lag at which (auto)correlation drops to its smallest (positive) value

    """

    # compute auto correlation function using statsmodels
    ac = acf(ts, nlags=nlags)
    
    # trim from first time ac goes values, if at all
    if np.any(ac < 0):
        ac = ac[:np.where(ac < 0)[0][0]]
    
    # get index of smallest value
    ac_min = np.argmin(np.abs(ac))
        
    return ac, ac_min


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

