import numpy as np
import scipy as sp
from scipy import signal
from statsmodels.tsa.stattools import acf


def compute_acf(ts, nlags=None, version=0):
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
        Lag at which (auto)correlation drops to its smallest absolute value

    """

    ac = acf(ts, nlags=nlags)
    if np.any(ac < 0):
        ac = ac[:np.where(ac < 0)[0][0]]
    
    ac_min = np.argmin(np.abs(ac))
        
    return ac, ac_min

