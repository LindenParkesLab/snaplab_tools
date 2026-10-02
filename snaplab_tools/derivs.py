"""Derived measures computed from regional time series.

:func:`compute_fc` takes a parcellated fMRI time series and reduces it to a Fisher-z functional
connectivity matrix across regions. :func:`compute_brain_states` clusters its time points into recurrent
patterns of activity ("brain states") with k-means.

Autocorrelation lives in :mod:`snaplab_tools.timescales` instead. This module used to carry its
own ``compute_acf``, which truncated the ACF at its first negative value -- fine for fitting a
decay to the leading edge, but it silently removes the zero crossing that most intrinsic-timescale
estimators are looking for.
"""
import warnings

import numpy as np
import scipy as sp
from sklearn.cluster import KMeans

__all__ = ['compute_fc', 'compute_brain_states']


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


def compute_brain_states(ts, n_states, zscore=True, random_state=0, n_init=10):
    """Find recurrent patterns of activity ("brain states") by k-means clustering of time points.

    Each time point is a pattern of activity across regions; k-means groups similar patterns, and
    the cluster centroids are the brain states. To pool runs or subjects, concatenate their time
    series along time first, e.g. ``np.concatenate(runs, axis=1)``.

    Parameters
    ----------
    ts : np.array (n_regions, n_timepoints)
        Time series.
    n_states : int
        Number of states (clusters). A common way to choose it is to compare the returned
        ``inertia`` across a range of values and look for the point where it stops falling steeply.
    zscore : bool
        Z-score each region across time before clustering, so that regions contribute on the same
        scale. Default True.
    random_state : int
        Seed for k-means' initialisation. Default 0.
    n_init : int
        Number of k-means runs with different initialisations; the best is kept. Default 10. It is
        set explicitly because scikit-learn's own default changed in version 1.4, which would
        otherwise make results depend on the installed version.

    Returns
    -------
    centroids : np.array (n_states, n_regions)
        The brain states: each cluster's mean pattern, in the units clustered (z-scores by default).
        Regions left out of clustering (see Notes) are NaN.
    labels : np.array (n_timepoints,)
        The state each time point was assigned to.
    inertia : float
        Sum of squared distances from each time point to its centroid.

        scikit-learn's k-means sums across threads, so repeated runs can differ in the last digits
        of ``centroids`` and ``inertia``.

    Notes
    -----
    A region with any non-finite value, or a constant region when ``zscore=True``, cannot be
    clustered. Such regions are left out, and are NaN in ``centroids``, rather than making the whole
    call fail: parcellations routinely carry a few unusable parcels.
    """
    ts = np.asarray(ts, dtype=float)
    frames = ts.T  # (n_timepoints, n_regions): one sample per time point
    if zscore:
        # a constant region z-scores to NaN, with warnings; such regions are handled below
        with np.errstate(invalid='ignore', divide='ignore'), warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            frames = sp.stats.zscore(frames, axis=0)
    usable = np.all(np.isfinite(frames), axis=0)
    if not usable.any():
        raise ValueError("No region has finite values at every time point; nothing to cluster.")

    kmeans = KMeans(n_clusters=n_states, random_state=random_state, n_init=n_init).fit(frames[:, usable])
    centroids = np.full((n_states, ts.shape[0]), np.nan)
    centroids[:, usable] = kmeans.cluster_centers_

    return centroids, kmeans.labels_, float(kmeans.inertia_)
