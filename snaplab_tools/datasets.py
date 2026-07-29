"""Synthetic data on real cortical geometry, for tutorials, examples, and tests.

Every generator here produces made-up numbers, but wherever geometry matters those numbers sit on
the *real* Schaefer parcellation shipped in :mod:`snaplab_tools.nulls` -- actual geodesic distances
along the fsLR midthickness surface, actual parcel centroids, actual Yeo system assignments. That
matters more than it might sound: a spatial null model has nothing to preserve unless the map it is
given is genuinely spatially autocorrelated, so a tutorial built on ``np.random.randn(400)`` would
demonstrate the machinery while quietly misrepresenting what it does.

Nothing here is fit to, derived from, or redistributed from any real dataset, so the examples run
anywhere with no data access and no data-use agreement.

Geometry
    :func:`schaefer_geometry` and :func:`schaefer_systems` read the bundled parcellation resources.

Brain maps
    :func:`make_spatial_map` generates one spatially autocorrelated map, and
    :func:`make_correlated_map` generates a second one correlated with the first at a target
    strength.

Networks
    :func:`make_connectome` generates a weighted undirected structural connectome whose edge
    weights decay with distance, with modular and hub structure layered on top -- the ingredients a
    null network model is there to separate.

Time series
    :func:`make_timeseries` generates BOLD-like series whose autocorrelation varies across regions,
    so an intrinsic-timescale estimator has a known answer to recover.

All functions take a `seed` and are fully deterministic. Examples in the documentation rely on
specific seeds, so treat the outputs as fixtures: changing the generators changes the docs.

This module is being restored a piece at a time alongside the tutorials that use it. Generators for
subject cohorts and developmental trajectories with a known change point will return with the
tutorials that need them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

from .nulls.maps import _centroid_csv_path, load_distance_matrix

__all__ = [
    'schaefer_geometry',
    'schaefer_systems',
    'make_spatial_map',
    'make_correlated_map',
    'make_connectome',
    'make_timeseries',
]

# Parcellation resolutions with bundled geodesic distance matrices.
_SUPPORTED_RESOLUTIONS = (100, 200, 400)

# Default exponential correlation length for generated maps, in mm of geodesic distance. Chosen so
# that neighbouring parcels are strongly correlated while distant ones are effectively independent
# -- within-hemisphere geodesic distances in this parcellation top out around 235 mm.
_DEFAULT_LENGTH_SCALE = 25.0


def _check_resolution(n_regions):
    if n_regions not in _SUPPORTED_RESOLUTIONS:
        raise ValueError(
            f"n_regions must be one of {_SUPPORTED_RESOLUTIONS} (these are the resolutions with "
            f"bundled geodesic distance matrices); got {n_regions}"
        )


def schaefer_geometry(n_regions=400):
    """Load the bundled geometry for a Schaefer parcellation.

    Everything here is real, and ships inside the package -- no download, no data agreement.

    Parameters
    ----------
    n_regions : {100, 200, 400}
        Parcellation resolution (7-network order).

    Returns
    -------
    dict
        ``distance_matrix`` -- (n_regions, n_regions) geodesic distances in mm along the fsLR
        midthickness surface. Cross-hemisphere entries are NaN, since geodesic paths do not cross
        the midline; ``hemi`` -- (n_regions,) array of 'L'/'R' labels; ``centroids`` -- (n_regions, 3)
        R/A/S coordinates in MNI space; ``roi_names`` -- (n_regions,) full Schaefer parcel names;
        ``systems`` -- (n_regions,) Yeo 7-network assignment per parcel.

    See Also
    --------
    snaplab_tools.nulls.load_distance_matrix : the underlying loader.
    """
    _check_resolution(n_regions)
    distance_matrix, hemi = load_distance_matrix(n_regions, kind="geodesic")

    parc = pd.read_csv(_centroid_csv_path(n_regions))
    return {
        "distance_matrix": distance_matrix,
        "hemi": np.asarray(hemi),
        "centroids": parc[["R", "A", "S"]].values.astype(float),
        "roi_names": parc["ROI Name"].values,
        "systems": _systems_from_names(parc["ROI Name"].values),
    }


def _systems_from_names(roi_names):
    """Pull the Yeo system out of Schaefer parcel names like '7Networks_LH_Vis_1'."""
    return np.asarray([str(name).split("_")[2] for name in roi_names])


def schaefer_systems(n_regions=400):
    """Yeo 7-network system assignment for each parcel.

    Parsed from the bundled Schaefer parcel names, saving you the string surgery that would
    otherwise open every tutorial.

    Parameters
    ----------
    n_regions : {100, 200, 400}
        Parcellation resolution.

    Returns
    -------
    (n_regions,) ndarray of str
        One of 'Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default'.

    Examples
    --------
    >>> systems = schaefer_systems(400)
    >>> systems[:3]
    array(['Vis', 'Vis', 'Vis'], dtype='<U11')
    """
    _check_resolution(n_regions)
    parc = pd.read_csv(_centroid_csv_path(n_regions))
    return _systems_from_names(parc["ROI Name"].values)


def _sa_kernel_map(rng, distance_matrix, hemi, length_scale):
    """One z-scored spatially autocorrelated map: white noise smoothed by an exponential kernel.

    Smoothing runs within each hemisphere separately because the geodesic matrix has no
    cross-hemisphere distances -- the same constraint that makes
    :func:`snaplab_tools.nulls.generate_surrogates` work per-hemisphere.
    """
    n = distance_matrix.shape[0]
    out = np.empty(n)
    for h in ("L", "R"):
        idx = np.where(hemi == h)[0]
        kernel = np.exp(-distance_matrix[np.ix_(idx, idx)] / length_scale)
        out[idx] = kernel @ rng.standard_normal(idx.size)
    return (out - out.mean()) / out.std()


def make_spatial_map(n_regions=400, seed=0, length_scale=_DEFAULT_LENGTH_SCALE,
                     geometry=None):
    """Generate a spatially autocorrelated brain map on real Schaefer geometry.

    White noise is smoothed with an exponential kernel over the bundled geodesic distance matrix,
    so nearby parcels take similar values and the resulting map has the smooth, patchy structure
    that real cortical maps have. The output is z-scored.

    This is what makes the null-model tutorials meaningful: a spatial autocorrelation-preserving
    null only says something when the map it is preserving actually has autocorrelation to begin
    with.

    Parameters
    ----------
    n_regions : {100, 200, 400}
        Parcellation resolution.
    seed : int
        Random seed. Different seeds give independent maps.
    length_scale : float
        Correlation length in mm. Larger values give smoother, more slowly varying maps; roughly
        10-60 mm is a sensible range for this parcellation.
    geometry : dict or None
        Pre-loaded output of :func:`schaefer_geometry`. Pass it when generating many maps to skip
        reloading the distance matrix each time.

    Returns
    -------
    (n_regions,) ndarray
        A z-scored map (mean 0, standard deviation 1).

    Examples
    --------
    >>> x = make_spatial_map(n_regions=400, seed=0)
    >>> x.shape
    (400,)
    """
    geom = geometry if geometry is not None else schaefer_geometry(n_regions)
    rng = np.random.default_rng(seed)
    return _sa_kernel_map(rng, geom["distance_matrix"], geom["hemi"], length_scale)


def make_correlated_map(reference, rho=0.5, seed=1, length_scale=_DEFAULT_LENGTH_SCALE,
                        geometry=None):
    """Generate a second spatial map correlated with `reference` at approximately `rho`.

    Mixes a second spatially autocorrelated map into the (z-scored) reference at the weight implied
    by `rho`. Both components carry the same kind of spatial structure, so the result is a
    plausible brain map rather than a noisy copy of the reference.

    The added component is first made exactly orthogonal to the reference, so the realised Pearson
    correlation comes out at `rho` to numerical precision. This matters more than it looks: two
    *merely independent* spatially autocorrelated maps still correlate substantially by chance,
    because autocorrelation drives the effective sample size far below `n_regions`. Orthogonalizing
    means a tutorial can state the planted effect and have it hold.

    Parameters
    ----------
    reference : (n_regions,) array-like
        The map to correlate with. Internally z-scored, so its scale does not matter.
    rho : float
        Target Pearson correlation, in [-1, 1]. Achieved exactly.
    seed : int
        Random seed for the added component.
    length_scale : float
        Correlation length in mm for the added component.
    geometry : dict or None
        Pre-loaded output of :func:`schaefer_geometry`.

    Returns
    -------
    (n_regions,) ndarray
        A z-scored map whose Pearson correlation with `reference` is `rho`.

    Examples
    --------
    >>> x = make_spatial_map(seed=0)
    >>> y = make_correlated_map(x, rho=0.5, seed=1)
    >>> round(float(scipy.stats.pearsonr(x, y)[0]), 3)
    0.5
    """
    reference = np.asarray(reference, dtype=float)
    if not -1.0 <= rho <= 1.0:
        raise ValueError(f"rho must be in [-1, 1]; got {rho}")

    geom = geometry if geometry is not None else schaefer_geometry(len(reference))
    rng = np.random.default_rng(seed)
    component = _sa_kernel_map(rng, geom["distance_matrix"], geom["hemi"], length_scale)

    ref_z = (reference - np.nanmean(reference)) / np.nanstd(reference)

    # Project the reference out of the added component so the two are exactly uncorrelated; only
    # then does the rho / sqrt(1 - rho^2) mixture hit its target correlation exactly.
    component = component - (component @ ref_z) / (ref_z @ ref_z) * ref_z
    component = (component - component.mean()) / component.std()

    mixed = rho * ref_z + np.sqrt(1.0 - rho ** 2) * component
    return (mixed - mixed.mean()) / mixed.std()


def make_connectome(n_regions=400, seed=0, density=0.25, length_scale=40.0,
                    modular_gain=1.0, hub_scale=0.6, weight_noise=0.5, geometry=None):
    """Generate a weighted undirected structural connectome on real Schaefer geometry.

    Built to carry the three things a null network model exists to tell apart:

    **Geometry.** Edge weights fall off exponentially with the Euclidean distance between parcel
    centroids, so nearby regions are connected far more strongly than distant ones. This dominates
    real connectomes, and it is exactly what :func:`~snaplab_tools.nulls.geomsurr` holds fixed.

    **Modules.** Parcels in the same Yeo system get a weight bonus, producing community structure
    that distance alone does not explain.

    **Hubs.** Each node carries a random affinity added to all of its edges, so node strengths vary
    over and above what geometry dictates -- which is what makes the strength-preserving variants
    of a geometry-preserving null (``Wsp``, ``Wssp``) a meaningful constraint rather than a
    formality.

    Weights are strictly positive and lognormally distributed, as real streamline counts roughly
    are, and as :func:`~snaplab_tools.nulls.geomsurr` requires (it works on log-weights).

    Parameters
    ----------
    n_regions : {100, 200, 400}
        Parcellation resolution.
    seed : int
        Random seed. Different seeds give independent connectomes.
    density : float
        Fraction of possible edges retained, in (0, 1]. The strongest edges survive, so
        thresholding keeps the short-range and within-module connections, as empirical
        consistency-thresholding does.
    length_scale : float
        Distance decay constant in mm. Smaller values give a more strongly spatially embedded
        network.
    modular_gain : float
        Log-weight bonus applied to within-system edges. 0 removes module structure entirely.
    hub_scale : float
        Standard deviation of the per-node log-weight affinity. 0 makes node strengths depend on
        geometry alone.
    weight_noise : float
        Standard deviation of the per-edge lognormal noise.
    geometry : dict or None
        Pre-loaded output of :func:`schaefer_geometry`. Only ``centroids`` and ``systems`` are
        used.

    Returns
    -------
    (n_regions, n_regions) ndarray
        Symmetric, non-negative, zero diagonal. Non-zero entries are strictly positive.

    Notes
    -----
    Pair this with **Euclidean** centroid distances, not the geodesic matrix from
    :func:`schaefer_geometry` -- geodesic paths do not cross the midline, so that matrix is NaN
    across hemispheres and :func:`~snaplab_tools.nulls.geomsurr` will reject it::

        from snaplab_tools.nulls import load_distance_matrix
        D, _ = load_distance_matrix(n_regions, kind="euclidean")

    Examples
    --------
    >>> W = make_connectome(n_regions=100, seed=0)
    >>> W.shape, bool(np.array_equal(W, W.T)), bool((W >= 0).all())
    ((100, 100), True, True)
    >>> round(float((W > 0).sum() / (100 * 99)), 2)   # realised density
    0.25
    """
    _check_resolution(n_regions)
    geom = geometry if geometry is not None else schaefer_geometry(n_regions)
    rng = np.random.default_rng(seed)

    # Euclidean, not geodesic: geomsurr needs a finite distance for every edge, and the geodesic
    # matrix is NaN across hemispheres.
    distance = squareform(pdist(np.asarray(geom["centroids"], dtype=float)))
    systems = np.asarray(geom["systems"])

    # Compose the three ingredients in log space, so the weights come out lognormal.
    log_w = -distance / length_scale
    log_w = log_w + modular_gain * (systems[:, None] == systems[None, :])
    affinity = rng.normal(scale=hub_scale, size=n_regions)
    log_w = log_w + affinity[:, None] + affinity[None, :]

    noise = rng.normal(scale=weight_noise, size=(n_regions, n_regions))
    log_w = log_w + np.triu(noise, 1) + np.triu(noise, 1).T   # symmetric noise

    # Threshold to the target density on the upper triangle, then mirror.
    if not 0.0 < density <= 1.0:
        raise ValueError(f"density must be in (0, 1]; got {density}")
    triu = np.triu_indices(n_regions, k=1)
    n_keep = max(1, int(round(density * triu[0].size)))
    keep = np.argsort(log_w[triu])[::-1][:n_keep]

    W = np.zeros((n_regions, n_regions))
    W[triu[0][keep], triu[1][keep]] = np.exp(log_w[triu][keep])
    return W + W.T


def make_timeseries(n_regions=50, n_timepoints=600, tr=0.8, seed=0, tau_range=(1.0, 8.0)):
    """Generate BOLD-like parcellated time series with region-varying autocorrelation.

    Each region is an AR(1) process whose decay constant varies systematically across regions,
    producing a gradient of intrinsic timescales like the one observed empirically along the
    cortical hierarchy. That gradient is the point: it gives
    :func:`~snaplab_tools.timescales.compute_acf` something real to recover, and
    :func:`~snaplab_tools.signal.apply_frequency_filter` something to visibly change.

    Parameters
    ----------
    n_regions : int
        Number of regions. Unlike the map generators this is unconstrained -- no parcellation
        geometry is involved.
    n_timepoints : int
        Number of volumes.
    tr : float
        Repetition time in seconds; sets the sampling rate for filtering.
    seed : int
        Random seed.
    tau_range : tuple of float
        (min, max) decay timescale in seconds, swept linearly across regions.

    Returns
    -------
    dict
        ``ts`` -- (n_regions, n_timepoints) time series; ``tau`` -- (n_regions,) the true decay
        timescale of each region, in seconds; ``tr`` -- the repetition time, echoed back for
        passing to the filter.

    Examples
    --------
    >>> data = make_timeseries(n_regions=50, n_timepoints=600, seed=0)
    >>> filtered = apply_frequency_filter(data['ts'], 1 / data['tr'],
    ...                                   lowpass=0.1, highpass=0.01)
    """
    rng = np.random.default_rng(seed)
    tau = np.linspace(tau_range[0], tau_range[1], n_regions)

    # AR(1) coefficient implied by an exponential decay of time constant tau, sampled every tr.
    phi = np.exp(-tr / tau)

    ts = np.zeros((n_regions, n_timepoints))
    innovations = rng.standard_normal((n_regions, n_timepoints))
    # Scale innovations so every region ends up with unit variance regardless of its phi.
    scale = np.sqrt(1.0 - phi ** 2)[:, None]
    ts[:, 0] = innovations[:, 0]
    for t in range(1, n_timepoints):
        ts[:, t] = phi * ts[:, t - 1] + scale[:, 0] * innovations[:, t]

    return {"ts": ts, "tau": tau, "tr": tr}
