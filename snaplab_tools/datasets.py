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

All functions take a `seed` and are fully deterministic. Examples in the documentation rely on
specific seeds, so treat the outputs as fixtures: changing the generators changes the docs.

This module is being restored a piece at a time alongside the tutorials that use it. Generators for
subject cohorts, structural connectomes, developmental trajectories with a known change point, and
BOLD-like time series will return with the tutorials that need them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .nulls.nulls import _centroid_csv_path, load_distance_matrix

__all__ = [
    'schaefer_geometry',
    'schaefer_systems',
    'make_spatial_map',
    'make_correlated_map',
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
