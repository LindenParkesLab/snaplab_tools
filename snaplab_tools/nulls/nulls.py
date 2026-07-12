"""Spatial null models for parcellated cortical maps (BrainSMASH variogram surrogates).

Generic, project-agnostic null-generating machinery for the lab: the distance bases, the
BrainSMASH surrogate generator, and the variogram-null inference helpers. Project-specific
conveniences (cache-dir conventions, dataset wiring) belong in the calling project, not here.

Distance basis
--------------
BrainSMASH matches the empirical *variogram* — map variance as a function of pairwise distance
between parcels. For a surface parcellation (Schaefer on fsLR) the map lives on the folded
cortical sheet, so the appropriate metric is **geodesic distance along the surface**, not
Euclidean distance between parcel centroids (Burt et al., 2020, NeuroImage). Euclidean distance
treats parcels on opposite banks of a sulcus as neighbours, which compresses the variogram and
tends to make the null too permissive.

Geodesic distance is only defined *within* a hemisphere (the L and R fsLR surfaces are
disconnected), so the geodesic matrix is block-diagonal: within-hemisphere blocks hold true
geodesic distances and the cross-hemisphere block is ``NaN``. Surrogates are therefore generated
**per hemisphere** and concatenated (``per_hemisphere=True``, the default for the geodesic basis).

Parcel-to-parcel geodesic distance uses the **centroid-vertex** method: each parcel is
represented by its most-central surface vertex, and distance is the surface geodesic between
centroid vertices (Connectome Workbench ``wb_command -surface-geodesic-distance``). This is the
fast, standard pragmatic choice; the full mean-pairwise-vertex variant is not implemented here.

``kind="euclidean"`` reproduces the legacy centroid-Euclidean basis exactly, for backwards
compatibility / comparison. Cache keys always include ``kind`` so geodesic and Euclidean
surrogates never collide on disk.

Resources (surfaces, parcellation dlabels, centroid CSVs, and the built distance matrices) are
bundled in ``snaplab_tools/nulls/resources`` and resolved via the package path, so no absolute
paths are baked in.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import scipy as sp
import scipy.stats  # noqa: F401  (populate sp.stats without relying on a transitive import)
import scipy.spatial.distance  # noqa: F401  (populate sp.spatial.distance)
import pandas as pd
import nibabel as nib
from tqdm import tqdm
from brainsmash.mapgen.base import Base
from statsmodels.stats.multitest import multipletests

from .utils import get_null_p

# --------------------------------------------------------------------------------------------
# Resource resolution (bundled in snaplab_tools/nulls/resources)
# --------------------------------------------------------------------------------------------
_RES = Path(__file__).resolve().parent / "resources"
_SURFACES = {
    "L": _RES / "surfaces" / "tpl-fsLR_den-32k_hemi-L_midthickness.surf.gii",
    "R": _RES / "surfaces" / "tpl-fsLR_den-32k_hemi-R_midthickness.surf.gii",
}
# fsLR-32k grayordinate layout: rows 0..32491 are the left cortex, 32492..64983 the right.
_HEMI_SLICES = {"L": slice(0, 32492), "R": slice(32492, 64984)}
# Connectome Workbench binary (on PATH by default; override with the WB_COMMAND env var).
WB_COMMAND = os.environ.get("WB_COMMAND", "wb_command")


def _dlabel_path(n_regions: int) -> Path:
    return _RES / "parcellations" / f"Schaefer2018_{n_regions}Parcels_7Networks_order.dlabel.nii"


def _centroid_csv_path(n_regions: int) -> Path:
    return (
        _RES
        / "parcellations"
        / f"Schaefer2018_{n_regions}Parcels_7Networks_order_FSLMNI152_1mm.Centroid_RAS.csv"
    )


def _distance_cache_path(n_regions: int, kind: str) -> Path:
    return _RES / "distances" / f"schaefer{n_regions}-7_{kind}_distance.npy"


def _hemi_cache_path(n_regions: int, kind: str) -> Path:
    return _RES / "distances" / f"schaefer{n_regions}-7_{kind}_hemi.npy"


# --------------------------------------------------------------------------------------------
# Distance matrices
# --------------------------------------------------------------------------------------------
def _geodesic_from_vertex(surface: str, vertex: int) -> np.ndarray:
    """Geodesic distance (mm) from one vertex to every vertex on ``surface``, via wb_command."""
    fd, out = tempfile.mkstemp(suffix=".func.gii")
    os.close(fd)
    try:
        subprocess.run(
            [WB_COMMAND, "-surface-geodesic-distance", surface, str(int(vertex)), out],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        return np.asarray(nib.load(out).darrays[0].data, dtype=float)
    finally:
        if os.path.exists(out):
            os.unlink(out)


def build_geodesic_distance_matrix(n_regions: int = 400):
    """Build the per-hemisphere centroid-vertex geodesic parcel distance matrix.

    Returns
    -------
    D : (n_regions, n_regions) ndarray
        Symmetric within-hemisphere geodesic distances (mm); the cross-hemisphere block is NaN.
    hemi : (n_regions,) ndarray of '<U1'
        'L'/'R' hemisphere label for each parcel.
    """
    labels = np.asarray(nib.load(str(_dlabel_path(n_regions))).get_fdata()).ravel().astype(int)
    if labels.shape[0] != 64984:
        raise ValueError(
            f"Expected an fsLR-32k dlabel with 64984 grayordinates, got {labels.shape[0]}."
        )
    D = np.full((n_regions, n_regions), np.nan, dtype=float)
    hemi = np.empty(n_regions, dtype="<U1")

    for h, sl in _HEMI_SLICES.items():
        surface = str(_SURFACES[h])
        coords = np.asarray(nib.load(surface).darrays[0].data, dtype=float)
        lab = labels[sl]
        keys = sorted(int(k) for k in set(lab.tolist()) if k > 0)
        # Represent each parcel by the in-parcel vertex closest to the parcel's mean coordinate.
        centroid_vertex = {}
        for k in keys:
            vids = np.where(lab == k)[0]
            centre = coords[vids].mean(axis=0)
            centroid_vertex[k] = int(vids[np.argmin(((coords[vids] - centre) ** 2).sum(axis=1))])
            hemi[k - 1] = h
        for k in keys:
            d = _geodesic_from_vertex(surface, centroid_vertex[k])
            row = k - 1
            for k2 in keys:
                D[row, k2 - 1] = d[centroid_vertex[k2]]

    D = (D + D.T) / 2.0  # symmetrise within-hemisphere blocks (cross-hemisphere stays NaN)
    return D, hemi


def _build_euclidean_distance_matrix(n_regions: int = 400):
    """Legacy basis: Euclidean distance between FSLMNI152 parcel centroids. Full (no NaN block)."""
    parc = pd.read_csv(
        _centroid_csv_path(n_regions),
        header=0,
        names=["ROI_Label", "ROI_Name", "R", "A", "S"],
        index_col=0,
    )
    D = sp.spatial.distance.squareform(
        sp.spatial.distance.pdist(parc[["R", "A", "S"]].values, "euclidean")
    )
    names = parc["ROI_Name"].str.strip()
    hemi = np.where(names.str.contains("LH"), "L", np.where(names.str.contains("RH"), "R", "?"))
    return D, hemi.astype("<U1")


def load_distance_matrix(n_regions: int = 400, kind: str = "geodesic", rebuild: bool = False):
    """Load (building/caching on first use) the parcel distance matrix and hemisphere labels.

    Parameters
    ----------
    n_regions : int
        Schaefer 7-network resolution (100, 200, 400).
    kind : {'geodesic', 'euclidean'}
        'geodesic' (default) — per-hemisphere surface geodesic distance (cross-hemisphere NaN).
        'euclidean' — legacy centroid-Euclidean basis (full matrix), for reproducing old results.
    rebuild : bool
        Force a rebuild of the geodesic cache even if it exists.

    Returns
    -------
    (D, hemi) : (ndarray (n_regions, n_regions), ndarray (n_regions,) of '<U1')
    """
    if kind == "euclidean":
        # Cheap; computed on demand (not cached) so it always tracks the bundled centroid CSV.
        return _build_euclidean_distance_matrix(n_regions)
    if kind != "geodesic":
        raise ValueError(f"kind must be 'geodesic' or 'euclidean', got {kind!r}.")

    dpath, hpath = _distance_cache_path(n_regions, "geodesic"), _hemi_cache_path(n_regions, "geodesic")
    if rebuild or not dpath.exists():
        D, hemi = build_geodesic_distance_matrix(n_regions)
        dpath.parent.mkdir(parents=True, exist_ok=True)
        np.save(dpath, D)
        np.save(hpath, hemi)
        return D, hemi
    return np.load(dpath), np.load(hpath)


# --------------------------------------------------------------------------------------------
# Surrogate generation
# --------------------------------------------------------------------------------------------
def generate_surrogates(
    brain_map,
    name=None,
    n_regions=None,
    n_perms=5000,
    kind="geodesic",
    per_hemisphere=None,
    distance_matrix=None,
    hemi=None,
    cache_dir=None,
    seed=0,
    **base_kwargs,
):
    """BrainSMASH SA-preserving surrogates of a parcellated map, NaN-safe and per-hemisphere.

    The map is surrogated on the finite (non-NaN) parcels only; invalid parcels stay NaN in the
    output. With the geodesic basis, surrogates are generated independently within each
    hemisphere (the geodesic matrix has no cross-hemisphere distances) and concatenated.

    Parameters
    ----------
    brain_map : (n_regions,) array-like
        The map to surrogate (may contain NaNs).
    name : str, optional
        Cache key. Required if ``cache_dir`` is given. The cache filename embeds ``kind``,
        ``n_regions`` and ``n_perms`` so different bases never collide.
    n_regions : int, optional
        Resolution used to load the distance matrix if ``distance_matrix`` is not supplied.
        Defaults to ``len(brain_map)``.
    n_perms : int
        Number of surrogates.
    kind : {'geodesic', 'euclidean'}
        Distance basis (used only when ``distance_matrix`` is not supplied).
    per_hemisphere : bool, optional
        Generate surrogates within each hemisphere separately. Defaults to True for the geodesic
        basis and False for Euclidean. Required True whenever the distance matrix has NaN
        cross-hemisphere entries.
    distance_matrix, hemi : ndarray, optional
        Precomputed distance matrix / hemisphere labels (skips loading).
    cache_dir : str, optional
        Directory for the on-disk surrogate cache. If the file exists it is loaded and returned.
    seed : int
        Base RNG seed (the R hemisphere uses ``seed + 1`` for reproducible independence).
    **base_kwargs :
        Extra keyword arguments forwarded to ``brainsmash.mapgen.base.Base`` (deltas, kernel, pv,
        nh, ...).

    Returns
    -------
    (n_perms, n_regions) ndarray
    """
    brain_map = np.asarray(brain_map, dtype=float)
    n = brain_map.shape[0]
    if n_regions is None:
        n_regions = n

    cache_file = None
    if cache_dir is not None:
        if name is None:
            raise ValueError("`name` is required when `cache_dir` is set (used as the cache key).")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(
            cache_dir, f"{name}_kind-{kind}_regions-{n_regions}_perms-{n_perms}_nulls.npy"
        )
        if os.path.exists(cache_file):
            return np.load(cache_file)

    if distance_matrix is None:
        distance_matrix, hemi = load_distance_matrix(n_regions, kind=kind)
    if per_hemisphere is None:
        per_hemisphere = kind == "geodesic"

    valid = ~np.isnan(brain_map)
    out = np.full((n_perms, n), np.nan)

    if per_hemisphere:
        if hemi is None:
            raise ValueError("per_hemisphere=True requires hemisphere labels (`hemi`).")
        for h_offset, h in enumerate(("L", "R")):
            idx = np.where(valid & (np.asarray(hemi) == h))[0]
            if idx.size < 2:
                continue
            D = distance_matrix[np.ix_(idx, idx)]
            out[:, idx] = Base(brain_map[idx], D, seed=seed + h_offset, **base_kwargs)(n=n_perms)
    else:
        idx = np.where(valid)[0]
        if np.isnan(distance_matrix[np.ix_(idx, idx)]).any():
            raise ValueError(
                "Distance matrix has NaN entries among valid parcels (cross-hemisphere block); "
                "use per_hemisphere=True (the default for the geodesic basis)."
            )
        D = distance_matrix[np.ix_(idx, idx)]
        out[:, idx] = Base(brain_map[idx], D, seed=seed, **base_kwargs)(n=n_perms)

    if cache_file is not None:
        np.save(cache_file, out)
    return out


# --------------------------------------------------------------------------------------------
# Inference on top of surrogates
# --------------------------------------------------------------------------------------------
def residualize(y, covariates):
    """OLS-residualize ``y`` on ``covariates`` (intercept always included). ``covariates`` is a
    (n,) or (n, k) array, or None for intercept-only (mean-centering)."""
    if covariates is None:
        Z = np.ones((len(y), 1))
    else:
        C = np.asarray(covariates, float)
        if C.ndim == 1:
            C = C[:, None]
        Z = np.column_stack([np.ones(len(y)), C])
    return y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]


def _corr(a, b, method):
    return sp.stats.spearmanr(a, b)[0] if method == "spearman" else sp.stats.pearsonr(a, b)[0]


def corr_with_covar_null(brain_map, target_map, covariates, target_surrogates, method="pearson",
                         two_tailed=True):
    """Spatial correlation between a (real) brain map and a target map, optionally residualizing
    BOTH on a covariate set, tested against a BrainSMASH null.

    The null surrogates the *target* map (brain map and covariates held fixed) and recomputes the
    identical (residualized) correlation. Returns ``dict(r, p_smash, n, null_lo, null_hi, null)``
    where ``null`` is the full finite null vector.

    ``two_tailed`` (default True) tests the magnitude of the correlation (``|r|``); set False for a
    one-tailed test in the observed effect's direction (``get_null_p(..., abs=False)``).
    """
    brain_map = np.asarray(brain_map, float)
    target_map = np.asarray(target_map, float)
    C = None
    if covariates is not None:
        C = np.asarray(covariates, float)
        if C.ndim == 1:
            C = C[:, None]
        if C.shape[0] != len(brain_map):  # accept (k, n_regions)
            C = C.T
    masks = [~np.isnan(brain_map), ~np.isnan(target_map)]
    if C is not None:
        masks.append(~np.isnan(C).any(axis=1))
    base_valid = np.logical_and.reduce(masks)

    def _stat(tvec, valid):
        b, t = brain_map[valid], tvec[valid]
        if C is None:
            return _corr(t, b, method)
        cc = C[valid]
        return _corr(residualize(t, cc), residualize(b, cc), method)

    obs = _stat(target_map, base_valid)
    null = np.full(target_surrogates.shape[0], np.nan)
    for i in range(target_surrogates.shape[0]):
        surr = target_surrogates[i]
        v = base_valid & ~np.isnan(surr)
        null[i] = _stat(surr, v)
    null = null[~np.isnan(null)]
    return {
        "r": obs,
        "p_smash": get_null_p(obs, null, version="standard", abs=two_tailed),
        "n": int(base_valid.sum()),
        "null_lo": float(np.nanpercentile(null, 2.5)),
        "null_hi": float(np.nanpercentile(null, 97.5)),
        "null": null,
    }


def corr_with_null(brain_map, target_map, target_surrogates, method="pearson", two_tailed=True):
    """Convenience wrapper: spatial correlation with a BrainSMASH null, no covariates.
    ``two_tailed`` (default True) tests ``|r|``; set False for a one-tailed directional test."""
    return corr_with_covar_null(brain_map, target_map, None, target_surrogates, method=method,
                                two_tailed=two_tailed)


def correlate_family(
    brain_map,
    df_targets,
    covariates=None,
    n_regions=None,
    kind="geodesic",
    n_perms=5000,
    cache_dir=None,
    desc="targets",
    method="pearson",
    two_tailed=True,
    distance_matrix=None,
    hemi=None,
):
    """Run :func:`corr_with_covar_null` for every column of ``df_targets`` vs ``brain_map`` under
    one covariate set (surrogates cached per target map), then BH-FDR within the family.

    Returns a DataFrame indexed by target with ``r, p_smash, n, null_lo, null_hi, fdr_sig`` and
    the full ``null`` vector. Pass ``distance_matrix``/``hemi`` to reuse a preloaded basis.

    ``two_tailed`` (default True) tests ``|r|`` per target; set False for one-tailed directional
    p-values (the BH-FDR step is applied to whichever p-values result).
    """
    if n_regions is None:
        n_regions = len(brain_map)
    if distance_matrix is None:
        distance_matrix, hemi = load_distance_matrix(n_regions, kind=kind)
    rows = {}
    for col in tqdm(df_targets.columns, desc=desc):
        surr = generate_surrogates(
            df_targets[col].values,
            name=col,
            n_regions=n_regions,
            n_perms=n_perms,
            kind=kind,
            distance_matrix=distance_matrix,
            hemi=hemi,
            cache_dir=cache_dir,
        )
        rows[col] = corr_with_covar_null(brain_map, df_targets[col].values, covariates, surr,
                                         method=method, two_tailed=two_tailed)
    df = pd.DataFrame(rows).T
    num = ["r", "p_smash", "n", "null_lo", "null_hi"]
    df[num] = df[num].astype(float)
    df["fdr_sig"] = multipletests(df["p_smash"].values, alpha=0.05, method="fdr_bh")[0]
    return df[num + ["fdr_sig", "null"]]


def network_enrichment(brain_map, systems, stage_surrogates):
    """Per-system mean of a map tested against a BrainSMASH surrogate null.

    The map is surrogated (pass ``stage_surrogates``); per-system means are recomputed per
    surrogate; the two-tailed p compares the observed mean's deviation from the null mean via
    ``get_null_p(abs=True)``. ``systems`` is any per-parcel label vector (e.g. Yeo-7 networks).
    Returns a DataFrame indexed by system with observed mean, null mean, 95% null band,
    ``p_smash`` and the full per-surrogate system-mean vector (``null``).
    """
    systems = np.asarray(systems)
    rows, nulls = {}, {}
    for net in sorted(set(systems)):
        m = systems == net
        obs = np.nanmean(brain_map[m])
        null_vals = np.nanmean(stage_surrogates[:, m], axis=1)
        center = np.nanmean(null_vals)
        rows[net] = {
            "mean": obs,
            "null_mean": center,
            "null_lo": np.nanpercentile(null_vals, 2.5),
            "null_hi": np.nanpercentile(null_vals, 97.5),
            "p_smash": get_null_p(obs - center, null_vals - center, version="standard", abs=True),
        }
        nulls[net] = null_vals[np.isfinite(null_vals)]
    df = pd.DataFrame(rows).T
    df["null"] = pd.Series(nulls)
    return df


# Backwards-compatible alias (the enrichment is not Yeo-specific; ``systems`` is any label vector).
yeo_network_enrichment = network_enrichment
