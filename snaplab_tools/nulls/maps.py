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
paths are baked in. Two atlases are supported, selected with ``atlas=``:

``atlas="schaefer"`` (default)
    All ten Schaefer2018 7-network resolutions, 100 to 1000 in steps of 100. Parcellation and
    matrix both bundled.
``atlas="glasser"``
    HCP-MMP1.0, 360 cortical areas. The distance matrix and centroids are bundled, but the
    parcellation is *not* -- it comes via BALSA under the HCP Data Use Terms, which restrict
    redistribution (see THIRD_PARTY_NOTICES.md). Loading works offline like Schaefer's; only
    rebuilding needs your own copy of the dlabel.

Nothing needs Connectome Workbench at run time; it is required only to build a matrix.

**Parcel order is each atlas's own**, and the two disagree: Schaefer runs left hemisphere first,
HCP-MMP1.0 runs right first (areas 1-180 right, 181-360 left). Neither is reordered here, so a map
parcellated with the published atlas lines up as-is -- but a map built by concatenating hemispheres
by hand needs care.

One upstream wrinkle: at 1000 parcels the CBIG fsLR-32k dlabel declares 1000 parcels but two of
them -- 533 ``7Networks_RH_Vis_33`` and 903 ``7Networks_RH_Cont_Cing_1`` -- have no vertices on
the surface, so the geodesic basis there covers 998 parcels. Those two keep an all-NaN row;
:func:`generate_surrogates` drops them (with a warning) and returns them as NaN. Their centroids
do exist, so ``kind="euclidean"`` covers all 1000.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import warnings
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

# get_null_p and residualize are generic statistics with nothing null-model-specific about them,
# so they live in snaplab_tools.stats and are used from there.
from ..stats import compute_stat, get_null_p, residualize

__all__ = [
    'load_distance_matrix',
    'build_geodesic_distance_matrix',
    'generate_surrogates',
    'corr_with_null',
    'corr_with_covar_null',
    'correlate_family',
    'network_enrichment',
    'WB_COMMAND',
]

# --------------------------------------------------------------------------------------------
# Resource resolution (bundled in snaplab_tools/nulls/resources)
# --------------------------------------------------------------------------------------------
_RES = Path(__file__).resolve().parent / "resources"
_SURFACES = {
    "L": _RES / "surfaces" / "tpl-fsLR_den-32k_hemi-L_midthickness.surf.gii",
    "R": _RES / "surfaces" / "tpl-fsLR_den-32k_hemi-R_midthickness.surf.gii",
}
# Number of vertices per hemisphere in the fsLR-32k mesh. A dlabel may store one grayordinate per
# vertex (Schaefer, from CBIG) or drop the medial wall (Glasser, from BALSA); either way the mesh
# it indexes into is this one, which is what makes the two comparable.
_FSLR_32K_VERTICES = 32492
_CORTEX_STRUCTURES = {"CIFTI_STRUCTURE_CORTEX_LEFT": "L", "CIFTI_STRUCTURE_CORTEX_RIGHT": "R"}
# Connectome Workbench binary (on PATH by default; override with the WB_COMMAND env var).
WB_COMMAND = os.environ.get("WB_COMMAND", "wb_command")

# Atlases with bundled resources. `dlabel` is a filename template for the parcellation the geodesic
# matrix is built from; it is None where the source cannot be redistributed, in which case rebuilds
# take an explicit path and only the derived matrix ships. Parcel order is the atlas's own: Schaefer
# runs left hemisphere first, Glasser (HCP-MMP1.0) runs right first, and both are preserved as-is so
# a map parcellated with the published atlas lines up without reordering.
_ATLASES = {
    "schaefer": {
        "resolutions": tuple(range(100, 1100, 100)),
        "dlabel": "Schaefer2018_{n}Parcels_7Networks_order.dlabel.nii",
        "centroids": "Schaefer2018_{n}Parcels_7Networks_order_FSLMNI152_1mm.Centroid_RAS.csv",
        "distances": "schaefer{n}-7",
    },
    "glasser": {
        "resolutions": (360,),
        # HCP-MMP1.0 is distributed via BALSA under the HCP Data Use Terms, which restrict
        # redistribution, so the parcellation itself is not bundled -- see THIRD_PARTY_NOTICES.md.
        "dlabel": None,
        "centroids": "Glasser360_HCPMMP1_FSLMNI152_1mm.Centroid_RAS.csv",
        "distances": "glasser{n}",
    },
}


def _atlas_spec(atlas: str) -> dict:
    try:
        return _ATLASES[atlas]
    except KeyError:
        raise ValueError(f"atlas must be one of {sorted(_ATLASES)}, got {atlas!r}.") from None


def _dlabel_path(n_regions: int, atlas: str = "schaefer") -> Path:
    template = _atlas_spec(atlas)["dlabel"]
    if template is None:
        raise ValueError(
            f"The {atlas} parcellation is not bundled (its licence does not permit "
            f"redistribution), so there is no dlabel to build from. Pass the path to your own copy "
            f"via `dlabel` to rebuild its distance matrix; the prebuilt matrix loads without it."
        )
    return _RES / "parcellations" / template.format(n=n_regions)


def _centroid_csv_path(n_regions: int, atlas: str = "schaefer") -> Path:
    return _RES / "parcellations" / _atlas_spec(atlas)["centroids"].format(n=n_regions)


def _distance_cache_path(n_regions: int, kind: str, atlas: str = "schaefer") -> Path:
    stem = _atlas_spec(atlas)["distances"].format(n=n_regions)
    return _RES / "distances" / f"{stem}_{kind}_distance.npy"


def _hemi_cache_path(n_regions: int, kind: str, atlas: str = "schaefer") -> Path:
    stem = _atlas_spec(atlas)["distances"].format(n=n_regions)
    return _RES / "distances" / f"{stem}_{kind}_hemi.npy"


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


def _hemisphere_from_name(name) -> str:
    """'L'/'R' read off a parcel name, across the naming schemes the bundled atlases use.

    Schaefer parcels are named '7Networks_LH_Vis_1'; HCP-MMP1.0 areas are named 'R_V1_ROI'. Returns
    '?' for anything else rather than guessing.
    """
    name = str(name).strip()
    if "_LH_" in name:
        return "L"
    if "_RH_" in name:
        return "R"
    if name.startswith("L_"):
        return "L"
    if name.startswith("R_"):
        return "R"
    return "?"


def _cortical_vertex_labels(img):
    """Per-hemisphere ``(vertex_ids, labels)`` for the cortical grayordinates of a dlabel.

    Handles both layouts in circulation. CBIG's Schaefer files store one grayordinate per vertex,
    so grayordinate index *is* vertex index; BALSA's HCP-MMP1.0 files drop the medial wall, so the
    two differ and the mapping has to be read off the CIFTI brain-model axis. Reading it either way
    keeps the two atlases on the same footing -- both index the same fsLR-32k mesh, which is the
    surface the geodesics are measured on.
    """
    labels = np.asarray(img.get_fdata()).ravel().astype(int)
    out = {}
    for structure, index, model in img.header.get_axis(1).iter_structures():
        hemisphere = _CORTEX_STRUCTURES.get(str(structure))
        if hemisphere is None:
            continue      # subcortical structures: the HCP-MMP1.0 file carries 19 of them
        n_mesh = model.nvertices[str(structure)]
        if n_mesh != _FSLR_32K_VERTICES:
            raise ValueError(
                f"{structure} indexes a {n_mesh}-vertex mesh; this expects fsLR-32k "
                f"({_FSLR_32K_VERTICES} vertices per hemisphere)."
            )
        out[hemisphere] = (np.asarray(model.vertex, dtype=int), labels[index])
    if set(out) != {"L", "R"}:
        raise ValueError(f"Expected both cortical hemispheres in the dlabel, found {sorted(out)}.")
    return out


def build_geodesic_distance_matrix(n_regions: int = 400, atlas: str = "schaefer", dlabel=None):
    """Build the per-hemisphere centroid-vertex geodesic parcel distance matrix.

    Parameters
    ----------
    n_regions : int
        Number of parcels to build for. Labels above it are ignored, which is what separates the
        360 cortical areas of HCP-MMP1.0 from the 19 subcortical structures its dlabel also carries.
    atlas : {'schaefer', 'glasser'}
        Which bundled parcellation to read, when ``dlabel`` is not given.
    dlabel : str or Path, optional
        An explicit dlabel to build from, for a parcellation that is not bundled. Required for
        atlases whose licence does not permit redistributing the parcellation itself.

    Returns
    -------
    D : (n_regions, n_regions) ndarray
        Symmetric within-hemisphere geodesic distances (mm); the cross-hemisphere block is NaN.
    hemi : (n_regions,) ndarray of '<U1'
        'L'/'R' hemisphere label for each parcel.
    """
    img = nib.load(str(dlabel if dlabel is not None else _dlabel_path(n_regions, atlas)))
    D = np.full((n_regions, n_regions), np.nan, dtype=float)
    hemi = np.empty(n_regions, dtype="<U1")

    for h, (vertex_ids, lab) in _cortical_vertex_labels(img).items():
        surface = str(_SURFACES[h])
        coords = np.asarray(nib.load(surface).darrays[0].data, dtype=float)
        keys = sorted(int(k) for k in set(lab.tolist()) if 0 < k <= n_regions)
        # Represent each parcel by the in-parcel vertex closest to the parcel's mean coordinate.
        centroid_vertex = {}
        for k in keys:
            vids = vertex_ids[lab == k]
            centre = coords[vids].mean(axis=0)
            centroid_vertex[k] = int(vids[np.argmin(((coords[vids] - centre) ** 2).sum(axis=1))])
            hemi[k - 1] = h
        for k in keys:
            d = _geodesic_from_vertex(surface, centroid_vertex[k])
            row = k - 1
            for k2 in keys:
                D[row, k2 - 1] = d[centroid_vertex[k2]]

    # A handful of parcels declared in the label table have no vertices on the fsLR-32k surface
    # (at 1000 parcels, 533 '7Networks_RH_Vis_33' and 903 '7Networks_RH_Cont_Cing_1'). They keep
    # their row/column of NaN -- there is no surface location to measure a geodesic from -- but
    # still get a hemisphere label, read off the label-table name, so callers can see what they
    # are. Their centroids do exist, so kind='euclidean' covers them.
    empty = np.where(hemi == "")[0]
    if empty.size:
        names = img.header.get_axis(0).label[0]
        for row in empty:
            name = names.get(row + 1, ("",))[0]
            hemi[row] = _hemisphere_from_name(name)
        warnings.warn(
            f"{atlas}{n_regions}: {empty.size} parcel(s) have no vertices on the fsLR-32k "
            f"surface and so have no geodesic distances (1-based labels "
            f"{[int(i) + 1 for i in empty]}); their rows stay NaN.",
            stacklevel=2,
        )

    D = (D + D.T) / 2.0  # symmetrise within-hemisphere blocks (cross-hemisphere stays NaN)
    return D, hemi


def _build_euclidean_distance_matrix(n_regions: int = 400, atlas: str = "schaefer"):
    """Legacy basis: Euclidean distance between FSLMNI152 parcel centroids. Full (no NaN block)."""
    parc = pd.read_csv(
        _centroid_csv_path(n_regions, atlas),
        header=0,
        names=["ROI_Label", "ROI_Name", "R", "A", "S"],
        index_col=0,
    )
    D = sp.spatial.distance.squareform(
        sp.spatial.distance.pdist(parc[["R", "A", "S"]].values, "euclidean")
    )
    hemi = np.array([_hemisphere_from_name(n) for n in parc["ROI_Name"]], dtype="<U1")
    return D, hemi


def load_distance_matrix(n_regions: int = 400, kind: str = "geodesic", rebuild: bool = False,
                         atlas: str = "schaefer"):
    """Load (building/caching on first use) the parcel distance matrix and hemisphere labels.

    Parameters
    ----------
    n_regions : int
        Number of parcels. For ``atlas='schaefer'``, all ten published 7-network resolutions (100
        to 1000 in steps of 100) ship with prebuilt geodesic matrices; for ``atlas='glasser'``,
        360. Anything else is built on demand and needs ``wb_command`` plus a matching dlabel.
    kind : {'geodesic', 'euclidean'}
        'geodesic' (default) — per-hemisphere surface geodesic distance (cross-hemisphere NaN).
        'euclidean' — legacy centroid-Euclidean basis (full matrix), for reproducing old results.
    rebuild : bool
        Force a rebuild of the geodesic cache even if it exists. Needs the parcellation, which is
        not bundled for every atlas -- see :func:`build_geodesic_distance_matrix`.
    atlas : {'schaefer', 'glasser'}
        Which parcellation family. Parcel order is the atlas's own, so a map parcellated with the
        published atlas needs no reordering: Schaefer runs left hemisphere first, HCP-MMP1.0 runs
        right first (areas 1-180 are right, 181-360 left).

    Returns
    -------
    (D, hemi) : (ndarray (n_regions, n_regions), ndarray (n_regions,) of '<U1')
    """
    _atlas_spec(atlas)
    if kind == "euclidean":
        # Cheap; computed on demand (not cached) so it always tracks the bundled centroid CSV.
        return _build_euclidean_distance_matrix(n_regions, atlas)
    if kind != "geodesic":
        raise ValueError(f"kind must be 'geodesic' or 'euclidean', got {kind!r}.")

    dpath = _distance_cache_path(n_regions, "geodesic", atlas)
    hpath = _hemi_cache_path(n_regions, "geodesic", atlas)
    if rebuild or not dpath.exists():
        D, hemi = build_geodesic_distance_matrix(n_regions, atlas)
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
    atlas="schaefer",
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
    atlas : {'schaefer', 'glasser'}
        Parcellation family, used only when ``distance_matrix`` is not supplied.
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
        # The atlas enters the key only when it is not the default, so caches written before
        # non-Schaefer atlases existed still hit rather than silently regenerating.
        tag = "" if atlas == "schaefer" else f"atlas-{atlas}_"
        cache_file = os.path.join(
            cache_dir, f"{name}_{tag}kind-{kind}_regions-{n_regions}_perms-{n_perms}_nulls.npy"
        )
        if os.path.exists(cache_file):
            return np.load(cache_file)

    if distance_matrix is None:
        distance_matrix, hemi = load_distance_matrix(n_regions, kind=kind, atlas=atlas)
    if per_hemisphere is None:
        per_hemisphere = kind == "geodesic"

    # A parcel with no finite distance to anything has no position on this basis and cannot be
    # surrogated -- at Schaefer 1000 two parcels have no vertices on the fsLR-32k surface, so the
    # geodesic matrix gives them an all-NaN row. Drop them rather than feed NaN to BrainSMASH.
    placed = np.isfinite(distance_matrix).any(axis=1)
    dropped = ~np.isnan(brain_map) & ~placed
    if dropped.any():
        warnings.warn(
            f"{int(dropped.sum())} parcel(s) with data have no {kind} distances (1-based indices "
            f"{[int(i) + 1 for i in np.where(dropped)[0]]}) and are returned as NaN.",
            stacklevel=2,
        )
    valid = ~np.isnan(brain_map) & placed
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
def _corr(a, b, method):
    """Correlation coefficient only, via stats.compute_stat so the n < 3 guard applies."""
    return compute_stat(a, b, method, return_p=False)[0]


def corr_with_covar_null(brain_map, target_map, covariates, target_surrogates, method="pearson",
                         two_tailed=True):
    """Spatial correlation between a (real) brain map and a target map, optionally residualizing
    BOTH on a covariate set, tested against a BrainSMASH null.

    The null surrogates the *target* map (brain map and covariates held fixed) and recomputes the
    identical (residualized) correlation. Returns ``dict(r, p_smash, n, null_lo, null_hi, null)``
    where ``null`` is the full finite null vector.

    ``two_tailed`` (default True) tests the magnitude of the correlation (``|r|``); set False for a
    one-tailed test in the observed effect's direction (``get_null_p(..., alternative='auto')``).
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
        "p_smash": get_null_p(obs, null, alternative="two-sided" if two_tailed else "auto"),
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
    atlas="schaefer",
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
        distance_matrix, hemi = load_distance_matrix(n_regions, kind=kind, atlas=atlas)
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
            atlas=atlas,
        )
        rows[col] = corr_with_covar_null(brain_map, df_targets[col].values, covariates, surr,
                                         method=method, two_tailed=two_tailed)
    df = pd.DataFrame(rows).T
    num = ["r", "p_smash", "n", "null_lo", "null_hi"]
    df[num] = df[num].astype(float)
    df["fdr_sig"] = multipletests(df["p_smash"].values, alpha=0.05, method="fdr_bh")[0]
    return df[num + ["fdr_sig", "null"]]


def network_enrichment(brain_map, systems, stage_surrogates):
    """Per-system mean of a map tested against a spatial-autocorrelation-preserving null.

    The map is surrogated (pass ``stage_surrogates``); per-system means are recomputed per
    surrogate; the two-tailed p compares the observed mean's deviation from the null mean via
    ``get_null_p(alternative='two-sided')``. ``systems`` is any per-parcel label vector (e.g. Yeo-7 networks).
    Returns a DataFrame indexed by system with observed mean, null mean, 95% null band,
    ``p_smash`` and the full per-surrogate system-mean vector (``null``).
    """
    systems = np.asarray(systems)
    brain_map = np.asarray(brain_map, dtype=float)
    stage_surrogates = np.asarray(stage_surrogates, dtype=float)
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
            "p_smash": get_null_p(obs - center, null_vals - center, alternative="two-sided"),
        }
        nulls[net] = null_vals[np.isfinite(null_vals)]
    df = pd.DataFrame(rows).T
    df["null"] = pd.Series(nulls)
    return df
