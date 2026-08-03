"""Tests for the bundled Schaefer resources behind the spatial null models.

These protect the *data* rather than the arithmetic. Every published Schaefer2018 7-network
resolution (100 to 1000 in steps of 100) ships with a parcellation, a centroid list and a prebuilt
geodesic distance matrix, and the promise made in the docs is that
``load_distance_matrix``/``schaefer_geometry`` work offline at any of them. A missing or truncated
file would only show up when someone ran an analysis at that resolution, which is exactly when it
is most expensive to discover.

The structural checks (symmetry, zero diagonal, block-diagonal NaN) are what the BrainSMASH
generator assumes about the basis. The 1000-parcel checks pin an upstream quirk: the CBIG fsLR-32k
dlabel declares 1000 parcels but assigns vertices to only 998, so two parcels have no geodesic
distances at all. That is a fact about the atlas, not a bug here, and the point of pinning it is
that the handling stays deliberate -- NaN and a warning -- rather than degrading into silence.
"""
import warnings

import numpy as np
import pytest

import pandas as pd

from snaplab_tools.datasets import _SUPPORTED_RESOLUTIONS, schaefer_geometry, schaefer_systems
from snaplab_tools.nulls import generate_surrogates, load_distance_matrix
from snaplab_tools.nulls.maps import (
    _centroid_csv_path,
    _dlabel_path,
    _distance_cache_path,
    _hemi_cache_path,
)

RESOLUTIONS = list(_SUPPORTED_RESOLUTIONS)

# Parcels declared by the Schaefer 1000 label table that have no vertices on the fsLR-32k surface,
# as 0-based indices into the distance matrix.
PLACELESS_1000 = (532, 902)

YEO_7 = {"Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"}


def test_supports_every_published_resolution():
    """The full Schaefer2018 set, not a subset -- the claim the docs make."""
    assert RESOLUTIONS == list(range(100, 1100, 100))


@pytest.mark.parametrize("n_regions", RESOLUTIONS)
def test_resources_are_bundled(n_regions):
    """All four files per resolution exist, so nothing has to be built or downloaded at run time."""
    for path in (
        _dlabel_path(n_regions),
        _centroid_csv_path(n_regions),
        _distance_cache_path(n_regions, "geodesic"),
        _hemi_cache_path(n_regions, "geodesic"),
    ):
        assert path.exists(), f"missing bundled resource {path.name}"


@pytest.mark.parametrize("n_regions", RESOLUTIONS)
def test_geodesic_matrix_has_the_shape_brainsmash_assumes(n_regions):
    """Square, symmetric, zero diagonal, and block-diagonal by hemisphere.

    The cross-hemisphere block is NaN by construction -- the L and R fsLR surfaces are
    disconnected, so there is no geodesic between them -- and that NaN block is what forces
    ``per_hemisphere=True`` downstream.
    """
    D, hemi = load_distance_matrix(n_regions, kind="geodesic")
    assert D.shape == (n_regions, n_regions)
    assert hemi.shape == (n_regions,)
    assert set(np.unique(hemi)) == {"L", "R"}

    placed = np.isfinite(D).any(axis=1)
    same_hemi = hemi[:, None] == hemi[None, :]
    both_placed = placed[:, None] & placed[None, :]

    np.testing.assert_allclose(D[both_placed], D.T[both_placed])
    np.testing.assert_allclose(np.diag(D)[placed], 0.0)
    assert np.isnan(D[~same_hemi]).all(), "cross-hemisphere distances must be NaN"
    assert np.isfinite(D[same_hemi & both_placed]).all()
    assert (D[same_hemi & both_placed] >= 0).all()
    # Sanity bound: the longest within-hemisphere geodesic on this surface is a little over 200 mm.
    assert D[same_hemi & both_placed].max() < 300.0


@pytest.mark.parametrize("n_regions", RESOLUTIONS)
def test_euclidean_basis_covers_every_parcel(n_regions):
    """Centroids exist for all parcels at every resolution, including the two with no vertices."""
    D, hemi = load_distance_matrix(n_regions, kind="euclidean")
    assert D.shape == (n_regions, n_regions)
    assert np.isfinite(D).all()
    assert set(np.unique(hemi)) == {"L", "R"}


@pytest.mark.parametrize("n_regions", RESOLUTIONS)
def test_geometry_and_systems_line_up(n_regions):
    """One row per parcel across every field, and systems parse to the Yeo 7."""
    geom = schaefer_geometry(n_regions)
    assert geom["distance_matrix"].shape == (n_regions, n_regions)
    assert geom["centroids"].shape == (n_regions, 3)
    assert len(geom["roi_names"]) == n_regions
    assert np.isfinite(geom["centroids"]).all()
    assert set(geom["systems"]) <= YEO_7

    systems = schaefer_systems(n_regions)
    assert len(systems) == n_regions
    np.testing.assert_array_equal(systems, geom["systems"])


@pytest.mark.parametrize("atlas,n_regions", [("schaefer", n) for n in RESOLUTIONS] + [("glasser", 360)])
def test_matrix_row_order_matches_parcel_order(atlas, n_regions):
    """The failure mode every structural test misses: a matrix whose rows are in the wrong order.

    It would be symmetric, zero-diagonal, correctly blocked by hemisphere and entirely plausible --
    and it would silently attach every parcel's distances to a different parcel, which for a
    spatial null is the whole ballgame. Distance matrices and centroid coordinates come from
    different files (surface parcellation vs centroid CSV), so the agreement is worth checking.

    A parcel's geodesically nearest neighbour must also be Euclidean-near. The same statistic on a
    row-permuted copy of the matrix is included so the test demonstrably detects what it looks for
    rather than passing vacuously.
    """
    D, hemi = load_distance_matrix(n_regions, atlas=atlas)
    xyz = pd.read_csv(_centroid_csv_path(n_regions, atlas))[["R", "A", "S"]].values.astype(float)
    euclidean = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)

    def median_step_to_geodesic_neighbour(matrix):
        matrix = np.where(np.isfinite(matrix), matrix, np.inf).copy()
        np.fill_diagonal(matrix, np.inf)
        placed = np.isfinite(matrix).any(axis=1)
        nearest = matrix[placed].argmin(axis=1)
        return np.median(euclidean[np.arange(n_regions)[placed], nearest])

    scrambled = D.copy()
    order = np.arange(n_regions)
    rng = np.random.default_rng(0)
    for h in ("L", "R"):                      # permute within hemisphere: a subtle scramble
        block = np.where(hemi == h)[0]
        order[block] = rng.permutation(block)
    scrambled = scrambled[np.ix_(order, order)]

    correct = median_step_to_geodesic_neighbour(D)
    wrong = median_step_to_geodesic_neighbour(scrambled)
    assert correct < 0.5 * wrong, (
        f"geodesic neighbours are {correct:.1f} mm apart, barely better than the {wrong:.1f} mm of "
        f"a scrambled matrix -- the row order may not match the parcel order"
    )


def test_rejects_a_resolution_with_no_bundled_matrix():
    """250 is not a Schaefer resolution; the error should say what is available."""
    with pytest.raises(ValueError, match="n_regions must be one of"):
        schaefer_geometry(250)


# --------------------------------------------------------------------------------------------
# The Schaefer 1000 quirk: two declared parcels have no place on the surface
# --------------------------------------------------------------------------------------------
def test_schaefer1000_has_exactly_two_placeless_parcels():
    """Pinned deliberately: if upstream ever fixes this, the test should tell us."""
    D, hemi = load_distance_matrix(1000, kind="geodesic")
    placeless = np.where(~np.isfinite(D).any(axis=1))[0]
    np.testing.assert_array_equal(placeless, PLACELESS_1000)

    # They still get a hemisphere, read off the label-table name, so they are identifiable.
    names = schaefer_geometry(1000)["roi_names"]
    assert list(names[list(PLACELESS_1000)]) == ["7Networks_RH_Vis_33", "7Networks_RH_Cont_Cing_1"]
    assert set(hemi[list(PLACELESS_1000)]) == {"R"}


@pytest.mark.parametrize("n_regions", RESOLUTIONS)
def test_only_schaefer1000_is_incomplete(n_regions):
    """Every other resolution places all of its parcels on the surface."""
    D, _ = load_distance_matrix(n_regions, kind="geodesic")
    n_placeless = int((~np.isfinite(D).any(axis=1)).sum())
    assert n_placeless == (2 if n_regions == 1000 else 0)


# --------------------------------------------------------------------------------------------
# Glasser (HCP-MMP1.0)
# --------------------------------------------------------------------------------------------
def test_glasser_matrix_is_bundled_but_the_parcellation_is_not():
    """The licence line, encoded.

    HCP-MMP1.0 comes via BALSA under the HCP Data Use Terms, which restrict redistribution, so the
    derived distance matrix and centroids ship but the parcellation itself does not. Asking for the
    dlabel should say why rather than raise a bare FileNotFoundError.
    """
    assert _distance_cache_path(360, "geodesic", "glasser").exists()
    assert _hemi_cache_path(360, "geodesic", "glasser").exists()
    assert _centroid_csv_path(360, "glasser").exists()

    with pytest.raises(ValueError, match="not bundled"):
        _dlabel_path(360, "glasser")


def test_glasser_geodesic_basis():
    """Same structural contract as Schaefer, and all 360 areas make it onto the surface."""
    D, hemi = load_distance_matrix(360, kind="geodesic", atlas="glasser")
    assert D.shape == (360, 360)
    assert np.isfinite(D).any(axis=1).all(), "every HCP-MMP1.0 area should have a surface position"

    same_hemi = hemi[:, None] == hemi[None, :]
    np.testing.assert_allclose(D, D.T, equal_nan=True)
    np.testing.assert_allclose(np.diag(D), 0.0)
    assert np.isnan(D[~same_hemi]).all()
    assert np.isfinite(D[same_hemi]).all()
    assert D[same_hemi].max() < 300.0


def test_glasser_is_right_hemisphere_first():
    """The ordering trap: HCP-MMP1.0 runs areas 1-180 right, 181-360 left -- the reverse of
    Schaefer. Anything that reorders one atlas to match the other silently mislabels every parcel,
    so pin it here rather than trust it."""
    _, hemi = load_distance_matrix(360, kind="geodesic", atlas="glasser")
    assert set(hemi[:180]) == {"R"}
    assert set(hemi[180:]) == {"L"}

    _, hemi_schaefer = load_distance_matrix(400, kind="geodesic")
    assert hemi_schaefer[0] == "L", "Schaefer runs the other way round"


def test_glasser_centroids_agree_with_the_geodesic_basis():
    """The two ship from different sources -- centroids derived from the volumetric atlas in
    data/atlases, geodesics built from the BALSA surface parcellation -- so their parcel order has
    to be checked, not assumed. Hemisphere assignment is the sharpest available cross-check."""
    centroids = pd.read_csv(_centroid_csv_path(360, "glasser"))
    assert len(centroids) == 360
    assert list(centroids["ROI Label"]) == list(range(1, 361))

    _, hemi_geodesic = load_distance_matrix(360, kind="geodesic", atlas="glasser")
    _, hemi_euclidean = load_distance_matrix(360, kind="euclidean", atlas="glasser")
    np.testing.assert_array_equal(hemi_euclidean, hemi_geodesic)

    # R_* areas sit in the right hemisphere (positive R), L_* in the left. Same sign convention as
    # the Schaefer CSVs, and a scrambled parcel order would break it immediately.
    is_right = centroids["ROI Name"].str.startswith("R_").values
    assert (centroids["R"].values[is_right] > 0).mean() > 0.95
    assert (centroids["R"].values[~is_right] < 0).mean() > 0.95


def test_rejects_an_unknown_atlas():
    with pytest.raises(ValueError, match="atlas must be one of"):
        load_distance_matrix(400, atlas="brodmann")


def test_surrogates_drop_parcels_with_no_distances_and_say_so():
    """A parcel with data but no position cannot be surrogated; it comes back NaN, loudly.

    Driven with a hand-built basis rather than the real 1000-parcel one so the test stays fast --
    the behaviour under test is the all-NaN row, not the atlas.
    """
    rng = np.random.default_rng(0)
    coords = rng.normal(size=(12, 1)) * 20.0
    D = np.abs(coords - coords.T)
    D[5, :] = D[:, 5] = np.nan          # parcel 5 has no place on the surface
    hemi = np.array(["L"] * 12, dtype="<U1")
    brain_map = rng.normal(size=12)

    with pytest.warns(UserWarning, match="no geodesic distances"):
        surrogates = generate_surrogates(
            brain_map, n_perms=2, distance_matrix=D, hemi=hemi, per_hemisphere=True,
        )

    assert surrogates.shape == (2, 12)
    assert np.isnan(surrogates[:, 5]).all()
    assert np.isfinite(np.delete(surrogates, 5, axis=1)).all()


def test_surrogates_are_quiet_when_every_parcel_is_placed():
    """The warning above must not fire on the nine resolutions that have nothing missing."""
    rng = np.random.default_rng(0)
    coords = rng.normal(size=(12, 1)) * 20.0
    D = np.abs(coords - coords.T)
    hemi = np.array(["L"] * 12, dtype="<U1")

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        generate_surrogates(
            rng.normal(size=12), n_perms=2, distance_matrix=D, hemi=hemi, per_hemisphere=True,
        )
