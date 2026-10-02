"""Tests for :func:`snaplab_tools.derivs.compute_brain_states`.

The useful test has a known answer: time series built from a few fixed patterns plus noise, from which
k-means must recover those patterns as its states.
"""
import numpy as np
import pytest
import scipy as sp
from sklearn.cluster import KMeans

from snaplab_tools.derivs import compute_brain_states

N_REGIONS, N_STATES, N_TIMEPOINTS = 30, 3, 600


@pytest.fixture
def known_states():
    """Time series whose every time point is one of three fixed patterns plus a little noise."""
    rng = np.random.default_rng(0)
    patterns = rng.normal(size=(N_STATES, N_REGIONS)) * 3
    truth = rng.integers(0, N_STATES, size=N_TIMEPOINTS)
    ts = (patterns[truth] + rng.normal(scale=0.3, size=(N_TIMEPOINTS, N_REGIONS))).T
    return ts, patterns, truth


def test_recovers_the_states(known_states):
    ts, patterns, truth = known_states
    centroids, labels, inertia = compute_brain_states(ts, N_STATES, zscore=False)
    # match each centroid to its nearest true pattern; every pattern must be found once
    match = [int(np.argmin(np.linalg.norm(patterns - c, axis=1))) for c in centroids]
    assert sorted(match) == list(range(N_STATES))
    np.testing.assert_allclose(centroids, patterns[match], atol=0.1)
    # labels agree with the truth up to relabelling
    assert np.all(np.asarray(match)[labels] == truth)
    assert inertia > 0


def test_shapes_and_determinism(known_states):
    ts, _, _ = known_states
    centroids, labels, inertia = compute_brain_states(ts, 4)
    assert centroids.shape == (4, N_REGIONS)
    assert labels.shape == (N_TIMEPOINTS,)
    # k-means sums across threads, so repeated runs agree to rounding rather than bit for bit
    again = compute_brain_states(ts, 4)
    np.testing.assert_allclose(again[0], centroids, rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(again[1], labels)
    assert again[2] == pytest.approx(inertia, rel=1e-10)


def test_zscores_each_region_by_default(known_states):
    ts, _, _ = known_states
    scaled = ts * np.arange(1, N_REGIONS + 1)[:, None] + 100  # per-region scale and offset
    np.testing.assert_allclose(compute_brain_states(scaled, N_STATES)[0], compute_brain_states(ts, N_STATES)[0],
                               atol=1e-10)


def test_matches_clustering_frames_directly(known_states):
    """The same as z-scoring the (time points, regions) frames and running scikit-learn's KMeans."""
    ts, _, _ = known_states
    frames = sp.stats.zscore(ts.T, axis=0)
    kmeans = KMeans(n_clusters=5, random_state=0, n_init=10).fit(frames)
    centroids, labels, inertia = compute_brain_states(ts, 5)
    np.testing.assert_allclose(centroids, kmeans.cluster_centers_, rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(labels, kmeans.labels_)
    assert inertia == pytest.approx(kmeans.inertia_, rel=1e-10)


@pytest.mark.parametrize("bad", ["nan", "constant"])
def test_unusable_regions_are_left_out(known_states, bad):
    ts, _, _ = known_states
    ts = ts.copy()
    if bad == "nan":
        ts[[2, 7], 10] = np.nan
    else:
        ts[[2, 7], :] = 1.0
    centroids, labels, _ = compute_brain_states(ts, N_STATES)
    assert np.all(np.isnan(centroids[:, [2, 7]]))
    assert np.all(np.isfinite(np.delete(centroids, [2, 7], axis=1)))
    expected = compute_brain_states(np.delete(ts, [2, 7], axis=0), N_STATES)
    np.testing.assert_array_equal(labels, expected[1])


def test_nothing_to_cluster_raises():
    with pytest.raises(ValueError, match="nothing to cluster"):
        compute_brain_states(np.full((5, 50), np.nan), 2)
