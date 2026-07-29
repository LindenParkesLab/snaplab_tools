"""Tests for geometry-preserving null network models.

Three kinds of test here, in descending order of what they protect.

The first is a regression test against ``fixtures/geomsurr_reference.npz``, which holds surrogates
produced by the *published* ``nctpy`` implementation (Parkes et al., 2024, Nature Protocols) at the
commit the paper was released from. ``geomsurr`` is a port, and the point of a port is that it
agrees with the thing it was ported from; if it drifts, results computed here stop being comparable
to results computed there. See ``fixtures/make_geomsurr_reference.py`` for provenance.

The second checks the structural invariants each surrogate is *defined* by -- ``Wwp`` preserves the
weight multiset, ``Wssp`` preserves each node's strength -- since those are the claims a user reads
off the docstring and relies on when choosing a null.

The third covers the ways a caller can get this wrong: non-positive weights, a geodesic distance
matrix with NaN across hemispheres, self-connections that silently vanish.
"""
import warnings
from pathlib import Path

import numpy as np
import pytest

from snaplab_tools.nulls import generate_network_nulls, geomsurr
from snaplab_tools.nulls.networks import _rank_reorder, _strength_correct

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "geomsurr_reference.npz"
REFERENCE_SEEDS = (0, 1, 123)


@pytest.fixture(scope="module")
def reference():
    """Inputs and published outputs from the pinned nctpy implementation."""
    if not FIXTURE.exists():  # pragma: no cover -- fixture is committed
        pytest.skip(f"missing {FIXTURE.name}; regenerate with fixtures/make_geomsurr_reference.py")
    return np.load(FIXTURE)


@pytest.fixture
def network(reference):
    """The undirected reference network and its distance matrix, as (W, D)."""
    return reference["W_und"], reference["D_und"]


# --------------------------------------------------------------------------------------------
# Agreement with the published implementation
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("label", ["und", "dir"])
@pytest.mark.parametrize("seed", REFERENCE_SEEDS)
def test_matches_published_nctpy_implementation(reference, label, seed):
    """Bit-for-bit agreement with the nctpy release the Nature Protocols paper was published from.

    Exact equality, not allclose: the port is meant to reproduce the original's arithmetic, so any
    difference at all is a signal worth failing on rather than a tolerance to be widened.
    """
    W, D = reference[f"W_{label}"], reference[f"D_{label}"]

    surrogates = geomsurr(W, D, seed=seed)

    expected = [reference[f"{name}_{label}_{seed}"] for name in ("wwp", "wsp", "wssp")]
    for got, want, name in zip(surrogates, expected, ("Wwp", "Wsp", "Wssp")):
        assert np.array_equal(got, want), f"{name} diverged from published nctpy output"


def test_published_reference_covers_both_directedness_paths(reference):
    """The fixture is only meaningful if it exercises the symmetric and asymmetric branches."""
    assert np.array_equal(reference["W_und"], reference["W_und"].T)
    assert not np.array_equal(reference["W_dir"], reference["W_dir"].T)


# --------------------------------------------------------------------------------------------
# What each surrogate preserves
# --------------------------------------------------------------------------------------------
def test_wwp_preserves_the_weight_multiset_exactly(network):
    """Wwp reassigns the original weights; it never invents or rescales one."""
    W, D = network
    Wwp, _, _ = geomsurr(W, D, seed=0)

    triu = np.triu_indices_from(W, k=1)
    assert np.allclose(np.sort(Wwp[triu]), np.sort(W[triu]))


def test_wssp_restores_each_nodes_own_strength(network):
    """Wssp is the strength-*sequence*-preserving null: node i recovers node i's strength.

    Only approximately, at the published default of n_iter=10 -- the correction is an iterative
    fit, so this asserts the documented ~0.2% rather than equality.
    """
    W, D = network
    _, _, Wssp = geomsurr(W, D, seed=0)

    observed, target = Wssp.sum(axis=0), W.sum(axis=0)
    assert np.max(np.abs(observed - target) / target) < 0.01
    assert np.corrcoef(observed, target)[0, 1] > 0.999


def test_wsp_preserves_the_strength_distribution_but_reassigns_it(network):
    """Wsp holds the *set* of strengths, moved to different nodes -- otherwise it would be Wssp."""
    W, D = network
    _, Wsp, _ = geomsurr(W, D, seed=0)

    target = W.sum(axis=0)
    assert np.allclose(np.sort(Wsp.sum(axis=0)), np.sort(target), rtol=0.01)
    # Same multiset, different assignment: the per-node match must be clearly worse than Wssp's.
    assert np.max(np.abs(Wsp.sum(axis=0) - target) / target) > 0.05


def test_raising_n_iter_tightens_strength_preservation(network):
    """The residual in Wssp is a convergence limit, so more passes must shrink it."""
    W, D = network
    target = W.sum(axis=0)

    errors = [
        np.max(np.abs(geomsurr(W, D, seed=0, n_iter=n)[2].sum(axis=0) - target) / target)
        for n in (1, 10, 50)
    ]

    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 1e-9  # 50 passes is exact for practical purposes


def test_preserves_the_weight_distance_relationship(network):
    """The defining property: weight-versus-distance survives, edge assignment does not.

    This is what separates these nulls from a naive edge shuffle, so it is worth asserting directly
    rather than trusting that the polynomial fits were inverted correctly.
    """
    W, D = network
    Wwp, _, _ = geomsurr(W, D, seed=0)

    def weight_distance_corr(A):
        edges = A > 0
        return np.corrcoef(np.log(A[edges]), D[edges])[0, 1]

    assert weight_distance_corr(Wwp) == pytest.approx(weight_distance_corr(W), abs=0.1)
    # ...while which edge carries which weight has genuinely changed.
    assert not np.allclose(Wwp, W)


def test_edge_set_is_untouched_so_binary_statistics_have_no_null(network):
    """These surrogates redistribute weights over a *fixed* edge set; they do not rewire.

    The binary adjacency matrix is therefore identical to the original's, and any statistic
    computed on the unweighted graph -- degree, binary path length, binary clustering -- would get
    a zero-variance null. That is a real way to draw a wrong conclusion, so it is asserted here and
    warned about in the module docstring rather than left for a user to discover.
    """
    W, D = network

    for surrogate in geomsurr(W, D, seed=0):
        assert np.array_equal(surrogate > 0, W > 0)
        assert np.array_equal((surrogate > 0).sum(axis=0), (W > 0).sum(axis=0))  # degree sequence


def test_preserves_edge_count_and_symmetry(network):
    """Sparsity is structural: the surrogates fill exactly the edges the rewiring assigned."""
    W, D = network

    for surrogate in geomsurr(W, D, seed=0):
        assert surrogate.shape == W.shape
        assert np.allclose(surrogate, surrogate.T)
        assert np.all(np.diag(surrogate) == 0)
        assert np.all(surrogate >= 0)
        assert np.count_nonzero(surrogate) == np.count_nonzero(W)


def test_directed_input_yields_an_asymmetric_wwp(reference):
    """A directed W must not be silently symmetrised -- Wwp is the valid output in that case."""
    Wwp, _, _ = geomsurr(reference["W_dir"], reference["D_dir"], seed=0)

    assert not np.allclose(Wwp, Wwp.T)


# --------------------------------------------------------------------------------------------
# Determinism and side effects
# --------------------------------------------------------------------------------------------
def test_same_seed_reproduces_and_different_seeds_diverge(network):
    W, D = network

    assert np.array_equal(geomsurr(W, D, seed=5)[0], geomsurr(W, D, seed=5)[0])
    assert not np.array_equal(geomsurr(W, D, seed=5)[0], geomsurr(W, D, seed=6)[0])


def test_does_not_modify_the_callers_matrices(network):
    """The published implementation zeroed the caller's diagonal in place; this must not."""
    W, D = network
    W_input = W.copy()
    W_input[np.diag_indices_from(W_input)] = 5.0  # self-connections, as real connectomes have
    before_W, before_D = W_input.copy(), D.copy()

    geomsurr(W_input, D, seed=0, verbose=False)

    assert np.array_equal(W_input, before_W)
    assert np.array_equal(D, before_D)


def test_does_not_disturb_the_global_numpy_rng(network):
    """The original called np.random.seed, which silently reseeded the caller's stream."""
    W, D = network

    np.random.seed(42)
    expected = np.random.rand()

    np.random.seed(42)
    geomsurr(W, D, seed=99)
    assert np.random.rand() == expected


# --------------------------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------------------------
def test_rejects_negative_weights(network):
    """Weights go through a logarithm, so a signed matrix would return NaN rather than fail."""
    W, D = network
    W_signed = W.copy()
    W_signed[0, 1] = W_signed[1, 0] = -1.0

    with pytest.raises(ValueError, match="positive"):
        geomsurr(W_signed, D, seed=0)


def test_rejects_nonfinite_distances(network):
    """Guards the likely mistake: this package's geodesic matrix is NaN across hemispheres."""
    W, D = network
    D_geodesic = D.copy()
    i, j = (int(a) for a in np.argwhere(W > 0)[0])  # NaN must land on an edge to be seen
    D_geodesic[i, j] = D_geodesic[j, i] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        geomsurr(W, D_geodesic, seed=0)


def test_nonfinite_distance_on_an_absent_edge_is_harmless(network):
    """Only distances the fit actually consumes matter, so a NaN off the edge set is fine."""
    W, D = network
    D_sparse = D.copy()
    i, j = (int(a) for a in np.argwhere((W == 0) & ~np.eye(len(W), dtype=bool))[0])
    D_sparse[i, j] = D_sparse[j, i] = np.nan

    assert np.all(np.isfinite(geomsurr(W, D_sparse, seed=0)[0]))


def test_rejects_mismatched_shapes(network):
    W, D = network

    with pytest.raises(ValueError, match="square"):
        geomsurr(W, D[:-1, :-1], seed=0)


def test_warns_that_self_connections_are_dropped(network):
    """Self-loops cannot be rewired, so they vanish -- which silently changes what the null is."""
    W, D = network
    W_self = W.copy()
    W_self[np.diag_indices_from(W_self)] = 5.0

    with pytest.warns(UserWarning, match="self-connection"):
        surrogates = geomsurr(W_self, D, seed=0)

    assert all(np.all(np.diag(s) == 0) for s in surrogates)


def test_self_connection_warning_is_silenceable_and_not_spurious(network):
    W, D = network
    W_self = W.copy()
    W_self[np.diag_indices_from(W_self)] = 5.0

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes a failure
        geomsurr(W_self, D, seed=0, verbose=False)
        geomsurr(W, D, seed=0)  # already zero-diagonal: nothing to warn about


# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------
def test_rank_reorder_takes_values_from_x_and_order_from_scaffold():
    x = np.array([10.0, 20.0, 30.0, 40.0])
    scaffold = np.array([0.5, 0.1, 0.9, 0.3])  # ranks: 2nd, 0th, 3rd, 1st

    result = _rank_reorder(x, scaffold)

    assert np.array_equal(result, [30.0, 10.0, 40.0, 20.0])
    assert np.array_equal(np.sort(result), np.sort(x))
    assert np.array_equal(np.argsort(result), np.argsort(scaffold))


def test_strength_correct_leaves_disconnected_nodes_alone_without_dividing_by_zero():
    """A zero-strength node has nothing to rescale; it must stay disconnected, not become NaN."""
    W = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a divide-by-zero RuntimeWarning would fail here
        result = _strength_correct(W, np.array([2.0, 2.0, 4.0]), n_iter=10)

    assert np.all(np.isfinite(result))
    assert np.all(result[2, :] == 0) and np.all(result[:, 2] == 0)
    assert result[:2, :2].sum(axis=0) == pytest.approx([2.0, 2.0], abs=1e-6)


# --------------------------------------------------------------------------------------------
# generate_network_nulls
# --------------------------------------------------------------------------------------------
def test_scalar_statistic_gives_one_value_per_permutation(network):
    W, D = network

    null = generate_network_nulls(W, D, n_perms=5, statistic=np.sum, progress=False)

    assert null.shape == (5,)
    assert np.all(np.isfinite(null))


def test_nodal_statistic_gives_one_row_per_permutation(network):
    W, D = network

    null = generate_network_nulls(W, D, n_perms=5, statistic=lambda A: A.sum(axis=0),
                                 progress=False)

    assert null.shape == (5, W.shape[0])


def test_omitting_the_statistic_returns_the_surrogate_matrices(network):
    W, D = network

    null = generate_network_nulls(W, D, n_perms=3, progress=False)

    assert null.shape == (3,) + W.shape
    assert np.array_equal(null[0], geomsurr(W, D, seed=0)[2])  # kind='ssp' by default


def test_requesting_several_kinds_returns_a_dict_keyed_by_kind(network):
    W, D = network

    null = generate_network_nulls(W, D, n_perms=4, kind=("wp", "sp", "ssp"), statistic=np.sum,
                                 progress=False)

    assert set(null) == {"wp", "sp", "ssp"}
    assert all(v.shape == (4,) for v in null.values())


def test_several_kinds_match_separate_single_kind_runs(network):
    """The multi-kind path shares one rewiring per permutation; that must not change the answer."""
    W, D = network
    kwargs = dict(n_perms=4, statistic=np.sum, progress=False)

    combined = generate_network_nulls(W, D, kind=("sp", "ssp"), **kwargs)

    for kind in ("sp", "ssp"):
        assert np.array_equal(combined[kind], generate_network_nulls(W, D, kind=kind, **kwargs))


def test_surrogate_i_uses_seed_plus_i(network):
    """Documented so that a longer run is a superset of a shorter one, not a different null."""
    W, D = network

    null = generate_network_nulls(W, D, n_perms=3, seed=10, statistic=np.sum, progress=False)

    expected = [np.sum(geomsurr(W, D, seed=10 + i)[2]) for i in range(3)]
    assert np.allclose(null, expected)


def test_longer_runs_extend_shorter_ones(network):
    W, D = network
    kwargs = dict(statistic=np.sum, progress=False)

    short = generate_network_nulls(W, D, n_perms=3, **kwargs)
    long = generate_network_nulls(W, D, n_perms=6, **kwargs)

    assert np.allclose(long[:3], short)


def test_self_connection_warning_fires_once_not_once_per_permutation(network):
    """A per-permutation warning would bury the console over thousands of surrogates."""
    W, D = network
    W_self = W.copy()
    W_self[np.diag_indices_from(W_self)] = 5.0

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        generate_network_nulls(W_self, D, n_perms=20, statistic=np.sum, progress=False)

    assert sum("self-connection" in str(w.message) for w in caught) == 1


@pytest.mark.parametrize("kind", ["strength", "WSSP", ("sp", "nope")])
def test_rejects_unknown_kinds(network, kind):
    W, D = network

    with pytest.raises(ValueError, match="kind must be"):
        generate_network_nulls(W, D, n_perms=2, kind=kind, progress=False)


def test_rejects_nonpositive_n_perms(network):
    W, D = network

    with pytest.raises(ValueError, match="n_perms"):
        generate_network_nulls(W, D, n_perms=0, statistic=np.sum, progress=False)
