"""Null network models: rewired connectomes that preserve spatial embedding.

Where :mod:`snaplab_tools.nulls.maps` surrogates a parcel-wise *map*, this module surrogates the
*network* itself. The question it answers is different: given that a connectome shows some
property -- efficient state transitions, a particular distribution of average controllability,
short path lengths -- is that property attributable to the network's non-trivial topology, or is
it what you would get from *any* network with the same geometry and weight statistics?

The generator is the geometry-preserving surrogate of Roberts et al. (2016). The insight it is
built on is that connectome edge weights are strongly and systematically related to the physical
distance between nodes: nearby regions are connected more strongly than distant ones, and this
weight-distance relationship alone reproduces a surprising amount of connectome topology. A null
that shuffles edges without respecting geometry therefore breaks something trivially true of the
brain, and any test against it is close to guaranteed to come out significant. These surrogates
instead hold the weight-distance relationship fixed and randomise everything else.

Concretely, :func:`geomsurr` fits the mean and variance of ``log(weight)`` as polynomial functions
of inter-nodal distance, shuffles the residuals, and inverts the fit -- so the surrogate has the
same weight-versus-distance profile as the original, but a randomised assignment of which pairs of
nodes are connected how strongly. It returns three surrogates, in increasing order of how much
they constrain:

``Wwp`` -- **weight-preserving.** The original multiset of edge weights, reassigned to a new,
distance-respecting random order. Node strengths are *not* preserved.

``Wsp`` -- **strength-preserving.** ``Wwp`` iteratively rescaled so the *distribution* of node
strengths matches the original, but assigned in the surrogate's own random order: the same set of
strengths, held by different nodes.

``Wssp`` -- **strength-sequence-preserving.** ``Wwp`` rescaled so each node recovers *its own*
original strength. The most conservative of the three: geometry and every node's strength are held
fixed, so a surviving effect cannot be explained by either.

The strength restoration behind ``Wsp`` and ``Wssp`` is an iterative fit, not an exact
construction -- rescaling rows to hit their target strengths perturbs the columns, so it converges
rather than solving outright. At the published default of ``n_iter=10`` the strengths land within
roughly 0.1-0.2% of target (correlation with the target > 0.9999), which is immaterial for the
statistics these nulls are used for but is not zero. Raise ``n_iter`` if a statistic is sensitive
to it; ``n_iter=50`` is exact to floating point and costs ~5x the (cheap) correction step.

Which to use follows from the hypothesis. For a nodal metric that is known to track strength --
average controllability being the canonical case (Gu et al., 2015) -- ``Wssp`` is the one that
makes the test non-trivial, since it removes the strength confound outright. For a network-level
statistic such as the control energy of a state transition, ``Wsp`` and ``Wssp`` reported together
separate "the strength distribution explains it" from "the strength sequence explains it".

.. note::

   ``Wsp`` and ``Wssp`` are derived under the assumption that ``W`` is undirected. ``Wwp`` is
   valid for both; for a directed connectome, use ``Wwp`` and ignore the other two.

This is a cleaned-up port of the implementation shipped with ``nctpy`` (Parkes et al., 2024,
*Nature Protocols*), which was itself translated from the original MATLAB by M. Breakspear and
J. Roberts. It is numerically identical to that implementation for the same seed.

References
----------
Roberts, J.A., Perry, A., Roberts, G., Mitchell, P.B., & Breakspear, M. (2016). Consistency-based
thresholding of the human connectome. *NeuroImage*, 145, 118-129.
https://doi.org/10.1016/j.neuroimage.2016.09.053

Roberts, J.A., Perry, A., Lord, A.R., Roberts, G., Mitchell, P.B., Smith, R.E., Calamante, F., &
Breakspear, M. (2016). The contribution of geometry to the human connectome. *NeuroImage*, 124,
379-393. https://doi.org/10.1016/j.neuroimage.2015.09.009

Parkes, L., Kim, J.Z., Stiso, J., et al. (2024). A network control theory pipeline for studying
the dynamics of the structural connectome. *Nature Protocols*.
https://doi.org/10.1038/s41596-024-01023-w
"""

from __future__ import annotations

import numpy as np
from tqdm import tqdm

__all__ = [
    'geomsurr',
    'generate_network_nulls',
]

_KINDS = ('wp', 'sp', 'ssp')


# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------
def _rank_reorder(x, scaffold):
    """Return the values of ``x`` permuted to share the rank order of ``scaffold``.

    The largest entry of the result sits where the largest entry of ``scaffold`` sits, and so on.
    This is the "amplitude adjustment" step familiar from Fourier surrogates for time series: it
    lets a surrogate keep the *exact* empirical distribution of the original while taking its
    ordering from something else.
    """
    order = np.argsort(np.argsort(np.asarray(scaffold)))
    return np.sort(np.asarray(x))[order]


def _strength_correct(W, target_strength, n_iter=10):
    """Iteratively rescale a symmetric ``W`` so node strengths approach ``target_strength``.

    Each pass scales row ``i`` by ``target_strength[i] / strength[i]`` and re-symmetrises, which
    perturbs the strengths it just fixed; repeating converges on a matrix whose strengths match
    the target while preserving the sparsity pattern and the relative weighting within each row.
    Nodes with zero strength are left disconnected rather than producing a division by zero.

    Convergence is geometric but not fast: max relative error runs to ~2e-1 after one pass, ~2e-3
    at the published default of 10, ~1e-5 at 20 and machine precision by 50.
    """
    W = np.asarray(W, dtype=float)
    n = W.shape[0]
    target_strength = np.asarray(target_strength, dtype=float)
    disconnected = W.sum(axis=0) == 0

    for _ in range(n_iter):
        strength = W.sum(axis=0)
        scale = np.divide(
            target_strength, strength, out=np.zeros(n), where=strength != 0
        )
        W = W * scale[:, np.newaxis]
        W[disconnected, :] = 0.0
        W = (W + W.T) / 2.0
    return W


# --------------------------------------------------------------------------------------------
# The generator
# --------------------------------------------------------------------------------------------
def geomsurr(W, D, nmean=3, nstd=2, seed=123, n_iter=10):
    """Geometry-preserving surrogate connectomes (Roberts et al., 2016).

    Randomises which node pairs are connected how strongly, while holding fixed the relationship
    between edge weight and inter-nodal distance -- see the module docstring for what each of the
    three returned surrogates preserves and how to choose between them.

    Parameters
    ----------
    W : (n_nodes, n_nodes) ndarray
        Adjacency matrix to rewire. Weights must be **positive** (the method works on
        ``log(weight)``); zeros denote absent edges. Not modified in place. Self-connections are
        removed before rewiring, so the surrogates always have a zero diagonal.
    D : (n_nodes, n_nodes) ndarray
        Inter-nodal distance, in any consistent unit. Typically Euclidean distance between parcel
        centroids; must be finite wherever ``W`` is non-zero.
    nmean : int, default=3
        Polynomial order for the weight-versus-distance *mean* fit.
    nstd : int, default=2
        Polynomial order for the weight-versus-distance *standard deviation* fit.
    seed : int, default=123
        Seed for the edge shuffle. Only local RNG state is used; the global NumPy seed is left
        alone.
    n_iter : int, default=10
        Passes of the iterative strength correction behind ``Wsp`` and ``Wssp``. The default
        reproduces the published implementation and lands strengths within ~0.2% of target; raise
        it (50 is exact to floating point) if your statistic is sensitive to that residual.

    Returns
    -------
    Wwp : (n_nodes, n_nodes) ndarray
        Space- and weight-distribution-preserving surrogate. The weight multiset is preserved
        exactly.
    Wsp : (n_nodes, n_nodes) ndarray
        Space- and strength-distribution-preserving surrogate, to within ``n_iter``. Assumes
        ``W`` is undirected.
    Wssp : (n_nodes, n_nodes) ndarray
        Space- and strength-sequence-preserving surrogate, to within ``n_iter``. Assumes ``W`` is
        undirected.

    Notes
    -----
    Directedness is detected from ``W``: if ``W`` is exactly symmetric the rewiring is done on one
    triangle and mirrored, otherwise every edge is treated independently.

    Examples
    --------
    >>> import numpy as np
    >>> from snaplab_tools.datasets import make_connectome, schaefer_geometry
    >>> from scipy.spatial.distance import squareform, pdist
    >>> W = make_connectome(n_regions=100, seed=0)
    >>> D = squareform(pdist(schaefer_geometry(100)['centroids']))
    >>> Wwp, Wsp, Wssp = geomsurr(W, D, seed=0)
    >>> triu = np.triu_indices_from(W, k=1)
    >>> bool(np.allclose(np.sort(Wwp[triu]), np.sort(W[triu])))     # weight multiset exact
    True
    >>> bool(np.allclose(Wssp.sum(axis=0), W.sum(axis=0), rtol=0.01))  # strengths ~restored
    True
    """
    W = np.asarray(W, dtype=float).copy()  # copied: the caller's matrix is never modified
    D = np.asarray(D, dtype=float)
    n = W.shape[0]
    if W.shape != (n, n) or D.shape != (n, n):
        raise ValueError(f"W and D must both be square and the same size; got {W.shape} and {D.shape}.")

    directed = not np.array_equal(W, W.T)
    np.fill_diagonal(W, 0.0)

    # For an undirected W the two triangles are redundant; rewire one and mirror it back.
    W_work = W if directed else np.tril(W)

    nz = np.where(W_work != 0)
    w = W_work[nz]
    d = D[nz]
    if np.any(w <= 0):
        raise ValueError(
            "geomsurr works on log-weights, so all non-zero entries of W must be positive; "
            f"found {int(np.sum(w < 0))} negative and {int(np.sum(w == 0))} zero. Take absolute "
            "values or threshold W first."
        )
    if not np.all(np.isfinite(d)):
        raise ValueError(
            "D has non-finite entries where W is non-zero. A per-hemisphere geodesic matrix has "
            "NaN across hemispheres and cannot be used here; use Euclidean centroid distances."
        )
    logw = np.log(w)

    # 1. Strip the distance-dependent mean of log-weight, to order `nmean`.
    fit_mean = np.polyfit(d, logw, nmean)
    residual = logw - np.polyval(fit_mean, d)

    # 2. Strip the distance-dependent spread, to order `nstd`, leaving standardised residuals.
    fit_std = np.polyfit(d, np.abs(residual), nstd)
    standardised = residual / np.polyval(fit_std, d)

    # 3. Shuffle the standardised residuals -- this is the only stochastic step. RandomState (not
    #    default_rng) is deliberate: it reproduces the published nctpy/MATLAB stream exactly.
    shuffled = np.random.RandomState(seed).permutation(standardised)

    # 4. Put the geometry back: restore the distance-dependent spread, then the mean.
    surrogate_logw = shuffled * np.polyval(fit_std, d) + np.polyval(fit_mean, d)

    # 5. Use those surrogate weights only as a *scaffold* for ordering, and lay the original
    #    weights onto it. The surrogate then holds the exact empirical weight distribution of the
    #    original, in a new distance-respecting random order.
    Wwp = np.zeros((n, n))
    Wwp[nz] = np.exp(_rank_reorder(logw, surrogate_logw))

    if not directed:
        Wwp = Wwp + Wwp.T

    # 6. Restore node strengths. Both variants assume W is undirected.
    strength = W.sum(axis=0)
    Wsp = _strength_correct(Wwp, _rank_reorder(strength, Wwp.sum(axis=0)), n_iter=n_iter)
    Wssp = _strength_correct(Wwp, strength, n_iter=n_iter)

    return Wwp, Wsp, Wssp


# --------------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------------
def generate_network_nulls(W, D, n_perms=1000, kind='ssp', statistic=None, seed=0,
                           nmean=3, nstd=2, n_iter=10, progress=True,
                           desc='surrogate networks'):
    """Build a null distribution by recomputing a statistic over ``n_perms`` rewired connectomes.

    Wraps the loop that :func:`geomsurr` is almost always used inside: rewire, recompute, collect.
    Feed the result to :func:`snaplab_tools.stats.get_null_p` for a p-value, or
    :func:`snaplab_tools.plotting.plotting.null_plot` for the figure.

    Parameters
    ----------
    W, D : (n_nodes, n_nodes) ndarray
        Adjacency and inter-nodal distance matrices, as for :func:`geomsurr`.
    n_perms : int, default=1000
        Number of surrogate networks.
    kind : {'wp', 'sp', 'ssp'} or sequence of those, default='ssp'
        Which surrogate(s) to evaluate. All three come out of one :func:`geomsurr` call, so asking
        for several costs no extra rewiring -- pass ``('sp', 'ssp')`` to report both nulls rather
        than running this function twice.
    statistic : callable, optional
        ``statistic(A) -> float or ndarray``, applied to each surrogate. If omitted, the surrogate
        matrices themselves are returned, which is ``n_perms * n_nodes**2 * 8`` bytes -- 6.4 GB for
        5,000 surrogates of a 400-node connectome. Pass a statistic unless you have a reason not to.
    seed : int, default=0
        Surrogate ``i`` uses ``seed + i``, so the surrogate sequence is reproducible and two runs
        with different ``n_perms`` share a prefix.
    nmean, nstd, n_iter : int
        Polynomial orders and strength-correction passes, forwarded to :func:`geomsurr`.
    progress : bool, default=True
        Show a tqdm progress bar. Worth leaving on: a few thousand surrogates of a 400-node
        connectome takes minutes, and longer again if ``statistic`` is expensive.
    desc : str
        Label for the progress bar.

    Returns
    -------
    ndarray or dict of ndarray
        For a single ``kind``, an array whose first axis is ``n_perms`` and whose remaining axes
        are the shape of one ``statistic`` return value -- ``(n_perms,)`` for a scalar statistic,
        ``(n_perms, n_nodes)`` for a nodal one. For a sequence of kinds, a dict keyed by kind.

    Examples
    --------
    A nodal null for average controllability, the test run in Parkes et al. (2024). The statistic
    normalises each surrogate exactly as the observed connectome was normalised -- the null is only
    interpretable if every step downstream of the rewiring is identical::

        from nctpy.metrics import ave_control
        from nctpy.utils import matrix_normalization
        from snaplab_tools.nulls import generate_network_nulls
        from snaplab_tools.stats import get_null_p, get_fdr_p

        null = generate_network_nulls(
            W, D, n_perms=5000, kind='ssp',
            statistic=lambda A: ave_control(matrix_normalization(A, system='continuous'),
                                            system='continuous'),
        )                                            # -> (5000, n_nodes)

        p = [get_null_p(observed[i], null[:, i], alternative='greater')
             for i in range(len(observed))]
        p = get_fdr_p(np.array(p))

    Two nulls for one scalar statistic, from a single pass of rewiring::

        nulls = generate_network_nulls(W, D, n_perms=5000, kind=('sp', 'ssp'),
                                       statistic=my_energy)
        nulls['sp'].shape, nulls['ssp'].shape       # -> ((5000,), (5000,))
    """
    single = isinstance(kind, str)
    kinds = (kind,) if single else tuple(kind)
    unknown = [k for k in kinds if k not in _KINDS]
    if unknown:
        raise ValueError(f"kind must be one of {_KINDS} (or a sequence of them); got {unknown}.")
    if n_perms < 1:
        raise ValueError(f"n_perms must be at least 1; got {n_perms}.")

    out = {}
    for i in tqdm(range(n_perms), desc=desc, disable=not progress):
        surrogates = dict(zip(_KINDS, geomsurr(W, D, nmean=nmean, nstd=nstd, seed=seed + i,
                                               n_iter=n_iter)))
        for k in kinds:
            value = surrogates[k] if statistic is None else np.asarray(statistic(surrogates[k]),
                                                                       dtype=float)
            # Allocate on the first pass, once the statistic's output shape is known.
            if k not in out:
                out[k] = np.empty((n_perms,) + np.shape(value), dtype=float)
            out[k][i] = value

    return out[kinds[0]] if single else out
