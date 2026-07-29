"""Regenerate ``geomsurr_reference.npz``: surrogates from the *published* geomsurr implementation.

``snaplab_tools.nulls.geomsurr`` is a port, and its value depends on being numerically identical to
the implementation shipped with ``nctpy`` for the Nature Protocols paper (Parkes et al., 2024) --
otherwise results computed with this package cannot be compared against results computed with that
one. This script captures the original's output so ``tests/test_nulls_networks.py`` can enforce
that, rather than the guarantee resting on someone having checked it once by hand.

The reference is pinned to the commit of ``nctpy`` that the paper was published from::

    NCTPY_COMMIT = c01e61bd2662bb50ac9c06468174e9723cadb17a   ("post acceptance release")

Run it only to add cases or to deliberately re-pin; it needs network access, and it imports the
upstream module directly so that no part of the port is in the loop::

    python tests/fixtures/make_geomsurr_reference.py

The stored inputs (``W``, ``D``) are part of the fixture rather than being rebuilt by the test.
That is deliberate: generating them from ``snaplab_tools.datasets`` would mean a change to the
example-data generators could break the reference comparison, which is not what that test is for.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist, squareform

NCTPY_COMMIT = "c01e61bd2662bb50ac9c06468174e9723cadb17a"
GEOMSURR_URL = (
    f"https://raw.githubusercontent.com/LindenParkesLab/nctpy/{NCTPY_COMMIT}"
    f"/src/null_models/geomsurr.py"
)
OUT = Path(__file__).resolve().parent / "geomsurr_reference.npz"

# Small enough to keep the fixture light, large enough for the polynomial fits to be well posed.
N_NODES = 40
SEEDS = (0, 1, 123)


def _load_published_geomsurr():
    """Import the pinned upstream module, without adding nctpy as a dependency."""
    source = urllib.request.urlopen(GEOMSURR_URL).read().decode()
    namespace: dict = {}
    exec(compile(source, GEOMSURR_URL, "exec"), namespace)  # noqa: S102 -- pinned, reviewed source
    return namespace["geomsurr"]


def _make_case(directed, seed=7):
    """A distance-embedded weighted network: lognormal weights decaying with distance, 70% dense."""
    rng = np.random.default_rng(seed)
    distance = squareform(pdist(rng.normal(size=(N_NODES, 3)) * 40.0))
    weights = np.exp(-distance / 30.0 + rng.normal(scale=0.4, size=(N_NODES, N_NODES)))
    if not directed:
        weights = np.tril(weights) + np.tril(weights, -1).T
    weights[rng.random((N_NODES, N_NODES)) < 0.3] = 0.0
    if not directed:
        weights = np.tril(weights) + np.tril(weights, -1).T
    np.fill_diagonal(weights, 0.0)
    return weights, distance


def main():
    geomsurr = _load_published_geomsurr()
    arrays = {}
    for label, directed in (("und", False), ("dir", True)):
        W, D = _make_case(directed)
        arrays[f"W_{label}"] = W
        arrays[f"D_{label}"] = D
        for seed in SEEDS:
            # The published implementation zeroes W's diagonal in place, so pass it a copy.
            Wwp, Wsp, Wssp = geomsurr(W.copy(), D, seed=seed)
            arrays[f"wwp_{label}_{seed}"] = Wwp
            arrays[f"wsp_{label}_{seed}"] = Wsp
            arrays[f"wssp_{label}_{seed}"] = Wssp

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, nctpy_commit=NCTPY_COMMIT, **arrays)
    print(f"wrote {OUT.relative_to(Path.cwd())} ({OUT.stat().st_size / 1024:.0f} KB)")
    print(f"  nctpy commit : {NCTPY_COMMIT}")
    print(f"  cases        : {len(arrays) // 2} arrays over seeds {SEEDS}")


if __name__ == "__main__":
    main()
