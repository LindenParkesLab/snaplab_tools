"""Common analysis tools used by the SNaP Lab.

A grab-bag of the routines that keep recurring across the lab's human-neuroimaging projects,
packaged so they can be shared rather than copy-pasted between repositories. Nothing here is a
pipeline: each module is a set of independent functions you import as needed.

Modules
-------
:mod:`~snaplab_tools.stats`
    Correlation and group statistics -- partial correlations, tests for comparing dependent
    correlations (Steiger, bootstrap, permutation), brain-map coupling and decoupling tests.
:mod:`~snaplab_tools.gams`
    Penalized-spline GAM fitting, derivative signals, change-point detection, bootstrap engine.
:mod:`~snaplab_tools.nulls`
    Spatial null models for parcellated cortical maps (BrainSMASH surrogates) and null p-values.
:mod:`~snaplab_tools.topology`
    Adjacency-matrix thresholding, normalization, and rich-club topology.
:mod:`~snaplab_tools.plotting`
    Publication-figure helpers: correlation plots, cortical surface rendering, colormaps.
:mod:`~snaplab_tools.prediction`
    Cross-validated regression with permutation testing.
:mod:`~snaplab_tools.signal`, :mod:`~snaplab_tools.derivs`
    Time-series filtering, and autocorrelation/functional-connectivity derivatives.
:mod:`~snaplab_tools.brainmaps`
    Loaders for published cortical maps (cytoarchitecture, microstructure, tau, S-A axis).
:mod:`~snaplab_tools.utils`
    Parcellation fetching, parcel-wise averaging, winsorizing, FDR, nuisance regression.

Documentation lives at https://snaplab-tools.readthedocs.io.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
