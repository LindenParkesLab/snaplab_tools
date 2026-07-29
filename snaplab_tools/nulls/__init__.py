"""Spatial null models for parcellated cortical maps.

Generating surrogate maps that preserve spatial autocorrelation, and testing observed correlations
against them.

The generic statistics these build on -- ``get_null_p`` for turning any null distribution into a
p-value, ``residualize`` for removing covariates -- live in :mod:`snaplab_tools.stats`, since
neither is specific to null models::

    from snaplab_tools.nulls import generate_surrogates
    from snaplab_tools.stats import get_null_p
"""

from .maps import (
    load_distance_matrix,
    build_geodesic_distance_matrix,
    generate_surrogates,
    corr_with_null,
    corr_with_covar_null,
    correlate_family,
    network_enrichment,
)

__all__ = [
    "load_distance_matrix",
    "build_geodesic_distance_matrix",
    "generate_surrogates",
    "corr_with_null",
    "corr_with_covar_null",
    "correlate_family",
    "network_enrichment",
]
