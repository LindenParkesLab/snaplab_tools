"""Null models for brain maps and brain networks.

Two families, for two different kinds of claim.

:mod:`~snaplab_tools.nulls.maps` surrogates a parcel-wise **map**: it generates random maps that
preserve the spatial autocorrelation of the original, which is what makes a correlation between
two brain maps testable. Use it when the statistic is a property of a map -- a correlation with
another map, a system-wise mean.

:mod:`~snaplab_tools.nulls.networks` surrogates the **network**: it redistributes edge weights
while preserving their relationship to inter-nodal distance, and optionally node strengths. Use it
when the statistic is a property of a weighted graph -- control energy, average controllability,
weighted modularity. It leaves the edge set untouched, so it is *not* a null for binary graph
statistics; see that module's warning.

The generic statistics both build on -- ``get_null_p`` for turning any null distribution into a
p-value, ``residualize`` for removing covariates -- live in :mod:`snaplab_tools.stats`, since
neither is specific to null models::

    from snaplab_tools.nulls import generate_surrogates, generate_network_nulls
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
from .networks import (
    geomsurr,
    generate_network_nulls,
)

__all__ = [
    "load_distance_matrix",
    "build_geodesic_distance_matrix",
    "generate_surrogates",
    "corr_with_null",
    "corr_with_covar_null",
    "correlate_family",
    "network_enrichment",
    "geomsurr",
    "generate_network_nulls",
]
