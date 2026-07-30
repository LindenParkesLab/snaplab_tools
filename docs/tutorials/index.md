# Tutorials

Worked examples for each major component of the package.

Every tutorial runs on synthetic data from {mod}`snaplab_tools.datasets`, so you can execute any of
them without access to a particular dataset. They are also executed when this documentation is
built, which means the figures and numbers on these pages were produced by the code shown above
them — if the API changes and a tutorial breaks, the build fails.

Synthetic does not mean arbitrary. The brain maps carry real spatial autocorrelation over the real
Schaefer parcellation geometry bundled with the package, which is what makes a spatial null model
meaningful rather than merely executable.

```{toctree}
:maxdepth: 1

plotting_correlations
colormaps_and_style
null_models
null_networks
intrinsic_neural_timescales
```

:::{note}
More tutorials — GAMs and change-point detection, correlation statistics, network topology,
cross-validated prediction, cortical surfaces — are being rewritten and will appear here as they
land. The [API reference](../api/index.md) documents every function in the meantime.
:::
