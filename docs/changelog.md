# Changelog

## 0.1.0

First versioned release, alongside the first published documentation.

### Added

- Sphinx documentation site with an API reference covering every public function, plus runnable
  tutorials executed as part of the build. More tutorials are being added.
- {mod}`snaplab_tools.datasets`: synthetic data generators built on the real Schaefer geometry
  bundled with the package. Brain maps with genuine spatial autocorrelation, subject cohorts,
  structural connectomes, developmental trajectories with a known change point, and BOLD-like time
  series. These make the tutorials runnable anywhere and serve as test fixtures.
- `__all__` on every module, defining the public API explicitly.
- `docs`, `surface`, and `changepoint` optional dependency extras.

### Removed

- Deprecated plotting functions `reg_plot`, `surface_plot` and `annotate_significance_brackets`.
  `plot_correlation` covers what `reg_plot` did; `plot_brain_surface_data` supersedes
  `surface_plot`.
- `snaplab_tools.decomposition` and `snaplab_tools.nulls.utils`, each of which held a single
  function that now lives in {mod}`snaplab_tools.stats`.
- `nuis_reg`, which duplicated {func}`~snaplab_tools.stats.residualize` but fitted no intercept,
  so its residuals were not mean-centred.
- `compute_correlation` and `determine_significance` from `snaplab_tools.plotting.utils`,
  duplicates of {func}`~snaplab_tools.stats.compute_stat` and
  {func}`~snaplab_tools.stats.significance_stars`. Also `get_p_val_string`, leaving
  `format_pvalue` as the single p-value formatter.

### Changed

- Statistics that lived in other modules moved into {mod}`snaplab_tools.stats`: `get_fdr_p`,
  `winsorize`, `winsorize_iqr` (from `utils`), `get_null_p` (from `nulls.utils`), and
  `pca_with_nan_handling` (from `decomposition`). Import paths change accordingly.
- `snaplab_tools.plotting.utils.set_plotting_params` now applies the 8pt font size it documents
  and no longer draws a background grid.
- Documentation moved from a `docs/README.md` stub to a full Sphinx site. The old tutorial notebooks
  in `scripts/` were removed: they depended on a hardcoded local path and private HCP data, so
  nobody but the author could run them. Their replacements live in `docs/tutorials/`, run on
  synthetic data, and are executed when the docs are built.

### Fixed

- `wget` was imported by `snaplab_tools.utils` and `snaplab_tools.brainmaps` but missing from
  `install_requires`, so a clean `pip install` produced a package that could not be imported.
- `snaplab_tools.plotting.plotting.plot_brain_surface_data` **raised when called with its own
  default arguments**: `parcellation='schaefer_400'` was not in the internal lookup table, which
  only held explicit `'schaefer_400-7'`-style keys. The short forms are now accepted as aliases for
  the 7-network order, the docstring lists the values that actually work, and the error message on
  an unrecognised value names the valid ones.
