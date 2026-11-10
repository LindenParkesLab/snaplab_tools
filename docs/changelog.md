# Changelog

## Unreleased

### Added

- Detrended fluctuation analysis in {mod}`~snaplab_tools.timescales`:
  {func}`~snaplab_tools.timescales.dfa_fluctuation` (the fluctuation function F(n), with overlapping
  segments, polynomial detrending of any order, an optional mask of usable samples whose segments are
  skipped, and segment counts for pooling several recordings),
  {func}`~snaplab_tools.timescales.dfa_exponent` (log-log slope over a range of scales, with r²) and
  {func}`~snaplab_tools.timescales.dfa`. Tested against white noise (α = 0.5), 1/f noise (α = 1.0),
  both also with 10 % of samples deleted, and the AR(1) crossover from α = 1.5 to 0.5, whose scale
  for DFA-1 is √28 / (1 − φ).
- {func}`~snaplab_tools.timescales.crossing_interp`: the first crossing of any threshold by an ACF at
  sub-lag resolution (``threshold=0.5`` gives an interpolated ACW-50).
  {func}`~snaplab_tools.timescales.zero_crossing_interp` now calls it with ``threshold=0``; its
  results are unchanged.
- {func}`~snaplab_tools.stats.icc`: intraclass correlation, ICC(1,1), ICC(2,1) or ICC(3,1) (Shrout &
  Fleiss 1979), per feature for a 3-D input, with listwise deletion of missing values.

## 0.1.1

### Added

- Amplitude-invariant timescale estimators in {mod}`~snaplab_tools.timescales`, all in
  `TIMESCALE_METHODS`: {func}`~snaplab_tools.timescales.acf_ratio_21` (ACF(2)/ACF(1)),
  {func}`~snaplab_tools.timescales.tau_ratio_21` (its implied decay time; aggregate the ratio, then
  convert), and {func}`~snaplab_tools.timescales.tau_expfit_free` (a free-amplitude exponential over
  a fixed window of lags 1..4, bounded fits censored at the bound). Additive white noise scales every
  lag >= 1 of an ACF, which these cancel. Also {func}`~snaplab_tools.timescales.zero_crossing_interp`,
  the first zero crossing at sub-TR resolution -- which, like
  {func}`~snaplab_tools.timescales.zero_crossing`, is *not* amplitude-invariant.
- {func}`~snaplab_tools.derivs.compute_brain_states`: recurrent patterns of activity ("brain states")
  by k-means clustering of time points, as in Kim et al. (Nat Commun 2025). Regions are z-scored across
  time by default, and regions that cannot be clustered (non-finite or constant) are left out and
  returned as NaN rather than raising. `n_init` is set explicitly (10), because scikit-learn's own
  default changed in 1.4.
- {func}`~snaplab_tools.plotting.utils.save_panel`: one place for a project's figure-export
  convention -- tight bbox, PNG plus editable SVG, and a content-aware DPI default (higher for
  panels containing an `imshow`, where pixels are the content, than for line panels whose PNG is
  a preview of the vector file). Supports transparent backgrounds for panels composited later.

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
