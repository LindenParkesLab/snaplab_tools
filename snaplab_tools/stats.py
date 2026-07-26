"""Correlation statistics, partial correlations, and tests on brain-map relationships.

Four groups of functions, roughly in order of how specialised they are.

Basic estimation
    :func:`compute_stat` and :func:`partial_pearsonr` (the single partial-correlation estimator
    used throughout), plus :func:`residualize` to remove covariates explicitly and
    :func:`partial_corr_controlled` for the DataFrame-oriented case.

Comparing two dependent correlations
    When you want to know whether X correlates more strongly with Y than with Z, the two
    correlations share a variable and are therefore not independent. :func:`steiger_test` gives
    the analytic answer, :func:`bootstrap_correlation_test` and
    :func:`permutation_correlation_test` the resampling ones. The resampling versions make fewer
    distributional assumptions and are the safer default for skewed brain data.

Many correlations at once
    :func:`correlate_dataframes` correlates every column of one table against every column of
    another and applies FDR correction across the grid.

Subject-level brain-map coupling
    :func:`subject_wise_coupling` reduces each subject's brain map to a single coupling value
    against a reference map; :func:`paired_coupling_test` and :func:`decoupling_test` then test
    whether coupling differs between conditions. :func:`paired_ttest_vs_reference` compares
    several conditions against a common baseline.

Multiple comparisons, outliers, and decomposition
    :func:`get_fdr_p` applies Benjamini-Hochberg correction while preserving the shape of a
    p-value grid. :func:`winsorize` and :func:`winsorize_iqr` clip outliers by percentile or by
    the IQR rule. :func:`pca_with_nan_handling` runs PCA on data with missing entries, which
    plain scikit-learn refuses to do. :func:`get_null_p` turns any observed statistic plus any
    vector of null values into a p-value.

Small helpers: :func:`significance_stars` formats a p-value for a figure annotation.

For spatial null models -- the right way to test a brain-map correlation, since parcels are not
independent observations -- see :mod:`snaplab_tools.nulls` rather than the parametric p-values
returned here. :func:`get_null_p` lives here rather than there because it is generic: it does not
care whether the null came from spatial surrogates, a permutation test, or anything else.
"""
import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

__all__ = [
    'significance_stars',
    'compute_stat',
    'residualize',
    'partial_pearsonr',
    'partial_corr_controlled',
    'correlate_dataframes',
    'steiger_test',
    'bootstrap_correlation_test',
    'permutation_correlation_test',
    'paired_ttest_vs_reference',
    'subject_wise_coupling',
    'paired_coupling_test',
    'decoupling_test',
    'get_null_p',
    'get_fdr_p',
    'winsorize',
    'winsorize_iqr',
    'pca_with_nan_handling',
]


def significance_stars(p_value):
    """Convert a p-value to a stars/ns string ('***' <.001, '**' <.01, '*' <.05, 'ns')."""
    if p_value is None or np.isnan(p_value):
        return 'ns'
    if p_value < 0.001:
        return '***'
    if p_value < 0.01:
        return '**'
    if p_value < 0.05:
        return '*'
    return 'ns'


def steiger_test(r_xy, r_xz, r_yz, n, alternative='two-sided'):
    """
    Steiger's test for comparing two dependent correlations sharing one variable.
    
    Compares r(X,Y) vs r(X,Z) where Y and Z are correlated.
    
    Parameters
    ----------
    r_xy : float
        Correlation between X and Y
    r_xz : float
        Correlation between X and Z
    r_yz : float
        Correlation between Y and Z
    n : int
        Sample size
    alternative : {'two-sided', 'less', 'greater'}, default='two-sided'
        Alternative hypothesis:
        - 'two-sided': r_xy != r_xz
        - 'less': r_xy < r_xz
        - 'greater': r_xy > r_xz
    
    Returns
    -------
    z : float
        Test statistic
    p : float
        P-value
    """
    # Fisher Z-transform
    z_xy = np.arctanh(r_xy)
    z_xz = np.arctanh(r_xz)
    
    # Compute covariance between the two correlations
    r_mean = (r_xy + r_xz) / 2
    cov = (r_yz * (1 - 2 * r_mean**2) - 0.5 * r_mean**2 * (1 - 2 * r_mean**2 - r_yz**2)) / (1 - r_mean**2)**2
    
    # Standard error
    se = np.sqrt(2 * (1 - cov) / (n - 3))
    
    # Test statistic
    z = (z_xy - z_xz) / se
    
    # P-value
    if alternative == 'two-sided':
        p = 2 * (1 - stats.norm.cdf(abs(z)))
    elif alternative == 'less':
        p = stats.norm.cdf(z)
    elif alternative == 'greater':
        p = 1 - stats.norm.cdf(z)
    else:
        raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")
    
    return z, p


def bootstrap_correlation_test(x, y, z, n_bootstrap=10000, method='spearman', 
                               alternative='two-sided', random_state=None):
    """
    Bootstrap test for comparing two dependent correlations sharing one variable.
    
    Compares r(X,Y) vs r(X,Z) using bootstrap resampling.
    
    Parameters
    ----------
    x, y, z : array-like
        Data vectors (must have same length)
    n_bootstrap : int, default=10000
        Number of bootstrap samples
    method : {'pearson', 'spearman'}, default='spearman'
        Correlation method to use
    alternative : {'two-sided', 'less', 'greater'}, default='two-sided'
        Alternative hypothesis:
        - 'two-sided': r_xy != r_xz
        - 'less': r_xy < r_xz
        - 'greater': r_xy > r_xz
    random_state : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    obs_diff : float
        Observed difference (r_xy - r_xz)
    p : float
        P-value
    """
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    
    if method not in ('pearson', 'spearman'):
        raise ValueError(f"method must be 'pearson' or 'spearman'; got {method!r}")
    # Via compute_stat so the n < 3 guard applies here too -- scipy silently returns r=1.0 for
    # two points, which a resample can easily produce.
    corr_func = lambda a, b: compute_stat(a, b, method, return_p=False)[0]
    
    # Observed difference
    r_xy_obs = corr_func(x, y)
    r_xz_obs = corr_func(x, z)
    obs_diff = r_xy_obs - r_xz_obs
    
    # Bootstrap
    rng = np.random.default_rng(random_state)
    n = len(x)
    bootstrap_diffs = np.zeros(n_bootstrap)
    
    for i in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        r_xy_boot = corr_func(x[idx], y[idx])
        r_xz_boot = corr_func(x[idx], z[idx])
        bootstrap_diffs[i] = r_xy_boot - r_xz_boot
    
    p = get_null_p(obs_diff, bootstrap_diffs, alternative=alternative)

    return obs_diff, p


def permutation_correlation_test(x, y, z, n_permutations=10000, method='spearman',
                                 alternative='two-sided', random_state=None):
    """
    Permutation test for comparing two dependent correlations sharing one variable.
    
    Compares r(X,Y) vs r(X,Z) by permuting Y and Z labels.
    
    Parameters
    ----------
    x, y, z : array-like
        Data vectors (must have same length)
    n_permutations : int, default=10000
        Number of permutations
    method : {'pearson', 'spearman'}, default='spearman'
        Correlation method to use
    alternative : {'two-sided', 'less', 'greater'}, default='two-sided'
        Alternative hypothesis:
        - 'two-sided': r_xy != r_xz
        - 'less': r_xy < r_xz
        - 'greater': r_xy > r_xz
    random_state : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    obs_diff : float
        Observed difference (r_xy - r_xz)
    p : float
        P-value
    """
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    
    if method not in ('pearson', 'spearman'):
        raise ValueError(f"method must be 'pearson' or 'spearman'; got {method!r}")
    # Via compute_stat so the n < 3 guard applies here too -- scipy silently returns r=1.0 for
    # two points, which a resample can easily produce.
    corr_func = lambda a, b: compute_stat(a, b, method, return_p=False)[0]
    
    # Observed difference
    r_xy_obs = corr_func(x, y)
    r_xz_obs = corr_func(x, z)
    obs_diff = r_xy_obs - r_xz_obs
    
    # Permutation test
    rng = np.random.default_rng(random_state)
    n = len(x)
    perm_diffs = np.zeros(n_permutations)
    
    for i in range(n_permutations):
        # Randomly swap Y and Z for each observation
        swap = rng.random(n) < 0.5
        y_perm = np.where(swap, z, y)
        z_perm = np.where(swap, y, z)
        
        r_xy_perm = corr_func(x, y_perm)
        r_xz_perm = corr_func(x, z_perm)
        perm_diffs[i] = r_xy_perm - r_xz_perm
    
    # P-value
    p = get_null_p(obs_diff, perm_diffs, alternative=alternative)

    return obs_diff, p


def _pearson_r(x, y):
    """Pearson correlation coefficient only, without scipy's p-value machinery.

    Roughly 8x faster than ``scipy.stats.pearsonr`` because it skips the input validation and
    the beta-distribution p-value, which is most of that function's cost. Agrees with scipy to
    machine precision. Returns NaN when either input is constant, matching what scipy does
    (scipy also emits a ConstantInputWarning; this does not).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xm = x - x.mean()
    ym = y - y.mean()

    denom = np.sqrt((xm * xm).sum() * (ym * ym).sum())
    if denom == 0:
        return np.nan
    # Clamp: floating-point error can push a perfect correlation a hair outside [-1, 1].
    return float(np.clip((xm * ym).sum() / denom, -1.0, 1.0))


def compute_stat(x, y, method='pearson', return_p=True):
    """Compute a correlation or R^2 statistic, optionally with its parametric p-value.

    Parameters
    ----------
    x, y : ndarray
        1-D arrays (NaN-free).
    method : {'pearson', 'spearman', 'r2'}
        Statistic to compute.
    return_p : bool
        Whether to compute the p-value. Set False in resampling loops and anywhere else the
        p-value is discarded: computing it is most of the cost, so skipping it is about 8x
        faster for 'pearson' and 2x for 'spearman' (where ranking dominates and cannot be
        avoided). The statistic itself is identical either way.

    Returns
    -------
    stat : float
        r, rho, or R^2. NaN when fewer than three observations are supplied.
    p : float
        Parametric p-value, or NaN when `return_p` is False. Always returned, so the two-value
        unpacking works regardless.

    Notes
    -----
    Fewer than three points cannot support a correlation, but scipy does not say so: two points
    always lie on a line, so ``scipy.stats.pearsonr`` returns ``r=1.0, p=1.0`` for ``n=2`` with no
    warning at all. Returning NaN makes that visible rather than letting a perfect correlation
    from two observations flow downstream.

    Examples
    --------
    >>> r, p = compute_stat(x, y, 'pearson')                    # with p-value
    >>> r, _ = compute_stat(x, y, 'pearson', return_p=False)    # coefficient only, ~8x faster
    """
    if len(x) < 3 or len(y) < 3:
        return np.nan, np.nan

    if method == 'pearson':
        if not return_p:
            return _pearson_r(x, y), np.nan
        return stats.pearsonr(x, y)
    elif method == 'spearman':
        if not return_p:
            # Spearman is Pearson on ranks. scipy.stats.rankdata handles ties correctly and is
            # the bulk of the remaining cost, so the saving here is smaller than for Pearson.
            return _pearson_r(stats.rankdata(x), stats.rankdata(y)), np.nan
        r, p = stats.spearmanr(x, y)
        return r, p
    elif method == 'r2':
        X = x.reshape(-1, 1)
        model = LinearRegression().fit(X, y)
        r2 = r2_score(y, model.predict(X))
        n = len(x)
        if not return_p:
            return r2, np.nan
        if r2 >= 1.0:
            return r2, 0.0
        elif r2 <= 0:
            return r2, 1.0
        f_stat = (r2 * (n - 2)) / (1 - r2)
        p = 1 - stats.f.cdf(f_stat, 1, n - 2)
        return r2, p
    else:
        raise ValueError(f"method must be 'pearson', 'spearman', or 'r2', got '{method}'")


def correlate_dataframes(df_neuro, df_ints, method='pearson', alpha=0.05,
                         null_distributions=None):
    """Correlate every column of df_neuro with every column of df_ints.

    Parameters
    ----------
    df_neuro : pd.DataFrame
        Regions x features.
    df_ints : pd.DataFrame
        Regions x conditions.
    method : {'pearson', 'spearman', 'r2'}
        Statistic to compute per pair.
    alpha : float
        Significance threshold (carried for the caller; not applied here).
    null_distributions : dict or None
        {int_col: (n_perms, n_regions)}. If provided, one-tailed permutation
        p-values are used instead of parametric ones; must cover all df_ints columns.

    Returns
    -------
    df_results : pd.DataFrame
        Statistics (neuro features x INT conditions).
    df_pvals : pd.DataFrame
        P-values, same shape.
    """
    if len(df_neuro) != len(df_ints):
        raise ValueError(
            f"DataFrames must have the same number of rows. "
            f"Got {len(df_neuro)} and {len(df_ints)}"
        )
    if null_distributions is not None:
        missing = set(df_ints.columns) - set(null_distributions)
        if missing:
            raise ValueError(f"null_distributions missing columns: {missing}")

    use_perm = null_distributions is not None
    results, pvals = [], []

    for neuro_col in df_neuro.columns:
        row_r, row_p = [], []
        for int_col in df_ints.columns:
            mask = ~(df_neuro[neuro_col].isna() | df_ints[int_col].isna())
            x = df_neuro.loc[mask, neuro_col].values.astype(float)
            y = df_ints.loc[mask, int_col].values.astype(float)

            if len(x) < 3:
                row_r.append(np.nan)
                row_p.append(np.nan)
                continue

            stat_obs, p_param = compute_stat(x, y, method)
            row_r.append(stat_obs)

            if use_perm:
                null = null_distributions[int_col]           # (n_perms, n_regions)
                null_masked = null[:, mask]
                perm_stats = np.array([
                    compute_stat(x, null_masked[i], method, return_p=False)[0]
                    for i in range(null.shape[0])
                ])
                # R^2 is a magnitude already, so a better fit always means a larger value and the
                # upper tail is the only meaningful one. For a signed correlation the tail
                # follows the direction of the observed effect.
                tail = 'greater' if method == 'r2' else 'auto'
                row_p.append(get_null_p(stat_obs, perm_stats, alternative=tail))
            else:
                row_p.append(p_param)

        results.append(row_r)
        pvals.append(row_p)

    df_results = pd.DataFrame(results, index=df_neuro.columns, columns=df_ints.columns)
    df_pvals   = pd.DataFrame(pvals,   index=df_neuro.columns, columns=df_ints.columns)
    return df_results, df_pvals


def residualize(y, covariates):
    """OLS-residualize ``y`` on ``covariates`` (intercept always included). ``covariates`` is a
    (n,) or (n, k) array, or None for intercept-only (mean-centering).

    Re-exported by :mod:`snaplab_tools.nulls` for backwards compatibility; this is the single
    definition.
    """
    y = np.asarray(y, dtype=float)
    if covariates is None:
        Z = np.ones((len(y), 1))
    else:
        C = np.asarray(covariates, float)
        if C.ndim == 1:
            C = C[:, None]
        Z = np.column_stack([np.ones(len(y)), C])
    return y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]


def partial_pearsonr(x, y, covariates=None):
    """Pearson correlation between ``x`` and ``y``, optionally controlling for covariates.

    Residualizes both ``x`` and ``y`` on the covariates (OLS, intercept included) and correlates
    the residuals. With no covariates this is an ordinary Pearson correlation.

    This is the array-level primitive: plain numpy, no DataFrame or statsmodels fit per call, so
    it is ~15x faster and safe to call inside per-region / per-voxel loops. See
    :func:`partial_corr_controlled` for a DataFrame interface over the same estimator (the two
    agree to machine precision).

    Parameters
    ----------
    x, y : array-like
        Matching-length 1-D vectors.
    covariates : array-like or None
        (n,) or (n, k) covariates partialled out of both ``x`` and ``y``.

    Returns
    -------
    tuple
        ``(r, p, n)`` over the entries where every input is non-NaN. Degrees of freedom are
        ``n - 2`` with no covariates and ``n - 2 - k`` with ``k`` of them. Returns
        ``(nan, nan, n)`` when the dof would be < 1, or when the covariates fully explain
        either variable (leaving a residual with no variance).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError(f"x and y must be 1-D; got {x.shape} and {y.shape}.")
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same length; got {x.shape} and {y.shape}.")

    valid = ~(np.isnan(x) | np.isnan(y))

    C = None
    if covariates is not None:
        C = np.asarray(covariates, dtype=float)
        if C.ndim == 1:
            C = C[:, None]
        if C.ndim != 2 or C.shape[0] != x.shape[0]:
            raise ValueError(
                f"covariates must be (n,) or (n, k) matching the length of x and y "
                f"({x.shape[0]}); got {np.asarray(covariates).shape}."
            )
        valid = valid & ~np.isnan(C).any(axis=1)

    n   = int(valid.sum())
    k   = 0 if C is None else C.shape[1]
    dof = n - 2 - k
    if dof < 1:
        return np.nan, np.nan, n

    x_v, y_v = x[valid], y[valid]

    if C is None:
        r, p = sp.stats.pearsonr(x_v, y_v)
        return float(r), float(p), n

    x_res = residualize(x_v, C[valid])
    y_res = residualize(y_v, C[valid])

    # A residual with no variance means the covariates fully explain that variable, so the
    # partial correlation is undefined. Compare against each variable's own scale rather than
    # against 0: a perfectly explained residual lands on floating-point dust (~1e-16), not
    # exactly zero, and corrcoef would happily report that noise as signal.
    if x_res.std() <= 1e-8 * x_v.std() or y_res.std() <= 1e-8 * y_v.std():
        return np.nan, np.nan, n

    r = float(np.clip(np.corrcoef(x_res, y_res)[0, 1], -1.0, 1.0))
    t = r * np.sqrt(dof / max(1e-12, 1.0 - r ** 2))
    p = float(2 * sp.stats.t.sf(abs(t), dof))
    return r, p, n


def partial_corr_controlled(df, predictor, outcome, covars):
    """Covariate-controlled partial correlation between a predictor and an outcome.

    DataFrame interface over :func:`partial_pearsonr`. Rows missing any of
    ``predictor``/``outcome``/``covars`` are dropped, then both the predictor and the outcome
    are residualized on the covariates (OLS, intercept included) and the Pearson correlation of
    the residuals is returned. A one-tailed p-value for a pre-specified positive direction is
    also returned.

    Parameters
    ----------
    df : pandas.DataFrame
        Data containing ``predictor``, ``outcome``, and every name in ``covars``.
    predictor : str
        Column correlated with ``outcome`` after controlling for ``covars``.
    outcome : str
        Outcome column.
    covars : list of str
        Covariate columns partialled out of both predictor and outcome.

    Returns
    -------
    dict
        With keys ``r`` (residual Pearson r), ``p_two`` (two-tailed p, equivalent to the
        predictor's coefficient p in ``outcome ~ predictor + covars``), ``p_one_pos``
        (one-tailed p for the positive hypothesis), ``n`` (rows used), and
        ``resid_x``/``resid_y`` (the covariate residuals of predictor and outcome).
    """
    d = df[[predictor, outcome] + covars].dropna()
    x = d[predictor].to_numpy(dtype=float)
    y = d[outcome].to_numpy(dtype=float)
    C = d[covars].to_numpy(dtype=float)

    r, p_two, n = partial_pearsonr(x, y, C)
    p_one_pos = p_two / 2 if r > 0 else 1 - p_two / 2   # one-tailed, hypothesis = positive
    return dict(r=r, p_two=p_two, p_one_pos=p_one_pos, n=n,
                resid_x=residualize(x, C), resid_y=residualize(y, C))


def paired_ttest_vs_reference(df, reference=None, columns=None,
                              alternative='two-sided', correction='fdr_bh'):
    """Paired-samples t-tests of each condition column against a reference column.

    Each row is one paired observation (e.g. a brain region); each column is a
    condition measured on those same observations. For every non-reference column
    a paired t-test (``scipy.stats.ttest_rel``) against the reference is computed
    on the rows where both values are non-NaN, and the p-values are optionally
    corrected across the set of comparisons.

    Parameters
    ----------
    df : pandas.DataFrame
        Observations (rows) x conditions (columns). Non-numeric columns that are
        neither the reference nor requested in ``columns`` are ignored.
    reference : hashable or None
        Column every other column is compared against. Defaults to the first column.
    columns : list or None
        Conditions to test against the reference, in the desired output order.
        Defaults to all columns except the reference, in dataframe order.
    alternative : {'two-sided', 'less', 'greater'}, default='two-sided'
        Passed to ``scipy.stats.ttest_rel``; the direction refers to
        ``column - reference`` (so 'less' tests column < reference).
    correction : {'holm', 'bonferroni', 'fdr_bh', ...} or None, default='fdr_bh'
        Multiple-comparison correction applied across the tested columns via
        ``statsmodels.stats.multitest.multipletests``. None leaves p uncorrected.

    Returns
    -------
    pandas.DataFrame
        One row per tested column (indexed by column name), with columns:
        ``n`` (paired observations used), ``dof``, ``mean_diff``
        (mean of column - reference), ``t``, ``p`` (uncorrected),
        ``p_corr`` (corrected, equal to ``p`` when correction is None), and
        ``sig`` (stars/ns string based on ``p_corr``).
    """
    if reference is None:
        reference = df.columns[0]
    if reference not in df.columns:
        raise ValueError(f"reference column {reference!r} not in dataframe")
    if columns is None:
        columns = [c for c in df.columns if c != reference]
    if reference in columns:
        raise ValueError("reference column must not appear in `columns`")

    ref = df[reference]
    records = []
    for col in columns:
        mask = ~(ref.isna() | df[col].isna())
        a = df.loc[mask, col].to_numpy(dtype=float)
        b = ref.loc[mask].to_numpy(dtype=float)
        n = int(mask.sum())
        if n < 2:
            records.append(dict(condition=col, n=n, dof=np.nan,
                                mean_diff=np.nan, t=np.nan, p=np.nan))
            continue
        t, p = stats.ttest_rel(a, b, alternative=alternative)
        records.append(dict(condition=col, n=n, dof=n - 1,
                            mean_diff=float(np.mean(a - b)), t=float(t), p=float(p)))

    out = pd.DataFrame(records).set_index('condition')

    # Correct across the valid (non-NaN) comparisons only.
    out['p_corr'] = out['p']
    valid = out['p'].notna()
    if correction is not None and valid.any():
        out.loc[valid, 'p_corr'] = multipletests(
            out.loc[valid, 'p'].to_numpy(), method=correction,
        )[1]

    out['sig'] = out['p_corr'].apply(significance_stars)
    return out


def subject_wise_coupling(brain_maps, reference_map, method='spearman'):
    """Per-subject correlation between each subject's brain map and a reference map.

    Parameters
    ----------
    brain_maps : ndarray, shape (n_subjects, n_regions)
        Per-subject, per-region brain maps.
    reference_map : ndarray, shape (n_regions,)
        Region-wise reference map correlated against within each subject.
    method : {'spearman', 'pearson'}, default='spearman'
        Correlation used.

    Returns
    -------
    ndarray, shape (n_subjects,)
        Per-subject correlation. Regions with a NaN in the map or the reference are
        excluded per subject; a subject with fewer than 3 usable regions yields NaN.
    """
    if method not in ('pearson', 'spearman'):
        raise ValueError(f"method must be 'pearson' or 'spearman'; got {method!r}")

    brain_maps    = np.asarray(brain_maps,    dtype=float)
    reference_map = np.asarray(reference_map, dtype=float)
    if brain_maps.ndim != 2:
        raise ValueError("brain_maps must be 2-D (subjects x regions)")
    if brain_maps.shape[1] != reference_map.shape[0]:
        raise ValueError("region axis of brain_maps must match len(reference_map)")

    rho = np.empty(brain_maps.shape[0])
    for i in range(brain_maps.shape[0]):
        m = ~(np.isnan(brain_maps[i]) | np.isnan(reference_map))
        # compute_stat returns NaN below three usable regions, so no explicit count guard here.
        rho[i] = compute_stat(brain_maps[i][m], reference_map[m], method, return_p=False)[0]
    return rho


def paired_coupling_test(coupling_a, coupling_b, alternative='greater'):
    """Paired Fisher-z t-test comparing two sets of per-subject couplings.

    Given per-subject correlations (e.g. from ``subject_wise_coupling``) measured in
    two conditions on the same subjects, Fisher z-transform them and run a paired
    t-test. Subjects whose coupling is undefined (NaN, or +/-1 -> +/-inf z) in either
    condition are dropped. Use this when the couplings are already computed and you
    do not want to recompute them from the maps (see ``decoupling_test``).

    Parameters
    ----------
    coupling_a, coupling_b : ndarray, shape (n_subjects,)
        Per-subject couplings in the two conditions (same subjects, same order).
    alternative : {'greater', 'less', 'two-sided'}, default='greater'
        Passed to ``scipy.stats.ttest_rel`` on (z_a - z_b). 'greater' tests
        condition-A coupling > condition-B coupling (decoupling in B).

    Returns
    -------
    rho_a, rho_b : float
        Mean coupling in each condition over the paired subjects used.
    t, p : float
        Paired-samples t statistic and p-value on the Fisher-z couplings.
    n : int
        Number of paired subjects used (after dropping undefined couplings).
    """
    coupling_a = np.asarray(coupling_a, dtype=float)
    coupling_b = np.asarray(coupling_b, dtype=float)
    if coupling_a.shape != coupling_b.shape:
        raise ValueError("coupling_a and coupling_b must have the same shape")

    z_a, z_b = np.arctanh(coupling_a), np.arctanh(coupling_b)   # Fisher z
    good = np.isfinite(z_a) & np.isfinite(z_b)
    res = sp.stats.ttest_rel(z_a[good], z_b[good], alternative=alternative)
    return (float(coupling_a[good].mean()), float(coupling_b[good].mean()),
            float(res.statistic), float(res.pvalue), int(good.sum()))


def decoupling_test(brain_maps_a, brain_maps_b, reference_map,
                    alternative='greater', method='spearman'):
    """Paired test of whether per-subject brain maps' coupling to a reference weakens.

    For each subject, the correlation between their per-region brain map and a fixed
    region-wise reference map is computed in each of two conditions (``brain_maps_a``
    and ``brain_maps_b``). The per-subject correlations are Fisher z-transformed and
    compared with a paired t-test, testing whether the coupling to the reference
    differs between the two conditions. A positive t means the coupling is stronger
    in condition A (i.e. it decouples in B).

    (Motivating use case: brain_maps_* are per-subject intrinsic-timescale maps at
    rest vs during a task, and reference_map is the sensorimotor-association axis.)

    Subjects whose map is entirely NaN in either condition are dropped, and within
    each subject regions with a NaN in the map or the reference are excluded from
    that subject's correlation.

    Parameters
    ----------
    brain_maps_a, brain_maps_b : ndarray, shape (n_subjects, n_regions)
        Per-subject, per-region brain maps in the two conditions. Rows must be the
        same subjects in the same order (a paired design).
    reference_map : ndarray, shape (n_regions,)
        Region-wise reference map correlated against within each subject.
    alternative : {'greater', 'less', 'two-sided'}, default='greater'
        Passed to ``scipy.stats.ttest_rel`` on (z_a - z_b). The default 'greater'
        is the one-sided decoupling test: condition-A coupling > condition-B coupling
        (i.e. coupling drops from A to B). Use 'two-sided' for a directionless test.
    method : {'spearman', 'pearson'}, default='spearman'
        Correlation between each subject's map and the reference.

    Returns
    -------
    rho_a : float
        Mean per-subject coupling in condition A (over the paired subjects used).
    rho_b : float
        Mean per-subject coupling in condition B (same subjects).
    t : float
        Paired-samples t statistic on the Fisher-z coupling (A vs B).
    p : float
        Corresponding p-value.
    n : int
        Number of paired subjects contributing to the test (after dropping any with
        an undefined coupling in either condition).
    """
    brain_maps_a = np.asarray(brain_maps_a, dtype=float)
    brain_maps_b = np.asarray(brain_maps_b, dtype=float)
    if brain_maps_a.shape != brain_maps_b.shape:
        raise ValueError("brain_maps_a and brain_maps_b must have the same shape (subjects x regions)")

    rho_a = subject_wise_coupling(brain_maps_a, reference_map, method=method)
    rho_b = subject_wise_coupling(brain_maps_b, reference_map, method=method)
    # Subjects with an undefined coupling in either condition are dropped by the test.
    return paired_coupling_test(rho_a, rho_b, alternative=alternative)

# =============================================================================
# Multiple comparisons, outliers, and decomposition
#
# Moved here from snaplab_tools.utils, snaplab_tools.nulls.utils and
# snaplab_tools.decomposition -- they are general-purpose statistics and belong with the
# rest of them rather than scattered across modules named for other things.
# =============================================================================


def get_fdr_p(p_vals, alpha=0.05):
    """Benjamini-Hochberg FDR correction that preserves the input shape.

    A thin wrapper over ``statsmodels.stats.multitest.multipletests`` that flattens a 2D array
    of p-values, corrects across all of them jointly, and reshapes the result -- convenient for
    correcting a full correlation matrix or a region-by-variable grid in one call.

    Parameters
    ----------
    p_vals : (n,) or (n, m) ndarray
        Uncorrected p-values.
    alpha : float
        Family-wise target FDR. Only affects the rejection decision, which is discarded here;
        the returned q-values are unaffected.

    Returns
    -------
    ndarray
        FDR-corrected p-values (q-values), same shape as `p_vals`.
    """
    if p_vals.ndim == 2:
        do_reshape = True
        dims = p_vals.shape
        p_vals = p_vals.flatten()
    else:
        do_reshape = False

    out = multipletests(p_vals, alpha=alpha, method='fdr_bh')
    p_fdr = out[1]

    if do_reshape:
        p_fdr = p_fdr.reshape(dims)

    return p_fdr


def _clip_preserving_type(data, lower_bound, upper_bound):
    """Clip to [lower_bound, upper_bound], returning the same container type as `data`.

    NaNs pass through untouched: np.clip leaves them as NaN rather than pulling them to a bound.
    """
    if isinstance(data, pd.Series):
        clipped = np.clip(data.values.astype(float), lower_bound, upper_bound)
        return pd.Series(clipped, index=data.index, name=data.name)
    return np.clip(np.asarray(data, dtype=float), lower_bound, upper_bound)


def winsorize(data, lower_percentile=1, upper_percentile=99):
    """Winsorize by clipping values at the given percentiles.

    Parameters
    ----------
    data : array-like or pandas.Series
        Data to winsorize. A Series comes back as a Series with its index and name intact.
    lower_percentile, upper_percentile : float
        Percentile thresholds to clip at. NaNs are ignored when computing them and are left as
        NaN in the output.

    Returns
    -------
    ndarray or pandas.Series
        A winsorized copy; the input is never modified.

    See Also
    --------
    winsorize_iqr : the same idea using Tukey's IQR rule instead of fixed percentiles.
    """
    values = data.values if isinstance(data, pd.Series) else np.asarray(data, dtype=float)
    lower_bound = np.nanpercentile(values, lower_percentile)
    upper_bound = np.nanpercentile(values, upper_percentile)
    return _clip_preserving_type(data, lower_bound, upper_bound)


def winsorize_iqr(data, k=1.5):
    """Winsorize by clipping at Tukey's IQR fences, ``Q1 - k*IQR`` and ``Q3 + k*IQR``.

    Unlike :func:`winsorize`, the thresholds adapt to the spread of the data rather than sitting
    at fixed percentiles, so a tight distribution gets tight fences.

    Parameters
    ----------
    data : array-like or pandas.Series
        Data to winsorize. A Series comes back as a Series with its index and name intact.
    k : float
        IQR multiplier. 1.5 is Tukey's conventional "outlier" fence; 3.0 marks "far out" points.

    Returns
    -------
    ndarray or pandas.Series
        A winsorized copy; the input is never modified.

    Notes
    -----
    Quartiles are computed with ``np.nanpercentile``. Using plain ``np.percentile`` here meant a
    single NaN made both fences NaN, and since every comparison against NaN is False, the
    function silently returned the data completely unwinsorized.
    """
    values = data.values if isinstance(data, pd.Series) else np.asarray(data, dtype=float)
    q1 = np.nanpercentile(values, 25)
    q3 = np.nanpercentile(values, 75)
    iqr = q3 - q1
    return _clip_preserving_type(data, q1 - k * iqr, q3 + k * iqr)


def get_null_p(observed, null, alternative='two-sided'):
    """Proportion of a null distribution at least as extreme as the observed statistic.

    Generic: it does not care whether the null came from spatial surrogates, a bootstrap, a
    permutation test, or anything else.

    Parameters
    ----------
    observed : float
        The observed test statistic.
    null : (n_perms,) array-like
        Null distribution of the same statistic.
    alternative : {'two-sided', 'greater', 'less', 'auto'}
        Which tail to test.

        - 'two-sided' (default): proportion of the null whose *magnitude* is at least the
          observed magnitude. The right choice when sign is not the hypothesis, as for a
          correlation.
        - 'greater': proportion of the null at or above `observed`.
        - 'less': proportion of the null at or below `observed`.
        - 'auto': one-tailed in whichever direction the observed effect points -- upper tail if
          `observed` is non-negative, lower tail otherwise. Convenient, but note the direction is
          chosen from the data, so it is not a pre-registered one-tailed test and will report
          smaller p-values than 'two-sided'.

    Returns
    -------
    float
        The p-value, in [0, 1]. Note the minimum attainable value is ``1 / n`` over the finite
        null entries -- a p of 0 means "smaller than this null can resolve", so report it as
        ``p < 1/n_perms`` rather than as zero. Returns NaN if `observed` is not finite, or if the
        null has no finite entries.

    Notes
    -----
    Non-finite entries in `null` are dropped and the p-value is computed over what remains.
    Surrogates whose statistic could not be computed carry no information about the null, so
    counting them in the denominator would make the p-value anti-conservative.

    Examples
    --------
    >>> p = get_null_p(0.42, null_distribution)                        # two-sided
    >>> p = get_null_p(0.42, null_distribution, alternative='greater')  # directional
    """
    valid = ('two-sided', 'greater', 'less', 'auto')
    if alternative not in valid:
        raise ValueError(f"alternative must be one of {valid}; got {alternative!r}")

    observed = float(observed)
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]

    # A NaN observed statistic (a constant input, an all-NaN region) should propagate as a NaN
    # p-value rather than raise, so callers looping over regions get a result vector with holes
    # instead of an exception part-way through.
    if not np.isfinite(observed) or null.size == 0:
        return np.nan

    if alternative == 'two-sided':
        return float(np.mean(np.abs(null) >= np.abs(observed)))

    if alternative == 'auto':
        alternative = 'greater' if observed >= 0 else 'less'

    if alternative == 'greater':
        return float(np.mean(null >= observed))
    return float(np.mean(null <= observed))


def pca_with_nan_handling(data, n_components=None, standardize=False, 
                          return_full_shape=False, impute=False):
    """
    Perform PCA with NaN handling.
    
    Parameters
    ----------
    data : ndarray, shape (n_samples, n_features)
        Input data where samples are in rows and features in columns.
        For your case: (n_regions, n_measures)
    n_components : int, float, or None
        Number of components to keep. If None, keeps all components.
        If float between 0 and 1, selects number of components such that
        the explained variance is greater than the percentage specified.
    standardize : bool, default=False
        If True, z-score the data (per feature) before PCA.
    return_full_shape : bool, default=False
        If True, returns scores in original shape with NaNs reinserted.
        If False, returns scores only for valid samples.
        Only relevant when impute=False.
    impute : bool, default=False
        If True, impute NaN values using median of each column.
        If False, remove rows containing any NaNs.
    
    Returns
    -------
    results : dict
        Dictionary containing:

        - 'scores': PC scores, shape (n_valid_samples, n_components) or
          (n_samples, n_components) if return_full_shape=True or impute=True
        - 'loadings': PC loadings (weights), shape (n_features, n_components)
        - 'explained_variance': Variance explained by each component
        - 'explained_variance_ratio': Proportion of variance explained
        - 'cumulative_variance_ratio': Cumulative proportion of variance
        - 'pca_object': Fitted PCA object
        - 'valid_mask': Boolean mask of valid (non-NaN) samples (None if impute=True)
        - 'mean': Mean of each feature (after NaN handling, before PCA)
        - 'std': Std of each feature if standardized, else None
        - 'imputer': SimpleImputer object if impute=True, else None
    """
    if impute:
        # Impute NaNs using median
        imputer = SimpleImputer(strategy='median')
        data_clean = imputer.fit_transform(data)
        valid_mask = None
    else:
        # Identify rows without NaNs
        valid_mask = ~np.isnan(data).any(axis=1)
        n_valid = valid_mask.sum()
        
        if n_valid == 0:
            raise ValueError("All samples contain NaNs")
        
        # Extract valid data
        data_clean = data[valid_mask]
        imputer = None
    
    # Store mean for centering information
    mean = np.mean(data_clean, axis=0)
    std = None
    
    # Optional standardization
    if standardize:
        data_clean = stats.zscore(data_clean, axis=0, ddof=1)
        std = np.std(data_clean, axis=0, ddof=1)
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(data_clean)
    
    # Get loadings (components_ is transposed)
    loadings = pca.components_.T
    
    # Optionally return full shape with NaNs (only when not imputing)
    if not impute and return_full_shape:
        scores_full = np.full((data.shape[0], scores.shape[1]), np.nan)
        scores_full[valid_mask] = scores
        scores = scores_full
    
    # Calculate cumulative variance
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    
    results = {
        'scores': scores,
        'loadings': loadings,
        'explained_variance': pca.explained_variance_,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cumulative_variance_ratio': cumulative_variance_ratio,
        'pca_object': pca,
        'valid_mask': valid_mask,
        'mean': mean,
        'std': std,
        'imputer': imputer
    }
    
    return results
