import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


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
    
    if method == 'pearson':
        corr_func = lambda a, b: sp.stats.pearsonr(a, b)[0]
    elif method == 'spearman':
        corr_func = lambda a, b: sp.stats.spearmanr(a, b)[0]
    else:
        raise ValueError("method must be 'pearson' or 'spearman'")
    
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
    
    # P-value
    if alternative == 'two-sided':
        p = np.mean(np.abs(bootstrap_diffs) >= np.abs(obs_diff))
    elif alternative == 'less':
        p = np.mean(bootstrap_diffs <= obs_diff)
    elif alternative == 'greater':
        p = np.mean(bootstrap_diffs >= obs_diff)
    else:
        raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")
    
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
    
    if method == 'pearson':
        corr_func = lambda a, b: sp.stats.pearsonr(a, b)[0]
    elif method == 'spearman':
        corr_func = lambda a, b: sp.stats.spearmanr(a, b)[0]
    else:
        raise ValueError("method must be 'pearson' or 'spearman'")
    
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
    if alternative == 'two-sided':
        p = np.mean(np.abs(perm_diffs) >= np.abs(obs_diff))
    elif alternative == 'less':
        p = np.mean(perm_diffs <= obs_diff)
    elif alternative == 'greater':
        p = np.mean(perm_diffs >= obs_diff)
    else:
        raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")

    return obs_diff, p


def compute_stat(x, y, method='pearson'):
    """Compute a correlation or R^2 statistic with its parametric p-value.

    Parameters
    ----------
    x, y : ndarray
        1-D arrays (NaN-free).
    method : {'pearson', 'spearman', 'r2'}
        Statistic to compute.

    Returns
    -------
    stat : float
        r, rho, or R^2.
    p : float
        Parametric p-value.
    """
    if method == 'pearson':
        return stats.pearsonr(x, y)
    elif method == 'spearman':
        r, p = stats.spearmanr(x, y)
        return r, p
    elif method == 'r2':
        X = x.reshape(-1, 1)
        model = LinearRegression().fit(X, y)
        r2 = r2_score(y, model.predict(X))
        n = len(x)
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
                    compute_stat(x, null_masked[i], method)[0]
                    for i in range(null.shape[0])
                ])
                if method == 'r2':
                    row_p.append(float(np.mean(perm_stats >= stat_obs)))
                elif stat_obs >= 0:
                    row_p.append(float(np.mean(perm_stats >= stat_obs)))
                else:
                    row_p.append(float(np.mean(perm_stats <= stat_obs)))
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
        excluded per subject; a subject with fewer than 2 usable regions yields NaN.
    """
    try:
        corr_func = {'spearman': sp.stats.spearmanr, 'pearson': sp.stats.pearsonr}[method]
    except KeyError:
        raise ValueError("method must be 'spearman' or 'pearson'")

    brain_maps    = np.asarray(brain_maps,    dtype=float)
    reference_map = np.asarray(reference_map, dtype=float)
    if brain_maps.ndim != 2:
        raise ValueError("brain_maps must be 2-D (subjects x regions)")
    if brain_maps.shape[1] != reference_map.shape[0]:
        raise ValueError("region axis of brain_maps must match len(reference_map)")

    rho = np.empty(brain_maps.shape[0])
    for i in range(brain_maps.shape[0]):
        m = ~(np.isnan(brain_maps[i]) | np.isnan(reference_map))
        rho[i] = corr_func(brain_maps[i][m], reference_map[m]).statistic if m.sum() >= 2 else np.nan
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
