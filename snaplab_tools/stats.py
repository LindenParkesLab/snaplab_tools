import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm


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


def partial_corr_controlled(df, predictor, outcome, covars):
    """Covariate-controlled partial correlation between a predictor and an outcome.

    Rows missing any of ``predictor``/``outcome``/``covars`` are dropped, then both the
    predictor and the outcome are residualized on the covariates (OLS, intercept included)
    and the Pearson correlation of the residuals is returned. The two-tailed p-value is taken
    from the predictor's coefficient in the OLS fit ``outcome ~ predictor + covars`` (so it
    carries the correct residual degrees of freedom). A one-tailed p-value for a pre-specified
    positive direction is also returned.

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
        With keys ``r`` (residual Pearson r), ``p_two`` (two-tailed p from the OLS coefficient),
        ``p_one_pos`` (one-tailed p for the positive hypothesis), ``n`` (rows used), and
        ``resid_x``/``resid_y`` (the covariate residuals of predictor and outcome).
    """
    d = df[[predictor, outcome] + covars].dropna()
    n = len(d)
    Xc = sm.add_constant(d[covars])
    rx = sm.OLS(d[predictor], Xc).fit().resid
    ry = sm.OLS(d[outcome],   Xc).fit().resid
    r  = sp.stats.pearsonr(rx.values, ry.values)[0]
    fit = sm.OLS(d[outcome], sm.add_constant(d[[predictor] + covars])).fit()
    p_two = fit.pvalues[predictor]
    p_one_pos = p_two / 2 if r > 0 else 1 - p_two / 2   # one-tailed, hypothesis = positive
    return dict(r=r, p_two=p_two, p_one_pos=p_one_pos, n=n,
                resid_x=rx.values, resid_y=ry.values)
