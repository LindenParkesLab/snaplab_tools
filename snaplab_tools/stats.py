import numpy as np
import scipy as sp
from scipy import stats


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
