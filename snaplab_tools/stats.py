import numpy as np
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
