"""Turning a null distribution into a p-value.

The one function here, :func:`get_null_p`, is deliberately generic: it does not care whether the
null came from spatial surrogates (:func:`snaplab_tools.nulls.generate_surrogates`), a permutation
test (:meth:`snaplab_tools.prediction.regression.Regression.run_perm`), or anything else. Give it
an observed statistic and a vector of null values and it returns the proportion of the null that is
at least as extreme.
"""
import numpy as np

__all__ = ['get_null_p']


def get_null_p(observed, null, version='standard', abs=False):
    """Proportion of a null distribution at least as extreme as the observed statistic.

    Parameters
    ----------
    observed : float
        The observed test statistic.
    null : (n_perms,) array-like
        Null distribution of the same statistic.
    version : {'standard', 'smallest'}
        How to pick the tail. 'standard' tests in the direction of the observed effect: the upper
        tail when `observed` is positive, the lower tail when negative. 'smallest' takes the
        smaller of the two one-tailed proportions, which is anti-conservative -- prefer
        ``abs=True`` for a genuinely two-tailed test.
    abs : bool
        Test magnitudes rather than signed values, by taking the absolute value of both
        `observed` and `null` first. This is the usual choice for a two-tailed test of a
        correlation, where sign is not the hypothesis.

    Returns
    -------
    float
        The p-value, in [0, 1]. Note the minimum attainable value is ``1 / len(null)`` -- a p of
        0 means "smaller than this null can resolve", so report it as ``p < 1/n_perms`` rather
        than as zero.

    Examples
    --------
    >>> observed = 0.42
    >>> p = get_null_p(observed, null_distribution, abs=True)
    """
    if abs:
        observed = np.abs(observed)
        null = np.abs(null)

    if version == 'standard':
        if observed >= 0:
            p_val = np.sum(null >= observed) / len(null)
        elif observed <= 0:
            p_val = np.sum(null <= observed) / len(null)
    elif version == 'smallest':
        p_val = np.min([np.sum(null >= observed) / len(null),
                        np.sum(observed >= null) / len(null)])

    return p_val
