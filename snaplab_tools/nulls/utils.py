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
    >>> observed = 0.42
    >>> p = get_null_p(observed, null_distribution, abs=True)
    """
    if version not in ('standard', 'smallest'):
        raise ValueError(f"version must be 'standard' or 'smallest'; got {version!r}")

    observed = float(observed)
    null = np.asarray(null, dtype=float)

    if abs:
        observed = np.abs(observed)
        null = np.abs(null)

    null = null[np.isfinite(null)]

    # A NaN observed statistic (a constant input, an all-NaN region) should propagate as a NaN
    # p-value rather than raise, so callers looping over regions get a result vector with holes
    # instead of an exception part-way through.
    if not np.isfinite(observed) or null.size == 0:
        return np.nan

    if version == 'standard':
        if observed >= 0:
            return float(np.sum(null >= observed) / null.size)
        return float(np.sum(null <= observed) / null.size)

    return float(np.min([np.sum(null >= observed) / null.size,
                         np.sum(observed >= null) / null.size]))
