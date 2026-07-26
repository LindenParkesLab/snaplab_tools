"""Tests for snaplab_tools.prediction.regression.

The nuisance-control tests pin down behaviour that used to fail silently: a bare ``except: pass``
around the residualization block meant any failure there produced results with no nuisance control
at all, indistinguishable from passing no covariates.
"""
import numpy as np
import pytest

from snaplab_tools.prediction.regression import Regression, get_cv, my_cross_val_score, shuffle_data


@pytest.fixture
def confounded():
    """A dataset where a single confound drives both the features and the target.

    Predicting y from X succeeds strongly without nuisance control and should collapse with it.
    """
    rng = np.random.default_rng(0)
    n = 200
    confound = rng.standard_normal(n)
    X = confound[:, None] * rng.uniform(0.5, 1.5, 40)[None, :] + 0.3 * rng.standard_normal((n, 40))
    y = 2.0 * confound + 0.3 * rng.standard_normal(n)
    return X, y, confound


def _score(X, y, c):
    model = Regression(X=X, y=y, c=c, alg='rr', score='corr', secondary_score=None,
                       n_splits=5, n_rand_splits=3, runpca=False, verbose=False)
    model.run()
    return model.accuracy_mean.mean()


def test_no_covariates_recovers_the_confounded_effect(confounded):
    X, y, _ = confounded
    assert _score(X, y, None) > 0.9


def test_covariates_remove_the_confounded_effect(confounded):
    X, y, confound = confounded
    assert _score(X, y, confound[:, np.newaxis]) < 0.3


def test_one_dimensional_covariate_controls_like_a_column(confounded):
    """A single covariate passed as a 1-D vector must actually be controlled for.

    This previously fell through StandardScaler's 2-D requirement into a bare `except: pass`, so
    `c=df['age'].values` silently produced completely uncontrolled results.
    """
    X, y, confound = confounded
    one_d = _score(X, y, confound)
    two_d = _score(X, y, confound[:, np.newaxis])

    assert one_d == pytest.approx(two_d, abs=1e-9)
    assert one_d < 0.3


def test_bad_covariate_shape_raises_rather_than_being_swallowed(confounded):
    """A covariate array of the wrong length is an error, not a reason to skip nuisance control."""
    X, y, confound = confounded
    with pytest.raises(Exception):
        Regression(X=X, y=y, c=confound[:5, np.newaxis], alg='rr', score='corr',
                   secondary_score=None, n_splits=5, n_rand_splits=1, verbose=False).run()


def test_my_cross_val_score_returns_out_of_sample_predictions(confounded):
    from sklearn.linear_model import Ridge
    from sklearn.metrics import make_scorer
    from snaplab_tools.prediction.regression import corr_true_pred

    X, y, _ = confounded
    scorer = make_scorer(corr_true_pred, greater_is_better=True)
    out = my_cross_val_score(X, y, None, get_cv(y, n_splits=5), Ridge(), scorer)

    assert out['y_pred_out'].shape == y.shape
    assert out['accuracy'].shape == (5,)
    assert np.all(np.isnan(out['secondary_accuracy']))


def test_shuffle_data_keeps_rows_aligned(confounded):
    X, y, confound = confounded
    c = confound[:, np.newaxis]
    X_s, y_s, c_s = shuffle_data(X, y, c, seed=0)

    # The same permutation is applied to all three, so the X-y-c correspondence survives.
    order = [int(np.flatnonzero((y == value))[0]) for value in y_s[:20]]
    assert np.allclose(X_s[:20], X[order])
    assert np.allclose(c_s[:20], c[order])


def test_shuffle_data_passes_none_through(confounded):
    X, y, _ = confounded
    assert shuffle_data(X, y, None, seed=0)[2] is None
