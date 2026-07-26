"""Cross-validated regression with nuisance control, PCA, and permutation testing.

The entry point is the :class:`Regression` class: hand it a feature matrix, a target, and
optionally a set of covariates, then call :meth:`Regression.run` for out-of-sample prediction
accuracy and :meth:`Regression.run_perm` for an empirical null.

Two design choices are worth knowing about.

*Repeated k-fold.* A single k-fold split is noisy, so ``run`` repeats the whole cross-validation
``n_rand_splits`` times under different shuffles and keeps the distribution of scores rather than a
point estimate. Summarise it however you like (the tutorials take the mean).

*Everything is fit inside the training fold.* Standardization, nuisance regression, and PCA are all
fit on training data only and applied to the held-out fold, so no test-set information leaks into
the model.

The module-level functions (:func:`run_reg`, :func:`run_perm`, :func:`my_cross_val_score`, ...) are
the machinery behind the class and can be called directly if you want a different loop structure.
"""
import numpy as np
import scipy as sp
from scipy import stats

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.metrics import make_scorer, r2_score, mean_squared_error, mean_absolute_error, explained_variance_score
from sklearn.decomposition import PCA
import copy
from tqdm import tqdm

__all__ = [
    'Regression',
    'corr_true_pred',
    'root_mean_squared_error',
    'shuffle_data',
    'get_cv',
    'my_cross_val_score',
    'run_reg',
    'run_perm',
]


class Regression():
    """Repeated k-fold cross-validated regression with an optional permutation test.

    Parameters
    ----------
    X : (n_obs, n_features) ndarray
        Predictors.
    y : (n_obs,) ndarray
        Target.
    c : (n_obs, n_covariates) ndarray or None
        Nuisance covariates. When given, `X` is residualized on `c` within each training fold
        and the fitted nuisance model is applied to the test fold. When None, the step is skipped.
    alg : {'linear', 'rr', 'lr', 'krr_lin', 'krr_rbf', 'svr_lin', 'svr_rbf'}
        Estimator: ordinary least squares, ridge, lasso, kernel ridge (linear/RBF), or support
        vector regression (linear/RBF). Default 'rr' (ridge).
    score : {'rmse', 'corr', 'r2', 'mse', 'mae', 'exp_var'}
        Primary scoring metric. Error metrics are wrapped with ``greater_is_better=False``, so
        they come back negated -- higher is always better in the stored scores.
    secondary_score : str or None
        A second metric reported alongside the primary one; same options as `score`. None
        disables it.
    n_splits : int
        Number of folds per cross-validation run.
    runpca : bool or str or int
        Dimensionality reduction applied inside each training fold. False disables it; an int
        requests that many components; '80%' keeps the components explaining 80% of cumulative
        variance; '1%' keeps every component explaining at least 1% individually.
    n_rand_splits : int
        Number of times the whole k-fold cross-validation is repeated under different shuffles.
    verbose : bool
        Print the settings block when ``run`` starts.

    Attributes
    ----------
    y_pred : (n_obs, n_rand_splits) ndarray
        Out-of-sample predictions from each repeat. Set by :meth:`run`.
    accuracy_mean, accuracy_std : (n_rand_splits,) ndarray
        Mean and standard deviation of the primary score across folds, per repeat. Set by
        :meth:`run`.
    secondary_accuracy_mean, secondary_accuracy_std : (n_rand_splits,) ndarray
        The same for the secondary score. Set by :meth:`run` when `secondary_score` is not None.
    accuracy_perm, secondary_accuracy_perm : (n_perm,) ndarray
        Null distributions of the primary and secondary scores. Set by :meth:`run_perm`.

    Examples
    --------
    >>> model = Regression(X, y, c=covariates, alg='rr', score='rmse', n_splits=5)
    >>> model.run()
    >>> model.run_perm(n_perm=1000)
    >>> observed = model.accuracy_mean.mean()
    """

    def __init__(self, X, y, c=None, alg='rr', score='rmse', secondary_score='corr', n_splits=10, runpca=False, n_rand_splits=100, verbose=True):
        self.X = X
        self.y = y
        self.c = c

        self.alg = alg
        self.score = score
        self.secondary_score = secondary_score
        self.n_splits = n_splits
        self.runpca = runpca
        self.n_rand_splits = n_rand_splits
        self.verbose = verbose

    def _print_settings(self):
        print('\tsettings:')
        print('\t\talg: {0}'.format(self.alg))
        print('\t\tscore: {0}'.format(self.score))
        print('\t\tsecondary score: {0}'.format(self.secondary_score))
        print('\t\tn_splits: {0}'.format(self.n_splits))
        print('\t\trunpca: {0}'.format(self.runpca))
        print('\t\tn_rand_splits: {0}'.format(self.n_rand_splits))

    def _get_reg(self):
        regs = {'linear': LinearRegression(),
                'rr': Ridge(),
                'lr': Lasso(),
                'krr_lin': KernelRidge(kernel='linear'),
                'krr_rbf': KernelRidge(kernel='rbf'),
                'svr_lin': SVR(kernel='linear'),
                'svr_rbf': SVR(kernel='rbf')
                }
        self.reg = regs[self.alg]

    def _get_scorer(self):
        if self.score == 'r2':
            self.scorer = make_scorer(r2_score, greater_is_better=True)
        elif self.score == 'corr':
            self.scorer = make_scorer(corr_true_pred, greater_is_better=True)
        elif self.score == 'mse':
            self.scorer = make_scorer(mean_squared_error, greater_is_better=False)
        elif self.score == 'rmse':
            self.scorer = make_scorer(root_mean_squared_error, greater_is_better=False)
        elif self.score == 'mae':
            self.scorer = make_scorer(mean_absolute_error, greater_is_better=False)
        elif self.score == 'exp_var':
            self.scorer = make_scorer(explained_variance_score, greater_is_better=True)
            
        if self.secondary_score == 'r2':
            self.secondary_scorer = make_scorer(r2_score, greater_is_better=True)
        elif self.secondary_score == 'corr':
            self.secondary_scorer = make_scorer(corr_true_pred, greater_is_better=True)
        elif self.secondary_score == 'mse':
            self.secondary_scorer = make_scorer(mean_squared_error, greater_is_better=False)
        elif self.secondary_score == 'rmse':
            self.secondary_scorer = make_scorer(root_mean_squared_error, greater_is_better=False)
        elif self.secondary_score == 'mae':
            self.secondary_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
        elif self.secondary_score == 'exp_var':
            self.secondary_scorer = make_scorer(explained_variance_score, greater_is_better=True)
        elif self.secondary_score is None:
            self.secondary_scorer = None

    def run(self):
        """Run the repeated cross-validation and store the score distributions.

        Repeats a full `n_splits`-fold cross-validation `n_rand_splits` times, each with a
        different shuffle of the observations (seeded by the repeat index, so results are
        reproducible). Populates ``y_pred``, ``accuracy_mean``, ``accuracy_std``, and -- when a
        secondary score is configured -- ``secondary_accuracy_mean`` / ``secondary_accuracy_std``.

        Returns
        -------
        None
            Results are stored on the instance.
        """
        if self.verbose:
            print('Pipeline: regression (out-of-sample regression)')
            self._print_settings()
        self._get_reg()
        self._get_scorer()

        y_pred = np.zeros((len(self.y), self.n_rand_splits))
        accuracy_mean = np.zeros(self.n_rand_splits)
        accuracy_std = np.zeros(self.n_rand_splits)
        if self.secondary_scorer is not None:
            secondary_accuracy_mean = np.zeros(self.n_rand_splits)
            secondary_accuracy_std = np.zeros(self.n_rand_splits)            

        for i in tqdm(np.arange(self.n_rand_splits)):
            outputs = run_reg(X=self.X, y=self.y, c=self.c, reg=self.reg,
                              scorer=self.scorer, secondary_scorer=self.secondary_scorer,
                              n_splits=self.n_splits, runpca=self.runpca,
                              seed=i)
            y_pred[:, i] = outputs['y_pred_out']
            accuracy_mean[i] = outputs['accuracy'].mean()
            accuracy_std[i] = outputs['accuracy'].std()
            if self.secondary_scorer is not None:
                secondary_accuracy_mean[i] = outputs['secondary_accuracy'].mean()
                secondary_accuracy_std[i] = outputs['secondary_accuracy'].std()
        
        print('Average prediction accuracy: {:.2f}+/-{:.2f} '.format(accuracy_mean.mean(), accuracy_std.mean()))
        self.y_pred = y_pred
        self.accuracy_mean = accuracy_mean
        self.accuracy_std = accuracy_std
        if self.secondary_scorer is not None:
            print('\tAverage prediction accuracy (secondary): {:.2f}+/-{:.2f} '.format(secondary_accuracy_mean.mean(), secondary_accuracy_std.mean()))
            self.secondary_accuracy_mean = secondary_accuracy_mean
            self.secondary_accuracy_std = secondary_accuracy_std

    def run_perm(self, n_perm=int(5e3)):
        """Build an empirical null by permuting the target and re-running cross-validation.

        Each permutation shuffles `y` against `X` (breaking the true association) and re-runs a
        single `n_splits`-fold cross-validation, keeping the mean score across folds. Compare the
        result against the observed score with :func:`snaplab_tools.nulls.get_null_p`.

        This is expensive: cost is roughly `n_perm` x `n_splits` model fits.

        Parameters
        ----------
        n_perm : int
            Number of permutations.

        Returns
        -------
        None
            Populates ``accuracy_perm`` and ``secondary_accuracy_perm`` on the instance.
        """
        print('Pipeline: prediction, permutation test')
        self._get_reg()
        self._get_scorer()

        outputs = run_perm(X=self.X, y=self.y, c=self.c, reg=self.reg,
                           scorer=self.scorer, secondary_scorer=self.secondary_scorer,
                           n_splits=self.n_splits, runpca=self.runpca, n_perm=n_perm)

        self.accuracy_perm = outputs['accuracy_perm']
        self.secondary_accuracy_perm = outputs['secondary_accuracy_perm']


def corr_true_pred(y_true, y_pred):
    """Pearson correlation between observed and predicted values, as a scorer.

    Both inputs are flattened first, so column vectors are handled. Wrapped with
    ``sklearn.metrics.make_scorer`` when ``score='corr'``.

    Parameters
    ----------
    y_true, y_pred : ndarray
        Observed and predicted targets.

    Returns
    -------
    float
        Pearson r (the p-value is discarded).
    """
    if type(y_true) == np.ndarray:
        y_true = y_true.flatten()
    if type(y_pred) == np.ndarray:
        y_pred = y_pred.flatten()

    r, p = sp.stats.pearsonr(y_true, y_pred)
    return r


def root_mean_squared_error(y_true, y_pred):
    """Root mean squared error, as a scorer.

    Parameters
    ----------
    y_true, y_pred : ndarray
        Observed and predicted targets. Reduction is over axis 0.

    Returns
    -------
    float or ndarray
        Square root of the mean squared error.
    """
    mse = np.mean((y_true - y_pred) ** 2, axis=0)
    rmse = np.sqrt(mse)
    return rmse


def shuffle_data(X, y, c, seed=0):
    """Jointly shuffle the rows of X, y, and c with a shared permutation.

    Because the same index order is applied to all three, the correspondence between
    observations is preserved -- this reorders the sample, it does not break the X-y link. Used
    to vary the k-fold partition across repeats in :func:`run_reg`.

    Parameters
    ----------
    X : (n_obs, n_features) or (n_obs,) ndarray
        Predictors.
    y : (n_obs,) ndarray
        Target.
    c : (n_obs, n_covariates) ndarray or None
        Covariates; None passes through as None.
    seed : int
        Seed for the shuffle.

    Returns
    -------
    X_shuf, y_shuf, c_shuf : ndarray
        The inputs reordered by a common random permutation.
    """
    np.random.seed(seed)
    idx = np.arange(y.shape[0])
    np.random.shuffle(idx)

    try:
        X_shuf = X[idx, :]
    except IndexError:
        X_shuf = X[idx]

    try:
        c_shuf = c[idx, :]
    except IndexError:
        c_shuf = c[idx]
    except TypeError:
        c_shuf = None

    y_shuf = y[idx]

    return X_shuf, y_shuf, c_shuf


def get_cv(y, n_splits=10):
    """Build a list of (train_idx, test_idx) pairs for k-fold cross-validation.

    Uses ``KFold(shuffle=False)``: randomisation comes from shuffling the data upstream in
    :func:`shuffle_data`, not from the splitter, so a given seed reproduces exactly.

    Parameters
    ----------
    y : (n_obs,) ndarray
        Target; only its length is used.
    n_splits : int
        Number of folds.

    Returns
    -------
    list of tuple of ndarray
        One ``(train_idx, test_idx)`` pair per fold.
    """
    my_cv = []

    kf = KFold(n_splits=n_splits, shuffle=False)

    for train_idx, test_idx in kf.split(y):
        my_cv.append((train_idx, test_idx))

    return my_cv


def my_cross_val_score(X, y, c, my_cv, reg, scorer, runpca=False, secondary_scorer=None):
    """Score a regressor over a set of folds, fitting all preprocessing inside each training fold.

    Per fold, in order: standardize `X`; if `c` is given, standardize it and residualize `X` on
    it (the nuisance model is fit on the training fold and applied to the test fold); optionally
    reduce with PCA; fit `reg` and score on the held-out fold.

    When `runpca` is a string the component count is chosen once, up front, from the full `X` --
    only the projection itself is refit per fold.

    Parameters
    ----------
    X : (n_obs, n_features) ndarray
        Predictors.
    y : (n_obs,) ndarray
        Target.
    c : (n_obs, n_covariates) ndarray or None
        Nuisance covariates; None skips the residualization step.
    my_cv : list of tuple of ndarray
        Fold indices, as returned by :func:`get_cv`.
    reg : estimator
        Any scikit-learn regressor. A deep copy is used for the nuisance model.
    scorer : callable
        Scorer with the ``scorer(estimator, X, y)`` signature (i.e. from ``make_scorer``).
    runpca : bool or str or int
        PCA setting; see :class:`Regression`.
    secondary_scorer : callable or None
        Optional second scorer, same signature.

    Returns
    -------
    dict
        ``y_pred_out`` -- (n_obs,) out-of-sample predictions, each observation predicted by the
        fold in which it was held out; ``accuracy`` -- (n_folds,) primary scores;
        ``secondary_accuracy`` -- (n_folds,) secondary scores, all NaN if `secondary_scorer`
        is None.
    """
    y_pred_out = np.zeros(y.shape)
    accuracy = np.zeros(len(my_cv), )
    secondary_accuracy = np.zeros(len(my_cv), )
    secondary_accuracy[:] = np.nan

    # find number of PCs
    if type(runpca) == str:
        pca = PCA(n_components=np.min(X.shape), svd_solver='full')
        pca.fit(StandardScaler().fit_transform(X))

        if runpca == '80%':
            cum_var = np.cumsum(pca.explained_variance_ratio_)
            n_components = np.where(cum_var >= 0.8)[0][0] + 1
            # print(n_components)
        elif runpca == '1%':
            var_idx = pca.explained_variance_ratio_ >= .01
            n_components = np.sum(var_idx)
            # print(n_components)

    elif type(runpca) == int:
        n_components = runpca

    for k in np.arange(len(my_cv)):
        tr = my_cv[k][0]
        te = my_cv[k][1]

        # Split into train test
        try:
            X_train = X[tr, :]
            X_test = X[te, :]
        except IndexError:
            X_train = X[tr]
            X_test = X[te]

        try:
            c_train = c[tr, :]
            c_test = c[te, :]
        except IndexError:
            c_train = c[tr]
            c_test = c[te]
        except TypeError:
            pass

        y_train = y[tr]
        y_test = y[te]

        # standardize predictors
        sc = StandardScaler()
        sc.fit(X_train)
        X_train = sc.transform(X_train)
        X_test = sc.transform(X_test)

        try:
            # standardize covariates
            sc = StandardScaler()
            sc.fit(c_train)
            c_train = sc.transform(c_train)
            c_test = sc.transform(c_test)

            # regress nuisance (X)
            nuis_reg = copy.deepcopy(reg)
            nuis_reg.fit(c_train, X_train)
            X_pred = nuis_reg.predict(c_train)
            X_train = X_train - X_pred
            X_pred = nuis_reg.predict(c_test)
            X_test = X_test - X_pred
        except:
            pass

        if type(runpca) == str or type(runpca) == int:
            pca = PCA(n_components=n_components, svd_solver='full')
            pca.fit(X_train)
            X_train = pca.transform(X_train)
            X_test = pca.transform(X_test)

        reg.fit(X_train, y_train)
        y_pred_out[te] = reg.predict(X_test)
        accuracy[k] = scorer(reg, X_test, y_test)
        if secondary_scorer is not None:
            secondary_accuracy[k] = secondary_scorer(reg, X_test, y_test)

    outputs = dict()
    outputs['y_pred_out'] = y_pred_out
    outputs['accuracy'] = accuracy
    outputs['secondary_accuracy'] = secondary_accuracy

    return outputs


def run_reg(X, y, c, reg, scorer, secondary_scorer=None, n_splits=10, runpca=False, seed=0):
    """One full cross-validation run: shuffle the sample, build folds, score.

    This is the unit that :meth:`Regression.run` repeats `n_rand_splits` times with different
    seeds.

    Parameters
    ----------
    X, y, c : ndarray
        Predictors, target, and optional covariates. See :class:`Regression`.
    reg : estimator
        Any scikit-learn regressor.
    scorer, secondary_scorer : callable or None
        Scorers with the ``scorer(estimator, X, y)`` signature.
    n_splits : int
        Number of folds.
    runpca : bool or str or int
        PCA setting; see :class:`Regression`.
    seed : int
        Seed for the shuffle that determines the fold partition.

    Returns
    -------
    dict
        As returned by :func:`my_cross_val_score`. Note that ``y_pred_out`` is in *shuffled*
        order, matching the permutation `seed` produced -- not the original row order of `y`.
    """
    X_shuf, y_shuf, c_shuf = shuffle_data(X=X, y=y, c=c, seed=seed)
    my_cv = get_cv(y_shuf, n_splits=n_splits)

    outputs = my_cross_val_score(X=X_shuf, y=y_shuf, c=c_shuf, my_cv=my_cv, reg=reg,
                                 scorer=scorer, runpca=runpca, secondary_scorer=secondary_scorer)
    
    return outputs


def run_perm(X, y, c, reg, scorer, secondary_scorer=None, n_splits=10, runpca=False, n_perm=int(5e3)):
    """Permutation null for out-of-sample prediction accuracy.

    Shuffles `y` alone (leaving `X` and `c` in place) to destroy the true association, then
    re-runs cross-validation and keeps the mean score across folds. The fold partition is fixed
    across permutations, so the only thing varying is the target ordering.

    Parameters
    ----------
    X, y, c : ndarray
        Predictors, target, and optional covariates. See :class:`Regression`.
    reg : estimator
        Any scikit-learn regressor.
    scorer, secondary_scorer : callable or None
        Scorers with the ``scorer(estimator, X, y)`` signature.
    n_splits : int
        Number of folds per permutation.
    runpca : bool or str or int
        PCA setting; see :class:`Regression`.
    n_perm : int
        Number of permutations.

    Returns
    -------
    dict
        ``accuracy_perm`` and ``secondary_accuracy_perm``, each (n_perm,) -- the null
        distribution of the fold-averaged score. Pass to
        :func:`snaplab_tools.nulls.get_null_p` alongside the observed score.
    """
    my_cv = get_cv(y, n_splits=n_splits)
    accuracy_perm = np.zeros(n_perm)
    secondary_accuracy_perm = np.zeros(n_perm)
    secondary_accuracy_perm[:] = np.nan

    for i in tqdm(np.arange(n_perm)):
        np.random.seed(i)
        idx = np.arange(y.shape[0])
        np.random.shuffle(idx)

        y_perm = y[idx].copy()

        outputs = my_cross_val_score(X=X, y=y_perm, c=c, my_cv=my_cv, reg=reg,
                                     scorer=scorer, runpca=runpca, secondary_scorer=secondary_scorer)
        accuracy_perm[i] = outputs['accuracy'].mean()
        secondary_accuracy_perm[i] = outputs['secondary_accuracy'].mean()

    outputs = dict()
    outputs['accuracy_perm'] = accuracy_perm
    outputs['secondary_accuracy_perm'] = secondary_accuracy_perm

    return outputs
