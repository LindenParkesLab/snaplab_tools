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


class Regression():
    def __init__(self, X, y, c=None, alg='rr', score='rmse', n_splits=10, runpca=False, n_rand_splits=100):
        self.X = X
        self.y = y
        self.c = c

        self.alg = alg
        self.score = score
        self.n_splits = n_splits
        self.runpca = runpca
        self.n_rand_splits = n_rand_splits

    def _print_settings(self):
        print('\tsettings:')
        print('\t\talg: {0}'.format(self.alg))
        print('\t\tscore: {0}'.format(self.score))
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

    def run(self):
        print('Pipeline: regression (out-of-sample regression)')
        self._print_settings()
        self._get_reg()
        self._get_scorer()

        accuracy_mean = np.zeros(self.n_rand_splits)
        accuracy_std = np.zeros(self.n_rand_splits)
        y_pred = np.zeros((len(self.y), self.n_rand_splits))

        for i in tqdm(np.arange(self.n_rand_splits)):
            accuracy, y_pred[:, i] = run_reg(X=self.X, y=self.y, c=self.c, reg=self.reg,
                                             scorer=self.scorer, n_splits=self.n_splits, runpca=self.runpca,
                                             seed=i)
            accuracy_mean[i] = accuracy.mean()
            accuracy_std[i] = accuracy.std()

        print('Average prediction accuracy: {:.2f}+/-{:.2f} '.format(accuracy_mean.mean(), accuracy_std.mean()))
        self.accuracy_mean = accuracy_mean
        self.accuracy_std = accuracy_std
        self.y_pred = y_pred

    def run_perm(self, n_perm=int(5e3)):
        print('Pipeline: prediction, permutation test')

        self._get_reg()
        self._get_scorer()

        accuracy_perm = run_perm(X=self.X, y=self.y, c=self.c, reg=self.reg,
                                scorer=self.scorer, n_splits=self.n_splits, runpca=self.runpca, n_perm=n_perm)

        self.accuracy_perm = accuracy_perm


def corr_true_pred(y_true, y_pred):
    if type(y_true) == np.ndarray:
        y_true = y_true.flatten()
    if type(y_pred) == np.ndarray:
        y_pred = y_pred.flatten()

    r, p = sp.stats.pearsonr(y_true, y_pred)
    return r


def root_mean_squared_error(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2, axis=0)
    rmse = np.sqrt(mse)
    return rmse


def shuffle_data(X, y, c, seed=0):
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
    my_cv = []

    kf = KFold(n_splits=n_splits, shuffle=False)

    for train_idx, test_idx in kf.split(y):
        my_cv.append((train_idx, test_idx))

    return my_cv


def my_cross_val_score(X, y, c, my_cv, reg, scorer, runpca=False):
    accuracy = np.zeros(len(my_cv), )
    y_pred_out = np.zeros(y.shape)

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
        accuracy[k] = scorer(reg, X_test, y_test)
        y_pred_out[te] = reg.predict(X_test)

    return accuracy, y_pred_out


def run_reg(X, y, c, reg, scorer, n_splits=10, runpca=False, seed=0):
    X_shuf, y_shuf, c_shuf = shuffle_data(X=X, y=y, c=c, seed=seed)

    my_cv = get_cv(y_shuf, n_splits=n_splits)

    accuracy, y_pred_out = my_cross_val_score(X=X_shuf, y=y_shuf, c=c_shuf, my_cv=my_cv, reg=reg,
                                              scorer=scorer, runpca=runpca)

    return accuracy, y_pred_out


def run_perm(X, y, c, reg, scorer, n_splits=10, runpca=False, n_perm=int(5e3)):
    my_cv = get_cv(y, n_splits=n_splits)

    permuted_acc = np.zeros(n_perm)

    for i in tqdm(np.arange(n_perm)):
        np.random.seed(i)
        idx = np.arange(y.shape[0])
        np.random.shuffle(idx)

        y_perm = y[idx].copy()

        temp_acc, y_pred_out_tmp = my_cross_val_score(X=X, y=y_perm, c=c, my_cv=my_cv, reg=reg,
                                                      scorer=scorer, runpca=runpca)
        permuted_acc[i] = temp_acc.mean()

    return permuted_acc
