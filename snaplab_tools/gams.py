"""Generic GAM fitting, derivative signals, change-point detection, and a bootstrap engine.

Everything here is domain-neutral: you fit a penalized 1D spline (an age curve, a dose-response
curve, anything) to each column of a data matrix and evaluate it on a shared grid, then optionally
detect change points in the fitted curves (or their derivatives) and bootstrap the whole thing over
observations. Nothing knows about brains, ages, or S-A axes -- project-specific reductions live in
the calling project and plug into ``bootstrap_gam_fits`` via a reduction callback.

Vocabulary (consistent across the module):
    x        : (n_obs,)          predictor values, one per observation
    y        : (n_obs,)          a single response column
    Y        : (n_obs, n_series) response matrix, one column per series to fit independently
    x_grid   : (n_grid,)         evaluation grid the fitted curves are sampled on
    curves   : (n_series, n_grid) fitted values (or a derivative of them) on x_grid
    nuisance : (n_obs, n_nuisance) linear covariates entered alongside the smooth

Sections: fitting -> signals & derivatives -> change-point detection -> bootstrap engine. Each is
usable on its own. ``pygam`` is required; ``ruptures`` is optional (only for >1 change point or a
non-L2 cost model) and imported lazily.
"""
from contextlib import nullcontext

import numpy as np
import pygam
from joblib import Parallel, delayed

try:
    # Pin BLAS to one thread per worker so N parallel processes don't oversubscribe the cores
    # (each GAM's linear algebra is tiny; process-level parallelism is the win).
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None

# Signal options understood by build_signal / detect_change_point.
SIGNALS = ('pred', 'deriv1', 'deriv2')

# Grid searched by fit_gam when lam='gcv' (log-spaced smoothing penalties).
_DEFAULT_LAM_GRID = np.logspace(-3, 3, 25)


# =============================================================================
# Fitting
# =============================================================================

def fit_gam(x, y, x_grid, n_splines=4, lam=0.1, show_ci=False, nuisance=None,
            lam_grid=_DEFAULT_LAM_GRID):
    """Fit a penalized spline of y on x (+ optional linear nuisance) and evaluate on x_grid.

    NaNs in y are dropped (per-observation missingness). A column with fewer than n_splines valid
    observations cannot be fit and yields NaN curves and gam=None.

    Parameters
    ----------
    x : (n_obs,) ndarray
        Predictor values.
    y : (n_obs,) ndarray
        Response for a single series.
    x_grid : (n_grid,) ndarray
        Evaluation points.
    n_splines : int
        Spline basis dimension for the smooth term.
    lam : float or 'gcv'
        Smoothing penalty for the smooth. A float fixes it; 'gcv' selects it by grid search over
        lam_grid (generalized cross-validation), readable afterwards from the returned gam.lam.
    show_ci : bool
        Whether to compute 95% confidence intervals.
    nuisance : (n_obs, n_nuisance) ndarray or None
        Linear covariates entered as l() terms. When predicting on x_grid they are fixed at their
        within-sample means so the curve reflects the smooth only.
    lam_grid : array-like
        Candidate penalties searched when lam='gcv'; ignored otherwise.

    Returns
    -------
    pred : (n_grid,) ndarray
        Fitted values on x_grid.
    ci_lo, ci_hi : (n_grid,) ndarray or None
        95% confidence band, or None when show_ci is False.
    deriv : (n_grid,) ndarray
        Backward-difference derivative (units / x); NaN at index 0.
    gam : pygam.LinearGAM or None
        Fitted model, or None if the column had too few valid observations.
    """
    mask = ~np.isnan(y)
    Y    = y[mask]

    # A column can be entirely (or nearly) NaN; pygam raises on an empty target, so skip these and
    # return NaN curves -- downstream consumers already tolerate NaN series.
    if mask.sum() < n_splines:
        nan_curve = np.full(len(x_grid), np.nan)
        return nan_curve, None, None, nan_curve.copy(), None

    # 'gcv' selects lam by grid search below; build the terms with a numeric placeholder.
    _is_gcv = isinstance(lam, str) and lam == 'gcv'
    if isinstance(lam, str) and not _is_gcv:
        raise ValueError(f"lam must be a float or 'gcv'; got {lam!r}")
    _term_lam = 0.6 if _is_gcv else lam

    if nuisance is not None:
        nuisance        = np.asarray(nuisance)
        nuisance_masked = nuisance[mask]
        X_fit           = np.column_stack([x[mask], nuisance_masked])
        terms           = pygam.s(0, n_splines=n_splines, lam=_term_lam)
        for i in range(1, nuisance.shape[1] + 1):
            terms = terms + pygam.l(i)
        nuisance_means = nuisance_masked.mean(axis=0)
        X_pred = np.column_stack([x_grid, np.tile(nuisance_means, (len(x_grid), 1))])
    else:
        X_fit  = x[mask].reshape(-1, 1)
        terms  = pygam.s(0, n_splines=n_splines, lam=_term_lam)
        X_pred = x_grid.reshape(-1, 1)

    gam = pygam.LinearGAM(terms)
    if _is_gcv:
        gam.gridsearch(X_fit, Y, lam=lam_grid, progress=False)
    else:
        gam.fit(X_fit, Y)
    pred  = gam.predict(X_pred)
    dt    = np.diff(x_grid)
    deriv = np.concatenate([[np.nan], np.diff(pred) / dt])

    ci_lo, ci_hi = None, None
    if show_ci:
        ci    = gam.confidence_intervals(X_pred, width=0.95)
        ci_lo = ci[:, 0]
        ci_hi = ci[:, 1]

    return pred, ci_lo, ci_hi, deriv, gam


def fit_series(x, Y, x_grid, n_splines=4, lam=0.1, demean=True, show_ci=False, nuisance=None,
               lam_grid=_DEFAULT_LAM_GRID):
    """Fit a GAM to every column of Y independently (see fit_gam).

    Parameters
    ----------
    x : (n_obs,) ndarray
    Y : (n_obs, n_series) ndarray
        One column per series to fit.
    x_grid : (n_grid,) ndarray
    n_splines, lam, show_ci, lam_grid : see fit_gam. lam may be 'gcv' to select the penalty per
        series by grid search over lam_grid.
    demean : bool
        If True, each column is mean-centred (ignoring NaNs) before fitting.
    nuisance : (n_obs, n_nuisance) ndarray or None

    Returns
    -------
    dict
        'pred', 'deriv' : (n_series, n_grid); 'ci_lo'/'ci_hi' : (n_series, n_grid) or None;
        'gams' : list of fitted models (None where a column was unfit); 'lams' : (n_series,) the
        smooth penalty actually used per series (the grid-searched value when lam='gcv', NaN where
        unfit) -- freeze these to reuse the same smoothing in the bootstrap.
    """
    n_ser  = Y.shape[1]
    preds  = np.full((n_ser, len(x_grid)), np.nan)
    derivs = np.full((n_ser, len(x_grid)), np.nan)
    ci_los = np.full((n_ser, len(x_grid)), np.nan) if show_ci else None
    ci_his = np.full((n_ser, len(x_grid)), np.nan) if show_ci else None
    lams   = np.full(n_ser, np.nan)
    gams   = []

    for i in range(n_ser):
        y = Y[:, i].copy()
        if demean:
            y = y - np.nanmean(y)
        pred, ci_lo, ci_hi, deriv, gam = fit_gam(
            x, y, x_grid, n_splines=n_splines, lam=lam,
            show_ci=show_ci, nuisance=nuisance, lam_grid=lam_grid)
        preds[i]  = pred
        derivs[i] = deriv
        if show_ci:
            ci_los[i] = ci_lo
            ci_his[i] = ci_hi
        if gam is not None:
            lams[i] = float(np.ravel(gam.lam)[0])   # smooth penalty actually used
        gams.append(gam)

    return {'pred': preds, 'deriv': derivs, 'ci_lo': ci_los, 'ci_hi': ci_his,
            'lams': lams, 'gams': gams}


def partial_r2_smooth(Y, x, gams, nuisance=None, demean=True):
    """Partial R^2 of the smooth term for each column, using already-fitted GAMs.

    Compares each full model against a reduced model with the nuisance terms only (or the intercept
    when nuisance is None):  partial_R2 = (R2_full - R2_reduced) / (1 - R2_reduced). Columns with
    near-zero variance or gam=None are returned as NaN.

    Parameters
    ----------
    Y : (n_obs, n_series) ndarray
        The same response matrix the GAMs were fit to.
    x : (n_obs,) ndarray
    gams : list of pygam.LinearGAM or None
        Fitted full models, one per column (e.g. fit_series(...)['gams']).
    nuisance : (n_obs, n_nuisance) ndarray or None
        Must match the covariates used during fitting.
    demean : bool
        Must match the demean flag used during fitting.

    Returns
    -------
    (n_series,) ndarray
        Partial R^2 of the smooth term; NaN for skipped columns.
    """
    n_ser      = Y.shape[1]
    partial_r2 = np.full(n_ser, np.nan)

    for i in range(n_ser):
        y = Y[:, i].copy()
        if demean:
            y = y - np.nanmean(y)
        mask = ~np.isnan(y)
        yv   = y[mask]
        x_m  = x[mask]

        SS_tot = np.sum((yv - yv.mean()) ** 2)
        if SS_tot < 1e-12:
            continue

        gam_full = gams[i]
        if gam_full is None:      # too few valid observations to fit (see fit_gam)
            continue
        if nuisance is not None:
            nuis_m     = nuisance[mask]
            X_full     = np.column_stack([x_m, nuis_m])
            y_hat_full = gam_full.predict(X_full)

            # reduced model: nuisance only (linear terms, no smooth)
            nuis_terms = pygam.l(0)
            for j in range(1, nuis_m.shape[1]):
                nuis_terms = nuis_terms + pygam.l(j)
            gam_red   = pygam.LinearGAM(nuis_terms).fit(nuis_m, yv)
            y_hat_red = gam_red.predict(nuis_m)
        else:
            y_hat_full = gam_full.predict(x_m.reshape(-1, 1))
            y_hat_red  = np.full(len(yv), yv.mean())

        R2_full = 1 - np.sum((yv - y_hat_full) ** 2) / SS_tot
        R2_red  = 1 - np.sum((yv - y_hat_red)  ** 2) / SS_tot
        denom   = 1 - R2_red
        if denom > 1e-10:
            partial_r2[i] = (R2_full - R2_red) / denom

    return partial_r2


# =============================================================================
# Signals & derivatives
# =============================================================================

def build_signal(curves, x_grid, signal):
    """Derive the signal to analyse from a stack of fitted curves.

    Parameters
    ----------
    curves : (n_series, n_grid) ndarray
        Fitted values on x_grid.
    x_grid : (n_grid,) ndarray
    signal : {'pred', 'deriv1', 'deriv2'}
        'pred' -> the curves themselves; 'deriv1'/'deriv2' -> first/second derivative w.r.t. x
        (central differences via np.gradient with edge_order=2, so there is no leading NaN and
        deriv2 does not hook at the grid extremes).

    Returns
    -------
    (n_series, n_grid) ndarray
    """
    curves = np.asarray(curves, float)
    if signal == 'pred':
        return curves
    d1 = np.gradient(curves, x_grid, axis=1, edge_order=2)
    if signal == 'deriv1':
        return d1
    if signal == 'deriv2':
        return np.gradient(d1, x_grid, axis=1, edge_order=2)
    raise ValueError(f"signal must be one of {SIGNALS}; got {signal!r}")


def integrate_derivative(deriv, x_grid, cols):
    """Trapezoidal integral of a (n_series, n_grid) derivative over the columns `cols` (net change).

    Non-finite grid columns (e.g. a leading finite-difference NaN) are dropped; all-NaN series stay
    NaN.
    """
    xc = x_grid[cols]
    d  = deriv[:, cols]
    finite = np.isfinite(xc) & np.isfinite(d).any(axis=0)
    return np.trapezoid(d[:, finite], xc[finite], axis=1)


def segment_windows_at(x_grid, split_idx):
    """Column indices and durations for the two segments split at grid index `split_idx`.

    Returns (cols_left, cols_right, dur_left, dur_right); the split column is shared by both so a
    derivative integrated over each window sums to the full-range change.
    """
    cols_l = np.arange(0, split_idx + 1)
    cols_r = np.arange(split_idx, len(x_grid))
    dur_l  = x_grid[split_idx] - x_grid[0]
    dur_r  = x_grid[-1]        - x_grid[split_idx]
    return cols_l, cols_r, dur_l, dur_r


# =============================================================================
# Change-point detection
# =============================================================================

def _normalize_rows(sig, mode):
    """Per-row normalization across the grid axis (axis=1), NaN-aware."""
    if mode is None:
        return sig
    if mode == 'center':
        return sig - np.nanmean(sig, axis=1, keepdims=True)
    if mode == 'zscore':
        mu = np.nanmean(sig, axis=1, keepdims=True)
        sd = np.nanstd(sig, axis=1, keepdims=True)
        return (sig - mu) / np.where(sd < 1e-12, 1.0, sd)
    raise ValueError(f"normalize must be None, 'center' or 'zscore'; got {mode!r}")


def _segment_costs(X):
    """Single-change-point L2 cost, vectorized via cumulative sums.

    X : (T, d) observations (one grid point per row). Returns (taus, costs) where costs[k] is the
    within-segment SS of splitting into left=[0:taus[k]) and right=[taus[k]:T), for every interior
    boundary taus[k] in 1..T-1.
    """
    T = X.shape[0]
    S1 = np.cumsum(X, axis=0)
    S2 = np.cumsum(X ** 2, axis=0)
    tot1, tot2 = S1[-1], S2[-1]
    taus = np.arange(1, T)
    nL = taus[:, None].astype(float)
    nR = (T - taus)[:, None].astype(float)
    sumL, sqL = S1[taus - 1], S2[taus - 1]
    sumR, sqR = tot1 - sumL, tot2 - sqL
    ssL = (sqL - sumL ** 2 / nL).sum(axis=1)
    ssR = (sqR - sumR ** 2 / nR).sum(axis=1)
    return taus, ssL + ssR


def _dynp_l2_costs(X, kmax):
    """Exact DP segmentation (== ruptures Dynp, model='l2'). Best total within-segment SS for
    K=1..kmax contiguous segments. X: (T, d)."""
    T = X.shape[0]
    S1 = np.vstack([np.zeros(X.shape[1]), np.cumsum(X, 0)])
    S2 = np.vstack([np.zeros(X.shape[1]), np.cumsum(X ** 2, 0)])

    def seg(i, j):
        n = j - i
        s, sq = S1[j] - S1[i], S2[j] - S2[i]
        return float((sq - s ** 2 / n).sum())

    dp = np.full((kmax + 1, T + 1), np.inf)
    dp[0, 0] = 0.0
    for k in range(1, kmax + 1):
        for t in range(k, T + 1):
            for s in range(k - 1, t):
                c = dp[k - 1, s] + seg(s, t)
                if c < dp[k, t]:
                    dp[k, t] = c
    return dp[1:kmax + 1, T]


def _ruptures_bkps(X, model, min_size, n_bkps):
    """n_bkps ordered breakpoint indices via ruptures Dynp (lazy import); trailing endpoint dropped."""
    try:
        import ruptures as rpt
    except ImportError as e:  # ruptures is optional; only needed for a non-L2 cost or n_bkps > 1
        raise ImportError("this code path requires the optional 'ruptures' package "
                          "(pip install ruptures).") from e
    algo = rpt.Dynp(model=model, min_size=min_size, jump=1).fit(np.ascontiguousarray(X))
    return [int(b) for b in algo.predict(n_bkps=n_bkps)[:-1]]   # drop the final endpoint (== T)


def detect_change_point(curves, x_grid, *, signal='deriv1', normalize='zscore',
                        min_segment_frac=0.10, search_bounds=None, cost=None, n_bkps=1):
    """Multivariate change-point detection across a stack of 1D curves.

    Splits x_grid into contiguous segments at the location(s) minimising the within-segment sum of
    squared deviations from each segment's per-series mean of ``signal`` (multivariate L2 cost ==
    1D-ordered k-means with a contiguity constraint == ruptures Dynp). Per-row z-scoring makes the
    result reflect reorganization of the joint PATTERN rather than a few high-amplitude series.

    With n_bkps=1 and cost=None it uses a fast exact vectorised L2 detector -- provably identical to
    ruptures Dynp(model='l2', n_bkps=1) but ~9x faster, ideal for a bootstrap hot path. n_bkps > 1
    or a non-L2 ``cost`` string routes to ruptures Dynp.

    Parameters
    ----------
    curves : (n_series, n_grid) ndarray
        Fitted values on x_grid.
    x_grid : (n_grid,) ndarray
    signal : {'deriv1', 'deriv2', 'pred'}
        Detect on the rate of change (default), acceleration, or the level. See build_signal.
    normalize : {'zscore', 'center', None}
        Per-series normalization across the grid before detection.
    min_segment_frac : float
        Minimum fraction of the grid each segment must span (guards the unreliable derivative tails).
    search_bounds : (lo, hi) or None
        Optional hard x-window the change point must fall within. Exact single-boundary
        (cost=None, n_bkps=1) backend only.
    cost : None or str
        None -> exact L2. A ruptures cost model string ('rbf', 'normal', 'cosine', 'l1', ...)
        detects with that cost via ruptures instead.
    n_bkps : int
        Number of change points to detect (>= 1). n_bkps > 1 requires ruptures.

    Returns
    -------
    dict
        'location', 'index'   : the FIRST change point (x value and x_grid index),
        'locations', 'indices': all detected change points (length n_bkps, ordered),
        'var_explained'       : 1 - SS(all segments)/SS(1 segment) of the normalized signal,
        'cost_curve'          : within-segment L2 SS per candidate split (x_grid-aligned; None
                                unless the exact single-boundary backend ran).
    """
    if n_bkps < 1:
        raise ValueError(f"n_bkps must be >= 1; got {n_bkps}")
    sig = _normalize_rows(build_signal(curves, x_grid, signal), normalize)
    finite = np.all(np.isfinite(sig), axis=1)   # keep series finite across the whole grid
    X = sig[finite].T                           # (T=n_grid, n_valid_series)
    grid = np.asarray(x_grid)
    T = X.shape[0]
    min_size = max(1, int(np.ceil(min_segment_frac * T)))

    if cost is None and n_bkps == 1:
        taus, costs = _segment_costs(X)
        adm = (taus >= min_size) & (taus <= T - min_size)
        if search_bounds is not None:
            a0, a1 = search_bounds
            adm &= (grid[taus] >= a0) & (grid[taus] <= a1)
        if not adm.any():
            raise ValueError("No admissible change point; relax min_segment_frac or search_bounds.")
        idxs = [int(taus[adm][int(np.argmin(costs[adm]))])]
        cost_curve = np.full(T, np.nan)
        cost_curve[taus] = costs
    else:
        if search_bounds is not None:
            raise ValueError("search_bounds is only supported for the exact L2 single-boundary "
                             "backend (cost=None, n_bkps=1).")
        idxs = sorted(_ruptures_bkps(X, cost or 'l2', min_size, n_bkps))
        cost_curve = None

    # L2 variance-explained of the full segmentation (descriptive; comparable across cost models)
    tot_ss = float((X.var(axis=0) * T).sum())
    bnds = [0] + idxs + [T]
    split_ss = float(sum(((X[a:b] - X[a:b].mean(0)) ** 2).sum() for a, b in zip(bnds[:-1], bnds[1:])))
    locs = [float(grid[i]) for i in idxs]
    return dict(location=locs[0], index=idxs[0],
                locations=locs, indices=idxs,
                var_explained=float(1 - split_ss / tot_ss) if tot_ss > 0 else np.nan,
                cost_curve=cost_curve)


def segmentation_costs(curves, x_grid, *, signal='deriv1', normalize='zscore', kmax=4):
    """Within-segment SS for K=1..kmax contiguous segments (exact DP; == ruptures Dynp l2).

    Diagnostic for how many segments the data support. On smooth curves the SS falls off smoothly
    with K (no sharp elbow), so use it to confirm that a chosen K captures the dominant regime shift
    rather than to *select* K, which tends to over-segment. Returns costs (kmax,) for K=1..kmax.
    """
    sig = _normalize_rows(build_signal(curves, x_grid, signal), normalize)
    X = sig[np.all(np.isfinite(sig), axis=1)].T
    return np.asarray(_dynp_l2_costs(X, kmax))


def select_n_segments(curves, x_grid, *, signal='deriv1', normalize='zscore', cost='l2',
                      min_segment_frac=0.10, pen=None):
    """Penalty-based selection of the number of segments (ruptures Pelt).

    Unlike segmentation_costs (which returns the SS-vs-K curve to inspect), this lets a penalty pick
    the breakpoint count directly. On smooth curves the result is penalty-sensitive and tends to
    over-segment at a plain BIC penalty, so treat it as exploratory -- sweep ``pen`` and report the
    range. Requires the optional 'ruptures' package.

    Parameters
    ----------
    curves, x_grid, signal, normalize : as in detect_change_point.
    cost : str
        ruptures cost model ('l2', 'rbf', 'normal', ...).
    min_segment_frac : float
        Minimum fraction of the grid per segment.
    pen : float or None
        Penalty per breakpoint. None -> a BIC-like default (n_series * log(T)).

    Returns
    -------
    dict
        'n_segments' : breakpoints + 1; 'locations' : x of the selected change points; 'pen' : used.
    """
    try:
        import ruptures as rpt
    except ImportError as e:
        raise ImportError("select_n_segments requires the optional 'ruptures' package "
                          "(pip install ruptures).") from e
    sig = _normalize_rows(build_signal(curves, x_grid, signal), normalize)
    X = sig[np.all(np.isfinite(sig), axis=1)].T
    T = X.shape[0]
    min_size = max(1, int(np.ceil(min_segment_frac * T)))
    if pen is None:
        pen = X.shape[1] * np.log(T)      # BIC-like default (dimension * log n)
    algo = rpt.Pelt(model=cost, min_size=min_size, jump=1).fit(np.ascontiguousarray(X))
    bkps = algo.predict(pen=pen)[:-1]     # drop the trailing endpoint (== T)
    grid = np.asarray(x_grid)
    return dict(n_segments=len(bkps) + 1,
                locations=[float(grid[b]) for b in bkps],
                pen=float(pen))


# =============================================================================
# Bootstrap engine
# =============================================================================

def _bootstrap_refit_one(idx, Y, x, nuisance, x_grid, n_splines, lam, demean, reduce):
    """One resample: refit every column on rows `idx`, then apply reduce(preds, derivs, x_grid).

    Runs in a worker process; BLAS pinned to one thread so many workers don't oversubscribe cores.
    lam may be a scalar (same for every column) or a per-column array of frozen penalties.
    """
    x_b    = x[idx]
    Y_b    = Y[idx]
    nuis_b = nuisance[idx] if nuisance is not None else None
    n_series = Y.shape[1]
    n_grid   = len(x_grid)
    preds  = np.full((n_series, n_grid), np.nan)
    derivs = np.full((n_series, n_grid), np.nan)
    lam_per_series = np.ndim(lam) > 0   # frozen per-column penalties vs a single scalar

    with (threadpool_limits(1) if threadpool_limits is not None else nullcontext()):
        for i in range(n_series):
            y = Y_b[:, i].copy()
            if demean:
                y = y - np.nanmean(y)
            lam_i = lam[i] if lam_per_series else lam
            if lam_per_series and not np.isfinite(lam_i):
                lam_i = 0.1     # column unfit on the full sample; harmless fallback
            pred, _, _, deriv, _ = fit_gam(x_b, y, x_grid, n_splines=n_splines, lam=lam_i,
                                           nuisance=nuis_b)
            preds[i]  = pred
            derivs[i] = deriv

    return reduce(preds, derivs, x_grid)


def bootstrap_gam_fits(Y, x, x_grid, *, reduce, n_boot=200, n_splines=4, lam=0.1,
                       nuisance=None, demean=True, random_state=None, n_jobs=-1, verbose=True):
    """Bootstrap column-wise GAM fits over observations, reduced per resample by `reduce`.

    Each iteration resamples observations (rows of Y) with replacement, refits a GAM to every column
    once, and calls ``reduce(preds, derivs, x_grid) -> dict`` on the (n_series, n_grid) fitted-value
    and derivative stacks. The per-resample dicts are stacked into arrays keyed the same way, so the
    return is ``{key: ndarray of shape (n_boot, ...)}``. This keeps the expensive refit generic
    while all domain-specific summaries live in your reduce callback.

    Resample indices are drawn once in the main process from ``random_state``, so results are
    identical for any ``n_jobs``.

    Parameters
    ----------
    Y : (n_obs, n_series) ndarray
    x : (n_obs,) ndarray
    x_grid : (n_grid,) ndarray
    reduce : callable
        ``reduce(preds, derivs, x_grid) -> dict``. Runs in each worker, so it must be picklable
        (a module-level function, or functools.partial over one). Must return the SAME set of keys
        every resample.
    lam : float, 'gcv', or (n_series,) array
        Per-column smoothing; an array freezes a per-column penalty (e.g. GCV values selected once
        on the full sample). 'gcv' re-selects per resample (slow -- prefer freezing).
    n_boot, n_splines, demean, nuisance, random_state, n_jobs, verbose : as usual.

    Returns
    -------
    dict
        {key: ndarray (n_boot, ...)} -- the reduce outputs stacked over resamples. Empty if n_boot=0.
    """
    rng     = np.random.default_rng(random_state)
    n_obs   = Y.shape[0]
    idx_all = [rng.integers(0, n_obs, size=n_obs) for _ in range(n_boot)]

    results = Parallel(n_jobs=n_jobs, verbose=(10 if verbose else 0))(
        delayed(_bootstrap_refit_one)(idx, Y, x, nuisance, x_grid, n_splines, lam, demean, reduce)
        for idx in idx_all)

    if not results:
        return {}
    return {k: np.asarray([r[k] for r in results]) for k in results[0]}
