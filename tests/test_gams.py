"""Tests for snaplab_tools.gams.

Everything runs on synthetic arrays. The exact vectorised single-boundary L2 detector is meant to
be identical to ruptures Dynp(model='l2', n_bkps=1) but ~9x faster; the equivalence tests lock that
in. Tests that need ruptures are skipped when it is not installed.
"""
import numpy as np
import pytest

from snaplab_tools.gams import (
    fit_gam, fit_series, partial_r2_smooth, build_signal, integrate_derivative,
    segment_windows_at, detect_change_point, segmentation_costs, bootstrap_gam_fits,
    _normalize_rows, _dynp_l2_costs, SIGNALS,
)


@pytest.fixture
def x_grid():
    return np.linspace(8.0, 22.0, 120)


@pytest.fixture
def curves(x_grid):
    """(n_series, n_grid) with a planted slope change at ~15, giving a decisive (tie-free)
    change point in the rate of change."""
    rng = np.random.default_rng(20260721)
    x    = (x_grid - x_grid[0]) / (x_grid[-1] - x_grid[0])            # 0..1
    x0   = (15.0 - x_grid[0]) / (x_grid[-1] - x_grid[0])
    ramp = np.where(x < x0, x, x0 + 0.25 * (x - x0))                  # slope change at x0
    amp  = rng.normal(size=(60, 1))
    return amp * ramp[None, :] + 0.05 * np.sin(2 * np.pi * x)[None, :] + 0.01 * rng.standard_normal((60, len(x_grid)))


@pytest.fixture
def gam_data():
    """(x, Y, x_grid) with a smooth nonlinear trend per column + noise."""
    rng  = np.random.default_rng(7)
    x    = rng.uniform(8.0, 22.0, 240)
    grid = np.linspace(x.min(), x.max(), 100)
    u    = (x - 8) / 14
    Y    = np.column_stack([np.sin(2.2 * u) * amp + 0.05 * rng.standard_normal(len(x))
                            for amp in (1.0, -0.7, 0.4)])
    return x, Y, grid


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------
def test_fit_gam_fixed_lam_used_verbatim(gam_data):
    x, Y, grid = gam_data
    _, _, _, _, gam = fit_gam(x, Y[:, 0], grid, n_splines=6, lam=0.3)
    assert np.isclose(float(np.ravel(gam.lam)[0]), 0.3)


def test_fit_gam_too_few_obs_returns_nan(gam_data):
    x, Y, grid = gam_data
    y = Y[:, 0].copy(); y[3:] = np.nan          # only 3 valid < n_splines
    pred, ci_lo, _, deriv, gam = fit_gam(x, y, grid, n_splines=6, lam=0.1)
    assert gam is None and ci_lo is None
    assert np.isnan(pred).all() and np.isnan(deriv).all()


def test_fit_series_reports_lams_and_gcv(gam_data):
    x, Y, grid = gam_data
    fixed = fit_series(x, Y, grid, n_splines=6, lam=0.2)
    assert fixed['pred'].shape == (3, len(grid)) and np.allclose(fixed['lams'], 0.2)
    gcv = fit_series(x, Y, grid, n_splines=8, lam='gcv')
    assert np.all(np.isfinite(gcv['lams'])) and not np.allclose(gcv['lams'], 0.6)


def test_partial_r2_smooth_bounds(gam_data):
    x, Y, grid = gam_data
    r = fit_series(x, Y, grid, n_splines=8, lam='gcv')
    pr2 = partial_r2_smooth(Y, x, r['gams'])
    assert pr2.shape == (3,) and np.all((pr2 > 0) & (pr2 <= 1))


def test_invalid_lam_string_rejected(gam_data):
    x, Y, grid = gam_data
    with pytest.raises(ValueError, match="lam"):
        fit_gam(x, Y[:, 0], grid, lam='reml')


# ---------------------------------------------------------------------------
# Signals & derivatives
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('signal', SIGNALS)
def test_build_signal_shape_no_leading_nan(curves, x_grid, signal):
    sig = build_signal(curves, x_grid, signal)
    assert sig.shape == curves.shape and np.isfinite(sig).all()


def test_integrate_derivative_matches_endpoint_difference(x_grid):
    # For smooth curves the central-difference derivative integrates back to the endpoint change
    # (fundamental theorem of calculus, up to O(h^2) quadrature error on a fine grid).
    u = (x_grid - x_grid[0]) / (x_grid[-1] - x_grid[0])
    smooth = np.vstack([np.sin(1.5 * np.pi * u) * a for a in (1.0, -0.7, 0.4)])
    d1  = build_signal(smooth, x_grid, 'deriv1')
    net = integrate_derivative(d1, x_grid, np.arange(len(x_grid)))
    assert np.allclose(net, smooth[:, -1] - smooth[:, 0], atol=5e-3)


def test_segment_windows_at_partition(x_grid):
    cl, cr, dl, dr = segment_windows_at(x_grid, 40)
    assert cl[-1] == cr[0] == 40                    # split column shared
    assert np.isclose(dl + dr, x_grid[-1] - x_grid[0])


# ---------------------------------------------------------------------------
# Change-point detection
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('signal', SIGNALS)
def test_change_point_within_window(curves, x_grid, signal):
    frac = 0.10
    r = detect_change_point(curves, x_grid, signal=signal, min_segment_frac=frac)
    T = len(x_grid)
    assert int(np.ceil(frac * T)) <= r['index'] <= T - int(np.ceil(frac * T))
    assert 0.0 <= r['var_explained'] <= 1.0 and r['cost_curve'].shape == (T,)


def test_detects_planted_slope_change(curves, x_grid):
    assert abs(detect_change_point(curves, x_grid, signal='deriv1')['location'] - 15.0) < 1.0


def test_default_single_change_point_shapes(curves, x_grid):
    r = detect_change_point(curves, x_grid, signal='deriv1')
    assert r['locations'] == [r['location']] and r['indices'] == [r['index']]


def test_search_bounds_constrains(curves, x_grid):
    r = detect_change_point(curves, x_grid, signal='deriv1', search_bounds=(18.0, 21.0))
    assert 18.0 <= r['location'] <= 21.0


def test_n_bkps_invalid(curves, x_grid):
    with pytest.raises(ValueError, match='n_bkps'):
        detect_change_point(curves, x_grid, signal='deriv1', n_bkps=0)


def test_segmentation_costs_monotone(curves, x_grid):
    costs = segmentation_costs(curves, x_grid, signal='deriv1', kmax=4)
    assert costs.shape == (4,) and np.all(np.diff(costs) <= 1e-9)


# ---------------------------------------------------------------------------
# Equivalence with ruptures (the point of the exact fast path)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('signal', SIGNALS)
def test_exact_matches_ruptures_l2(curves, x_grid, signal):
    pytest.importorskip('ruptures')
    exact = detect_change_point(curves, x_grid, signal=signal, cost=None)
    rup   = detect_change_point(curves, x_grid, signal=signal, cost='l2')
    assert exact['index'] == rup['index']


def test_n_bkps_returns_ordered_boundaries(curves, x_grid):
    pytest.importorskip('ruptures')
    r = detect_change_point(curves, x_grid, signal='deriv1', n_bkps=3)
    assert len(r['locations']) == 3 and r['locations'] == sorted(r['locations'])
    assert r['location'] == r['locations'][0]


def test_dynp_costs_match_ruptures(curves, x_grid):
    rpt = pytest.importorskip('ruptures')
    sig = _normalize_rows(build_signal(curves, x_grid, 'deriv1'), 'zscore')
    X   = np.ascontiguousarray(sig[np.all(np.isfinite(sig), axis=1)].T)
    mine = _dynp_l2_costs(X, 4)

    def seg_cost(bkps):
        bnds, tot = [0] + list(bkps) + [X.shape[0]], 0.0
        for a, b in zip(bnds[:-1], bnds[1:]):
            s = X[a:b]
            tot += ((s - s.mean(0)) ** 2).sum()
        return tot

    for k in range(1, 5):
        bkps = rpt.Dynp(model='l2', min_size=1, jump=1).fit(X).predict(n_bkps=k - 1)[:-1]
        assert np.isclose(mine[k - 1], seg_cost(bkps))


# ---------------------------------------------------------------------------
# Bootstrap engine
# ---------------------------------------------------------------------------
def _reduce_cp(preds, derivs, x_grid):
    """Module-level reduce so joblib can pickle it."""
    return {'cp': detect_change_point(preds, x_grid, signal='deriv1')['location'],
            'net': integrate_derivative(derivs, x_grid, np.arange(len(x_grid)))}


def test_bootstrap_engine_stacks_reduce_outputs(gam_data):
    x, Y, grid = gam_data
    frozen = fit_series(x, Y, grid, n_splines=8, lam='gcv')['lams']
    out = bootstrap_gam_fits(Y, x, grid, reduce=_reduce_cp, n_boot=8, n_splines=8,
                             lam=frozen, random_state=0, n_jobs=2, verbose=False)
    assert out['cp'].shape == (8,)                 # scalar per resample
    assert out['net'].shape == (8, Y.shape[1])     # (n_series,) per resample
