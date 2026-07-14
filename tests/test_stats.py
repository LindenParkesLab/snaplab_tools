"""Tests for snaplab_tools.stats.partial_corr_controlled."""

import numpy as np
import pandas as pd
import scipy as sp

from snaplab_tools.stats import partial_corr_controlled


def _make_df(n=200, seed=0):
    rng = np.random.default_rng(seed)
    c1, c2 = rng.normal(size=n), rng.normal(size=n)
    x = 0.6 * c1 - 0.3 * c2 + rng.normal(size=n)
    y = 0.4 * x + 0.5 * c1 + 0.2 * c2 + rng.normal(size=n)
    return pd.DataFrame({"x": x, "y": y, "c1": c1, "c2": c2})


def test_r_matches_manual_residual_correlation():
    df = _make_df()
    covars = ["c1", "c2"]
    res = partial_corr_controlled(df, "x", "y", covars)
    # Manual: residualize x and y on [const, c1, c2], correlate residuals.
    Z = np.column_stack([np.ones(len(df)), df[covars].values])
    rx = df["x"].values - Z @ np.linalg.lstsq(Z, df["x"].values, rcond=None)[0]
    ry = df["y"].values - Z @ np.linalg.lstsq(Z, df["y"].values, rcond=None)[0]
    r_manual = sp.stats.pearsonr(rx, ry)[0]
    assert np.isclose(res["r"], r_manual)
    assert np.allclose(res["resid_x"], rx)
    assert np.allclose(res["resid_y"], ry)


def test_p_two_matches_partial_correlation_t_test():
    # The OLS-coefficient p-value must equal the analytic partial-correlation t-test p-value,
    # with residual df = n - 2 - n_covars.
    df = _make_df(seed=3)
    covars = ["c1", "c2"]
    res = partial_corr_controlled(df, "x", "y", covars)
    n, r = res["n"], res["r"]
    dof = n - 2 - len(covars)
    t = r * np.sqrt(dof / (1 - r**2))
    p_analytic = 2 * sp.stats.t.sf(abs(t), dof)
    assert np.isclose(res["p_two"], p_analytic, atol=1e-10)


def test_one_tailed_is_half_when_positive():
    df = _make_df(seed=1)  # positive x->y association
    res = partial_corr_controlled(df, "x", "y", ["c1", "c2"])
    assert res["r"] > 0
    assert np.isclose(res["p_one_pos"], res["p_two"] / 2)


def test_dropna_rows_are_excluded():
    df = _make_df(seed=2)
    df.loc[df.index[:15], "y"] = np.nan  # 15 missing outcomes
    res = partial_corr_controlled(df, "x", "y", ["c1", "c2"])
    assert res["n"] == len(df) - 15


# ---------------------------------------------------------------------------
# paired_ttest_vs_reference
# ---------------------------------------------------------------------------

from snaplab_tools.stats import paired_ttest_vs_reference, significance_stars


def _make_repeated_df(n=400, seed=0):
    rng = np.random.default_rng(seed)
    rest = rng.normal(5.0, 0.3, n)
    movies = rest - 0.2 + rng.normal(0, 0.1, n)   # lower than rest
    tasks = rest - 0.4 + rng.normal(0, 0.1, n)    # much lower than rest
    return pd.DataFrame({"Rest": rest, "Movies": movies, "Tasks": tasks})


def test_paired_ttest_matches_scipy_ttest_rel():
    df = _make_repeated_df()
    res = paired_ttest_vs_reference(df, reference="Rest", correction=None)
    for cond in ("Movies", "Tasks"):
        t, p = sp.stats.ttest_rel(df[cond], df["Rest"])
        assert np.isclose(res.loc[cond, "t"], t)
        assert np.isclose(res.loc[cond, "p"], p)
        assert np.isclose(res.loc[cond, "mean_diff"], (df[cond] - df["Rest"]).mean())
    # Reference excluded; conditions preserve dataframe order.
    assert list(res.index) == ["Movies", "Tasks"]


def test_default_reference_is_first_column():
    df = _make_repeated_df()
    res = paired_ttest_vs_reference(df)
    assert list(res.index) == ["Movies", "Tasks"]


def test_pairwise_nan_handling():
    df = _make_repeated_df()
    df.loc[0, "Movies"] = np.nan
    df.loc[1, "Rest"] = np.nan
    res = paired_ttest_vs_reference(df)
    assert res.loc["Movies", "n"] == len(df) - 2   # two rows dropped pairwise
    assert res.loc["Tasks", "n"] == len(df) - 1     # only the NaN Rest row dropped


def test_holm_correction_matches_statsmodels():
    from statsmodels.stats.multitest import multipletests
    df = _make_repeated_df()
    res = paired_ttest_vs_reference(df, correction="holm")
    expected = multipletests(res["p"].to_numpy(), method="holm")[1]
    assert np.allclose(res["p_corr"].to_numpy(), expected)


def test_alternative_less_gives_one_sided_p():
    df = _make_repeated_df()
    two = paired_ttest_vs_reference(df, alternative="two-sided", correction=None)
    less = paired_ttest_vs_reference(df, alternative="less", correction=None)
    # Effects are genuinely negative (condition < reference), so one-sided 'less'
    # p is half the two-sided p.
    for cond in ("Movies", "Tasks"):
        assert np.isclose(less.loc[cond, "p"], two.loc[cond, "p"] / 2)


def test_significance_stars_thresholds():
    assert significance_stars(0.0005) == "***"
    assert significance_stars(0.005) == "**"
    assert significance_stars(0.03) == "*"
    assert significance_stars(0.5) == "ns"
    assert significance_stars(np.nan) == "ns"


# ---------------------------------------------------------------------------
# decoupling_test
# ---------------------------------------------------------------------------

from snaplab_tools.stats import decoupling_test


def _make_coupling_data(n_sub=30, n_reg=400, rest_slope=1.0, task_slope=0.3, seed=0):
    rng = np.random.default_rng(seed)
    sa = np.arange(n_reg, dtype=float)
    int_rest = rest_slope * sa[None, :] + rng.normal(0, 50, (n_sub, n_reg))
    int_task = task_slope * sa[None, :] + rng.normal(0, 50, (n_sub, n_reg))
    return int_rest, int_task, sa


def test_decoupling_matches_manual():
    int_rest, int_task, sa = _make_coupling_data()
    rr, rt, t, p, n = decoupling_test(int_rest, int_task, sa, alternative='two-sided')
    n_sub = int_rest.shape[0]
    rho_r = np.array([sp.stats.spearmanr(int_rest[i], sa).statistic for i in range(n_sub)])
    rho_t = np.array([sp.stats.spearmanr(int_task[i], sa).statistic for i in range(n_sub)])
    tt = sp.stats.ttest_rel(np.arctanh(rho_r), np.arctanh(rho_t))
    assert np.isclose(rr, rho_r.mean())
    assert np.isclose(rt, rho_t.mean())
    assert np.isclose(t, tt.statistic)
    assert np.isclose(p, tt.pvalue)
    assert n == n_sub


def test_decoupling_default_is_one_sided_greater():
    # Default alternative='greater' tests A-coupling > B-coupling; with a genuine
    # decoupling (rho_rest > rho_task) its p is half the two-sided p.
    int_rest, int_task, sa = _make_coupling_data(rest_slope=1.0, task_slope=0.2)
    _, _, t_g, p_g, _ = decoupling_test(int_rest, int_task, sa)                      # default 'greater'
    _, _, t_2, p_2, _ = decoupling_test(int_rest, int_task, sa, alternative='two-sided')
    assert np.isclose(t_g, t_2)          # same statistic
    assert t_g > 0
    assert np.isclose(p_g, p_2 / 2)      # one-sided halving in the tested direction


def test_decoupling_pearson_matches_manual():
    int_rest, int_task, sa = _make_coupling_data()
    rr, rt, t, p, n = decoupling_test(int_rest, int_task, sa,
                                      alternative='two-sided', method='pearson')
    n_sub = int_rest.shape[0]
    rho_r = np.array([sp.stats.pearsonr(int_rest[i], sa).statistic for i in range(n_sub)])
    rho_t = np.array([sp.stats.pearsonr(int_task[i], sa).statistic for i in range(n_sub)])
    tt = sp.stats.ttest_rel(np.arctanh(rho_r), np.arctanh(rho_t))
    assert np.isclose(rr, rho_r.mean())
    assert np.isclose(t, tt.statistic)
    assert np.isclose(p, tt.pvalue)


def test_decoupling_bad_method_raises():
    int_rest, int_task, sa = _make_coupling_data()
    try:
        decoupling_test(int_rest, int_task, sa, method='kendall')
        assert False, "expected ValueError on unknown method"
    except ValueError:
        pass


def test_decoupling_detects_weaker_task_coupling():
    int_rest, int_task, sa = _make_coupling_data(rest_slope=1.0, task_slope=0.2)
    rr, rt, t, p, n = decoupling_test(int_rest, int_task, sa)
    assert rr > rt        # rest coupling stronger
    assert t > 0 and p < 0.05


def test_decoupling_identical_inputs_are_degenerate():
    # Identical conditions -> all paired differences are exactly zero, so the
    # paired t-test has zero variance and scipy returns NaN. Couplings still match.
    int_rest, _, sa = _make_coupling_data()
    rr, rt, t, p, n = decoupling_test(int_rest, int_rest, sa)
    assert np.isclose(rr, rt)
    assert np.isnan(t) and np.isnan(p)


def test_decoupling_not_significant_when_only_noise_differs():
    # Same coupling strength, different noise -> no systematic decoupling.
    rng = np.random.default_rng(7)
    sa = np.arange(400, dtype=float)
    int_rest = sa[None, :] + rng.normal(0, 50, (40, 400))
    int_task = sa[None, :] + rng.normal(0, 50, (40, 400))
    _, _, t, p, n = decoupling_test(int_rest, int_task, sa)
    assert p > 0.05


def test_decoupling_nan_region_is_masked_per_subject():
    int_rest, int_task, sa = _make_coupling_data(seed=1)
    # Introduce a NaN region for one subject; that region should be dropped for
    # that subject only, matching a manual masked correlation.
    int_rest2 = int_rest.copy()
    int_rest2[0, 5] = np.nan
    rr, rt, t, p, n = decoupling_test(int_rest2, int_task, sa)
    m = ~np.isnan(int_rest2[0])
    rho0 = sp.stats.spearmanr(int_rest2[0][m], sa[m]).statistic
    rho0_clean = sp.stats.spearmanr(int_rest[0], sa).statistic
    assert not np.isclose(rho0, rho0_clean)   # masking actually changed subject 0
    assert np.isfinite(t) and np.isfinite(p)


def test_decoupling_drops_constant_subject():
    # A subject with a constant INT array has an undefined Spearman coupling; it
    # must be dropped rather than poisoning the whole test with NaN.
    int_rest, int_task, sa = _make_coupling_data(seed=4, rest_slope=1.0, task_slope=0.2)
    int_rest[0, :] = 3.0   # constant -> rho undefined for subject 0
    rr, rt, t, p, n = decoupling_test(int_rest, int_task, sa)
    assert n == int_rest.shape[0] - 1        # subject 0 dropped
    assert np.isfinite(t) and np.isfinite(p)
    assert p < 0.05                          # remaining subjects still show decoupling


def test_decoupling_shape_validation():
    int_rest, int_task, sa = _make_coupling_data(n_reg=400)
    try:
        decoupling_test(int_rest, int_task, sa[:399])
        assert False, "expected ValueError on region-length mismatch"
    except ValueError:
        pass
