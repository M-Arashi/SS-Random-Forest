"""
simulation.py
=============

Simulation study for Stein-shrunken leaf estimation in random forests.

The study is organised around one question: ensemble averaging and leaf-level
shrinkage both attack the same quantity, the sampling variance of a leaf mean,
so how much does the second still buy once the first has been applied?

Experiment A, ``substitution``
    The shrinkage gain as a function of the ensemble size B and the minimum
    leaf size n_0, holding the process and the sample size fixed.  This is the
    experiment that measures the substitution directly.

Experiment B, ``grid``
    Accuracy across sample sizes and dimensions at two ensemble sizes, one
    small and one large, against the full set of forest and penalised
    competitors.

Experiment C, ``extended``
    The expensive competitors, BART, local linear forests and natural-gradient
    boosting, at the largest configuration of each process.

Experiment D, ``equivalence``
    How many additional trees a standard forest needs in order to match a
    shrunken forest of a given size.  This is the practically relevant
    summary, because trees cost memory and prediction latency.

Within a replication every method sees the same training and test samples, and
RF, SSRF-EB and SSRF-JS share one fitted forest, so those three differ only in
the value stored at each leaf.  Risk is reported both against the observed
responses and against the true conditional mean, the latter being the
quantity the theory speaks about.

Usage
-----
    python simulation.py --experiment substitution --reps 40
    python simulation.py --experiment grid        --reps 40
    python simulation.py --experiment extended    --reps 20
    python simulation.py --experiment equivalence --reps 40
"""

from __future__ import annotations

import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd

from dgps import DGPS, signal_to_noise
from ssrf import ShrunkenForest
from calibration import (ForestCalibrator, calibrate_by_cv, ij_reliability,
                         per_tree_predictions)
import competitors as C
import evaluation as E

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(os.path.dirname(HERE), "outputs", "sim")
os.makedirs(OUT_DIR, exist_ok=True)

DGP_LIST = ("dgp1", "dgp2", "dgp3")


# ---------------------------------------------------------------------------
#  Experiment A: the substitution surface
# ---------------------------------------------------------------------------

def experiment_substitution(reps, n=600, p=10, n_test=1500,
                            B_list=(1, 3, 5, 10, 25, 50, 100, 250, 500),
                            leaf_list=(2, 5, 20), seed0=10_000):
    rows = []
    for dgp in DGP_LIST:
        gen = DGPS[dgp]
        for n0 in leaf_list:
            for r in range(reps):
                rng = np.random.default_rng(seed0 + 7919 * r + 13 * n0)
                X, y, f = gen(n, p, rng)
                Xt, yt, ft = gen(n_test, p, rng)
                # one forest of the largest size, truncated to each B, so that
                # the comparison across B is paired rather than independent
                big = ShrunkenForest(n_estimators=max(B_list),
                                     min_samples_leaf=n0,
                                     random_state=seed0 + r).fit(X, y)
                all_trees = big.trees_
                for B in B_list:
                    big.trees_ = all_trees[:B]
                    pr_rf = big.predict(Xt, "rf")
                    pr_eb = big.predict(Xt, "ssrf")
                    pr_js = big.predict(Xt, "ssrf_js")
                    lam = float(np.mean(big.mean_lambda("ssrf_js")[:B]))
                    for name, pr in (("RF", pr_rf), ("SSRF", pr_eb),
                                     ("SSRF-JS", pr_js)):
                        rows.append(dict(
                            experiment="substitution", dgp=dgp, n=n, p=p,
                            B=B, min_leaf=n0, rep=r, method=name,
                            mse=E.mse(pr, yt), mae=E.mae(pr, yt),
                            mse_true=float(np.mean((pr - ft) ** 2)),
                            lam_js=lam,
                            leaves=float(np.mean(big.leaf_counts())),
                            snr=signal_to_noise(y, f)))
                big.trees_ = all_trees
            print(f"  substitution {dgp} n0={n0} done", flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
#  Experiment B: accuracy grid with competitors
# ---------------------------------------------------------------------------

GRID_NP = ((300, 10), (600, 20), (1200, 20), (2000, 24))


def _fit_core(X, y, Xt, B, n0, seed, with_competitors=True):
    preds, times, diag = {}, {}, {}

    t0 = time.perf_counter()
    F = ShrunkenForest(n_estimators=B, min_samples_leaf=n0,
                       random_state=seed).fit(X, y)
    t_fit = time.perf_counter() - t0
    for key, rule in (("RF", "rf"), ("SSRF", "ssrf"), ("SSRF-JS", "ssrf_js")):
        preds[key] = F.predict(Xt, rule)
        times[key] = t_fit
    diag["lam_eb"] = float(np.mean(F.mean_lambda("ssrf")))
    diag["lam_js"] = float(np.mean(F.mean_lambda("ssrf_js")))
    diag["leaves"] = float(np.mean(F.leaf_counts()))
    diag["n_eff"] = float(np.mean(F.effective_leaf_sizes()))
    s2, t2 = F.variance_components()
    diag["sigma2"] = float(np.mean(s2))
    diag["tau2"] = float(np.mean(t2))

    t0 = time.perf_counter()
    H = ShrunkenForest(n_estimators=B, min_samples_leaf=max(2, n0 // 2),
                       honest=True, bootstrap=False, max_samples=0.5,
                       random_state=seed + 1).fit(X, y)
    t_h = time.perf_counter() - t0
    preds["RF-H"], preds["SSRF-H"] = H.predict(Xt, "rf"), H.predict(Xt, "ssrf")
    preds["SSRF-JS-H"] = H.predict(Xt, "ssrf_js")
    times["RF-H"] = times["SSRF-H"] = times["SSRF-JS-H"] = t_h

    t0 = time.perf_counter()
    R = ShrunkenForest(n_estimators=B, min_samples_leaf=n0,
                       sigma_method="huber", tau_method="dl",
                       random_state=seed).fit(X, y)
    preds["SSRF-rob"] = R.predict(Xt, "ssrf_js")
    times["SSRF-rob"] = time.perf_counter() - t0

    if with_competitors:
        for name, fn in (("RidgeRF", C.ridge_rf), ("LassoRF", C.lasso_rf)):
            t0 = time.perf_counter()
            preds[name] = fn(X, y, Xt, B, n0, seed)
            times[name] = time.perf_counter() - t0
        t0 = time.perf_counter()
        preds["GBM"] = C.gbm(X, y, Xt, max(B, 100), seed)
        times["GBM"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        pr_t, best = C.rf_tuned(X, y, Xt, B, seed)
        preds["RF-tuned"] = pr_t
        times["RF-tuned"] = time.perf_counter() - t0
        diag["tuned_leaf"], diag["tuned_mtry"] = best
        # the same shrinkage applied on top of the cross-validated partition
        t0 = time.perf_counter()
        T = ShrunkenForest(n_estimators=B, min_samples_leaf=best[0],
                           max_features=best[1], random_state=seed).fit(X, y)
        preds["SSRF-tuned"] = T.predict(Xt, "ssrf_js")
        times["SSRF-tuned"] = times["RF-tuned"] + (time.perf_counter() - t0)
    return preds, times, diag


def _score(preds, times, diag, y_te, f_te, rng, meta):
    idx = rng.permutation(y_te.size)
    cal, ev = idx[: y_te.size // 2], idx[y_te.size // 2:]
    ref = (preds["SSRF-JS"] - y_te) ** 2
    rows = []
    for name, pr in preds.items():
        lo, hi = E.split_conformal(pr[cal], y_te[cal], pr[ev], alpha=0.05)
        stat, pval = E.dm_test((pr - y_te) ** 2, ref, h=1)
        rows.append(dict(meta, method=name,
                         mse=E.mse(pr, y_te), mae=E.mae(pr, y_te),
                         mse_true=float(np.mean((pr - f_te) ** 2)),
                         coverage=float(np.mean((y_te[ev] >= lo) & (y_te[ev] <= hi))),
                         interval_score=E.interval_score(lo, hi, y_te[ev]),
                         interval_width=float(np.mean(hi - lo)),
                         dm_stat=stat, dm_p=pval,
                         runtime=times.get(name, np.nan), **diag))
    return rows


def experiment_grid(reps, B_small=10, B_large=500, n0=5, seed0=20_000):
    rows = []
    for dgp in DGP_LIST:
        gen = DGPS[dgp]
        for (n, p) in GRID_NP:
            t0 = time.perf_counter()
            for r in range(reps):
                rng = np.random.default_rng(seed0 + 104_729 * r + n + p)
                X, y, f = gen(n, p, rng)
                Xt, yt, ft = gen(max(1000, n), p, rng)
                for B in (B_small, B_large):
                    preds, times, diag = _fit_core(X, y, Xt, B, n0,
                                                   seed0 + r, True)
                    meta = dict(experiment="grid", dgp=dgp, n=n, p=p, B=B,
                                min_leaf=n0, rep=r, snr=signal_to_noise(y, f))
                    rows.extend(_score(preds, times, diag, yt, ft, rng, meta))
            print(f"  grid {dgp} n={n} p={p} in "
                  f"{time.perf_counter()-t0:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(
                os.path.join(OUT_DIR, "sim_grid.csv"), index=False)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
#  Experiment C: expensive competitors
# ---------------------------------------------------------------------------

def experiment_extended(reps, n=1200, p=20, n0=5, B=25, seed0=30_000):
    rows = []
    for dgp in DGP_LIST:
        gen = DGPS[dgp]
        t0 = time.perf_counter()
        for r in range(reps):
            rng = np.random.default_rng(seed0 + 1_299_709 * r)
            X, y, f = gen(n, p, rng)
            Xt, yt, ft = gen(1000, p, rng)
            preds, times, diag = _fit_core(X, y, Xt, B, n0, seed0 + r, True)

            t1 = time.perf_counter()
            llf = C.LocalLinearForest(n_estimators=max(100, B),
                                      min_samples_leaf=max(n0, 10),
                                      random_state=seed0 + r).fit(X, y)
            preds["LLF"] = llf.predict(Xt)
            times["LLF"] = time.perf_counter() - t1

            t1 = time.perf_counter()
            pb = C.bart(X, y, Xt, n_draws=1000, n_burn=1000,
                        seed=seed0 + r)
            if pb is not None:
                preds["BART"], times["BART"] = pb, time.perf_counter() - t1

            t1 = time.perf_counter()
            pn = C.ngboost(X, y, Xt, n_estimators=400, seed=seed0 + r)
            if pn is not None:
                preds["NGBoost"], times["NGBoost"] = pn[0], time.perf_counter() - t1

            meta = dict(experiment="extended", dgp=dgp, n=n, p=p, B=B,
                        min_leaf=n0, rep=r, snr=signal_to_noise(y, f))
            rows.extend(_score(preds, times, diag, yt, ft, rng, meta))
        print(f"  extended {dgp} in {time.perf_counter()-t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(
            os.path.join(OUT_DIR, "sim_extended.csv"), index=False)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
#  Experiment H: how far the heterogeneous rule sits from the common-factor one
# ---------------------------------------------------------------------------

def experiment_heterogeneity(reps=15, p=10, n_list=(300, 1200),
                             B_list=(10, 100, 500), n0=5, n_test=1000,
                             seed0=80_000):
    """Measure the across-tree covariance term exactly.

    The rule in use applies a different weight in every tree, while the
    aggregation result is stated for a single common factor.  The two differ by
    an exact term, the across-tree empirical covariance between the fitted
    weight and the tree's deviation from its own anchor.  This experiment
    computes that term, the bound on it, and its size relative to the
    prediction's own deviation from the anchor, for both leaf rules.
    """
    rows = []
    for dgp in DGP_LIST:
        gen = DGPS[dgp]
        for n in n_list:
            for B in B_list:
                for r in range(reps):
                    rng = np.random.default_rng(seed0 + 431 * r + n + B)
                    X, y, f = gen(n, p, rng)
                    Xt, yt, ft = gen(n_test, p, rng)
                    F = ShrunkenForest(n_estimators=B, min_samples_leaf=n0,
                                       random_state=seed0 + r).fit(X, y)
                    raw = F.per_tree(Xt, "rf")                   # (m, B)
                    theta_b = np.array([rec.theta for rec in F.trees_],
                                       dtype=float)
                    theta_b = np.where(np.isfinite(theta_b), theta_b,
                                       F.y_mean_)
                    D = raw - theta_b[None, :]
                    for rule, key in (("EB", "ssrf"), ("JS", "ssrf_js")):
                        shr = F.per_tree(Xt, key)
                        # the weight each tree applied at each test point,
                        # recovered from the values themselves so that the
                        # measurement does not depend on how it was formed
                        lam = np.where(np.abs(D) > 1e-9,
                                       (shr - theta_b[None, :]) / np.where(
                                           np.abs(D) > 1e-9, D, 1.0), np.nan)
                        lam = np.where(np.isfinite(lam), lam, 1.0)
                        lambar = lam.mean(axis=1)
                        Dbar = D.mean(axis=1)
                        Cx = ((lam - lambar[:, None])
                              * (D - Dbar[:, None])).mean(axis=1)
                        f_sl = shr.mean(axis=1)
                        f_lam = theta_b.mean() + lambar * (
                            raw.mean(axis=1) - theta_b.mean())
                        s_lam = lam.std(axis=1)
                        s_D = D.std(axis=1)
                        scale = np.mean(np.abs(raw.mean(axis=1)
                                               - theta_b.mean()))
                        rows.append(dict(
                            experiment="heterogeneity", dgp=dgp, n=n, p=p,
                            B=B, min_leaf=n0, rep=r, rule=rule,
                            mean_abs_C=float(np.mean(np.abs(Cx))),
                            mean_bound=float(np.mean(s_lam * s_D)),
                            rel_C=float(np.mean(np.abs(Cx)) / max(scale, 1e-12)),
                            s_lambda=float(np.mean(s_lam)),
                            identity_err=float(np.max(np.abs(
                                f_sl - (f_lam + Cx)))),
                            mse_true=float(np.mean((f_sl - ft) ** 2)),
                            mse_common=float(np.mean((f_lam - ft) ** 2))))
                print(f"  heterogeneity {dgp} n={n} B={B} done", flush=True)
        pd.DataFrame(rows).to_csv(
            os.path.join(OUT_DIR, "sim_heterogeneity.csv"), index=False)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
#  Experiment F: calibrating the contraction at the forest level
# ---------------------------------------------------------------------------

def experiment_calibration(reps=20, p=10, n_list=(300, 1200),
                           B_list=(10, 100, 500), n0=5, n_test=1500,
                           seed0=60_000):
    """Does estimating lambda* directly beat estimating it inside a tree?

    Four predictors are compared on one fitted forest per replication: the
    unshrunken forest, the leaf-level Stein rule, and the forest-level rule
    with its factor calibrated by cross-validation and by the out-of-bag
    route with the infinitesimal jackknife.  The cross-validated factor and
    the leaf-level factor are recorded so that the two can be compared
    directly rather than only through their risks.
    """
    rows = []
    for dgp in DGP_LIST:
        gen = DGPS[dgp]
        for n in n_list:
            for B in B_list:
                for r in range(reps):
                    rng = np.random.default_rng(seed0 + 1013 * r + n + B)
                    X, y, f = gen(n, p, rng)
                    Xt, yt, ft = gen(n_test, p, rng)
                    seed = seed0 + r

                    def fit_forest(Xa, ya, sd, _B=B, _n0=n0):
                        return ShrunkenForest(n_estimators=_B,
                                              min_samples_leaf=_n0,
                                              random_state=sd).fit(Xa, ya)

                    t0 = time.perf_counter()
                    F = fit_forest(X, y, seed)
                    t_fit = time.perf_counter() - t0

                    preds = {"RF": F.predict(Xt, "rf"),
                             "SSRF-JS": F.predict(Xt, "ssrf_js")}
                    times = {"RF": t_fit, "SSRF-JS": t_fit}

                    t0 = time.perf_counter()
                    cal = ForestCalibrator(method="cv", random_state=seed)
                    cal.fit(F, X, y, fit_forest=fit_forest)
                    pg, _ = cal.predict(F, Xt, "global")
                    t_cv = time.perf_counter() - t0
                    preds["Cal-CV"] = pg
                    times["Cal-CV"] = t_fit + t_cv
                    pl, lam_l = cal.predict(F, Xt, "local")
                    preds["Cal-CV-local"] = pl
                    times["Cal-CV-local"] = t_fit + t_cv

                    t0 = time.perf_counter()
                    cij = ForestCalibrator(method="ij", random_state=seed)
                    cij.fit(F, X, y)
                    pij, _ = cij.predict(F, Xt, "global")
                    preds["Cal-IJ"] = pij
                    times["Cal-IJ"] = t_fit + (time.perf_counter() - t0)

                    # the factor that would have been optimal, by grid search
                    theta = float(y.mean())
                    pr = per_tree_predictions(F, Xt).mean(axis=1)
                    grid_l = np.linspace(0.0, 1.4, 141)
                    risks = [float(np.mean((theta + l * (pr - theta) - ft) ** 2))
                             for l in grid_l]
                    lam_oracle = float(grid_l[int(np.argmin(risks))])

                    for name, pv in preds.items():
                        rows.append(dict(
                            experiment="calibration", dgp=dgp, n=n, p=p, B=B,
                            min_leaf=n0, rep=r, method=name,
                            mse=E.mse(pv, yt),
                            mse_true=float(np.mean((pv - ft) ** 2)),
                            runtime=times.get(name, np.nan),
                            lam_tree=float(np.mean(F.mean_lambda("ssrf_js"))),
                            lam_cv=cal.lambda_, lam_ij=cij.lambda_,
                            lam_oracle=lam_oracle,
                            lam_local_mean=float(np.mean(lam_l))))
                print(f"  calibration {dgp} n={n} B={B} done", flush=True)
        pd.DataFrame(rows).to_csv(
            os.path.join(OUT_DIR, "sim_calibration.csv"), index=False)
    return pd.DataFrame(rows)


def experiment_ij_reliability(reps=25, p=10, n=300,
                              B_list=(10, 50, 200, 1000), seed0=61_000):
    """At which ensemble sizes may the one-sample variance estimate be trusted?"""
    def fit_forest(Xa, ya, B, r):
        return ShrunkenForest(n_estimators=B, min_samples_leaf=5,
                              random_state=r).fit(Xa, ya)
    rows = []
    for dgp in DGP_LIST:
        for rec in ij_reliability(fit_forest, DGPS[dgp], n, p,
                                  B_list=B_list, reps=reps, seed0=seed0):
            rows.append(dict(experiment="ij", dgp=dgp, n=n, p=p, **rec))
        print(f"  ij reliability {dgp} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "sim_ij.csv"), index=False)
    return df


# ---------------------------------------------------------------------------
#  Experiment G: where to put the anchor
# ---------------------------------------------------------------------------

def experiment_anchor(reps=20, p=10, n_list=(300, 1200), B_list=(10, 500),
                      n0=5, n_test=1500, seed0=70_000):
    """Ablation over the anchoring depth, with an out-of-bag selection rule.

    Changing the anchoring depth changes only the leaf values, so every
    candidate is scored on the same fitted forest and the whole ablation costs
    one pass over the leaves per candidate.  The selected row uses the
    out-of-bag error to choose among the same candidates, so it pays nothing
    extra either and is available in practice, which the oracle row is not.
    """
    cands = [("tree", 0), ("ancestor", 1), ("ancestor", 2), ("ancestor", 3)]
    rows = []
    for dgp in DGP_LIST:
        gen = DGPS[dgp]
        for n in n_list:
            for B in B_list:
                for r in range(reps):
                    rng = np.random.default_rng(seed0 + 617 * r + n + B)
                    X, y, f = gen(n, p, rng)
                    Xt, yt, ft = gen(n_test, p, rng)
                    F = ShrunkenForest(n_estimators=B, min_samples_leaf=n0,
                                       random_state=seed0 + r).fit(X, y)
                    base = dict(experiment="anchor", dgp=dgp, n=n, p=p, B=B,
                                min_leaf=n0, rep=r)
                    rows.append(dict(base, method="RF",
                                     mse=E.mse(F.predict(Xt, "rf"), yt),
                                     mse_true=float(np.mean(
                                         (F.predict(Xt, "rf") - ft) ** 2))))
                    scores = {}
                    for (kind, depth) in cands:
                        F.anchor, F.anchor_depth = kind, depth
                        F._apply_shrinkage()
                        pv = F.predict(Xt, "ssrf_js")
                        nm = "tree" if kind == "tree" else f"depth {depth}"
                        scores[nm] = float(np.mean((pv - ft) ** 2))
                        rows.append(dict(base, method=nm, mse=E.mse(pv, yt),
                                         mse_true=scores[nm]))
                    sel = F.select_anchor(X, y)
                    pv = F.predict(Xt, "ssrf_js")
                    rows.append(dict(base, method="selected",
                                     mse=E.mse(pv, yt),
                                     mse_true=float(np.mean((pv - ft) ** 2)),
                                     chosen=("tree" if sel[0] == "tree"
                                             else f"depth {sel[1]}")))
                    rows.append(dict(base, method="oracle depth",
                                     mse=np.nan,
                                     mse_true=min(scores.values())))
                print(f"  anchor {dgp} n={n} B={B} done", flush=True)
        pd.DataFrame(rows).to_csv(
            os.path.join(OUT_DIR, "sim_anchor.csv"), index=False)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
#  Experiment E: the best single contraction factor, as a function of B
# ---------------------------------------------------------------------------

def experiment_level(reps=15, p=10, n_list=(200, 600, 1200),
                     B_list=(5, 25, 100, 500), n0=5, n_test=2000,
                     seed0=50_000,
                     lam_grid=np.linspace(0.0, 1.4, 141)):
    """The factor that minimises forest risk, measured directly.

    The aggregation result says the optimal contraction of a forest prediction
    toward the training mean depends on the variance of that prediction, which
    the ensemble size controls, and that the optimal factor can exceed one when
    the forest's own target is already contracted relative to the regression
    function.  This experiment measures that factor by a grid search against
    the true conditional mean, alongside the factor the tree-level rule
    produces, so that the two can be compared rather than assumed equal.
    """
    rows = []
    for dgp in DGP_LIST:
        gen = DGPS[dgp]
        for n in n_list:
            acc = {B: np.zeros(lam_grid.size) for B in B_list}
            lam_tree = {B: [] for B in B_list}
            for r in range(reps):
                rng = np.random.default_rng(seed0 + 31 * r + n)
                X, y, f = gen(n, p, rng)
                Xt, yt, ft = gen(n_test, p, rng)
                big = ShrunkenForest(n_estimators=max(B_list),
                                     min_samples_leaf=n0,
                                     random_state=seed0 + r).fit(X, y)
                allt = big.trees_
                theta = float(y.mean())
                for B in B_list:
                    big.trees_ = allt[:B]
                    pr = big.predict(Xt, "rf")
                    for i, lam in enumerate(lam_grid):
                        acc[B][i] += float(np.mean(
                            (theta + lam * (pr - theta) - ft) ** 2))
                    lam_tree[B].append(float(np.mean(
                        big.mean_lambda("ssrf_js"))))
                big.trees_ = allt
            for B in B_list:
                curve = acc[B] / reps
                rows.append(dict(experiment="level", dgp=dgp, n=n, p=p, B=B,
                                 min_leaf=n0, reps=reps,
                                 lam_best=float(lam_grid[int(np.argmin(curve))]),
                                 risk_best=float(curve.min()),
                                 risk_one=float(curve[int(np.argmin(
                                     np.abs(lam_grid - 1.0)))]),
                                 lam_tree=float(np.mean(lam_tree[B]))))
            print(f"  level {dgp} n={n} done", flush=True)
        pd.DataFrame(rows).to_csv(
            os.path.join(OUT_DIR, "sim_level.csv"), index=False)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
#  Experiment D: how many extra trees does shrinkage buy?
# ---------------------------------------------------------------------------

def experiment_equivalence(reps, n=600, p=10, n0=5,
                           B_shrunk=(5, 10, 25, 50),
                           B_ref=(5, 10, 15, 25, 40, 60, 100, 150, 250, 400, 600),
                           seed0=40_000):
    """Risk of the shrunken forest at B trees against the RF risk curve in B.

    The tree-equivalence factor reported by the aggregation script is the
    ratio B_eq / B, where B_eq is the number of unshrunken trees whose risk
    matches that of B shrunken ones, obtained by interpolating the RF risk
    curve.  Both curves come from the same fitted forest, truncated to each
    size, so the comparison is paired.
    """
    rows = []
    for dgp in DGP_LIST:
        gen = DGPS[dgp]
        for r in range(reps):
            rng = np.random.default_rng(seed0 + 15_485_863 % (r + 3) + r)
            X, y, f = gen(n, p, rng)
            Xt, yt, ft = gen(1500, p, rng)
            big = ShrunkenForest(n_estimators=max(B_ref), min_samples_leaf=n0,
                                 random_state=seed0 + r).fit(X, y)
            allt = big.trees_
            for B in sorted(set(B_ref) | set(B_shrunk)):
                big.trees_ = allt[:B]
                for name, rule in (("RF", "rf"), ("SSRF-JS", "ssrf_js")):
                    pr = big.predict(Xt, rule)
                    rows.append(dict(experiment="equivalence", dgp=dgp, n=n,
                                     p=p, B=B, min_leaf=n0, rep=r,
                                     method=name, mse=E.mse(pr, yt),
                                     mse_true=float(np.mean((pr - ft) ** 2))))
            big.trees_ = allt
        print(f"  equivalence {dgp} done", flush=True)
        pd.DataFrame(rows).to_csv(
            os.path.join(OUT_DIR, "sim_equivalence.csv"), index=False)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------

EXPERIMENTS = {"substitution": experiment_substitution,
               "grid": experiment_grid,
               "extended": experiment_extended,
               "equivalence": experiment_equivalence,
               "level": experiment_level,
               "calibration": experiment_calibration,
               "ij": experiment_ij_reliability,
               "anchor": experiment_anchor,
               "heterogeneity": experiment_heterogeneity}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True, choices=sorted(EXPERIMENTS))
    ap.add_argument("--reps", type=int, default=40)
    a = ap.parse_args()
    t0 = time.perf_counter()
    df = EXPERIMENTS[a.experiment](a.reps)
    path = os.path.join(OUT_DIR, f"sim_{a.experiment}.csv")
    df.to_csv(path, index=False)
    print(f"{a.experiment}: {len(df)} rows in "
          f"{time.perf_counter()-t0:.0f}s -> {path}")
