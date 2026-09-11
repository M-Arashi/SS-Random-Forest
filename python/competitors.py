"""
competitors.py
==============

Competing predictors used in the simulation and real-data comparisons.

Every competitor is given the tuning it needs to be seen at its best.  The
forest-based competitors receive the same number of trees and the same
minimum leaf size as SSRF; the penalised and boosted competitors are tuned by
cross-validation on the training sample only.  Nothing here is deliberately
handicapped, because a shrinkage gain that survives only against
under-tuned alternatives is not a gain worth reporting.

Predictors provided
-------------------
ridge_rf         feature rescaling by ridge coefficients, then a forest
lasso_rf         LASSO screening, then a forest on the selected features
gbm              gradient-boosted regression trees
rf_tuned         forest with min_samples_leaf and max_features chosen by CV
local_linear_forest
                 forest kernel weights combined with a local ridge fit, in
                 the sense of Friedberg, Tibshirani, Athey and Wager
bart             Bayesian additive regression trees (via ``bartz``)
ngboost          natural-gradient boosting of a normal predictive
                 distribution, which models the conditional scale explicitly
"""

from __future__ import annotations

import warnings
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.model_selection import KFold

_EPS = 1e-12


# ---------------------------------------------------------------------------

def _householder_basis(v):
    """Orthonormal basis whose first column is v / ||v||."""
    v = np.asarray(v, float).ravel()
    nrm = np.linalg.norm(v)
    p = v.size
    if nrm < _EPS:
        return np.eye(p)
    e1 = np.zeros(p)
    e1[0] = 1.0
    u = v / nrm - e1
    if np.linalg.norm(u) < _EPS:
        return np.eye(p)
    u = u / np.linalg.norm(u)
    H = np.eye(p) - 2.0 * np.outer(u, u)          # H e1 = v/||v||
    return H


def ridge_rf(X_tr, y_tr, X_te, n_estimators=500, min_samples_leaf=20, seed=0):
    """Ridge-rotated forest.

    Rescaling each feature by its ridge coefficient, which is what the phrase
    "ridge-rotated forest" is sometimes taken to mean, leaves a CART forest
    completely unchanged: the impurity reduction of a split depends only on
    the induced partition, and a positive per-feature rescaling is a monotone
    map that preserves every candidate partition.  Such a benchmark is the
    standard forest under another name.

    The forest is therefore grown on a genuine rotation.  A cross-validated
    ridge fit supplies a direction, an orthonormal basis is completed around
    it by a Householder reflection, and the design is expressed in that basis,
    so that the leading coordinate is the ridge direction and splits on it are
    oblique in the original coordinates.
    """
    sd = X_tr.std(axis=0, ddof=1)
    sd[sd < _EPS] = 1.0
    Z_tr, Z_te = X_tr / sd, X_te / sd
    ridge = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(Z_tr, y_tr)
    Hb = _householder_basis(ridge.coef_)
    rf = RandomForestRegressor(n_estimators=n_estimators,
                               max_features=max(1, X_tr.shape[1] // 3),
                               min_samples_leaf=min_samples_leaf,
                               random_state=seed, n_jobs=1)
    rf.fit(Z_tr @ Hb, y_tr)
    return rf.predict(Z_te @ Hb)


def lasso_rf(X_tr, y_tr, X_te, n_estimators=500, min_samples_leaf=20, seed=0):
    """Screen the features by LASSO, then grow a forest on the survivors."""
    sd = X_tr.std(axis=0, ddof=1)
    sd[sd < _EPS] = 1.0
    Z_tr, Z_te = X_tr / sd, X_te / sd
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        las = LassoCV(cv=5, n_alphas=50, max_iter=20000,
                      random_state=seed).fit(Z_tr, y_tr)
    sel = np.flatnonzero(np.abs(las.coef_) > 0)
    if sel.size < 2:                       # fall back to marginal screening
        r = np.abs([np.corrcoef(Z_tr[:, j], y_tr)[0, 1] for j in range(Z_tr.shape[1])])
        r = np.nan_to_num(r)
        sel = np.argsort(r)[::-1][:min(5, Z_tr.shape[1])]
    rf = RandomForestRegressor(n_estimators=n_estimators,
                               max_features=max(1, sel.size // 3),
                               min_samples_leaf=min_samples_leaf,
                               random_state=seed, n_jobs=1)
    rf.fit(Z_tr[:, sel], y_tr)
    return rf.predict(Z_te[:, sel])


def gbm(X_tr, y_tr, X_te, n_estimators=500, seed=0, learning_rate=0.05,
        max_depth=4, subsample=0.7):
    """Gradient-boosted trees with the number of rounds chosen by early stopping.

    Running a fixed large number of boosting rounds in a low signal-to-noise
    problem overfits badly, and reporting that as the boosting benchmark would
    understate what boosting can do.  A tenth of the training sample is held
    out and boosting stops when it stops improving on it.
    """
    g = GradientBoostingRegressor(n_estimators=n_estimators,
                                  learning_rate=learning_rate,
                                  max_depth=max_depth, subsample=subsample,
                                  validation_fraction=0.15,
                                  n_iter_no_change=25, tol=1e-5,
                                  random_state=seed)
    g.fit(X_tr, y_tr)
    return g.predict(X_te)


def rf_tuned(X_tr, y_tr, X_te, n_estimators=500, seed=0, n_folds=3):
    """Forest whose leaf size and split candidate count are chosen by CV.

    This is the regularised-forest benchmark.  Enlarging the minimum leaf
    size is the standard way a practitioner reduces leaf-level estimation
    variance, so tuning it is the fairest available comparison for a method
    whose stated purpose is to reduce that same variance.
    """
    p = X_tr.shape[1]
    grid = [(m, f) for m in (2, 5, 10, 20, 40, 80)
            for f in sorted({max(1, p // 3), max(1, p // 2)})]
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    best, best_mse = grid[0], np.inf
    n_cv = 40
    for (m, f) in grid:
        err = 0.0
        for tr, va in kf.split(X_tr):
            rf = RandomForestRegressor(n_estimators=n_cv,
                                       max_features=f, min_samples_leaf=m,
                                       random_state=seed, n_jobs=1)
            rf.fit(X_tr[tr], y_tr[tr])
            err += np.mean((rf.predict(X_tr[va]) - y_tr[va]) ** 2)
        if err < best_mse:
            best_mse, best = err, (m, f)
    rf = RandomForestRegressor(n_estimators=n_estimators, max_features=best[1],
                               min_samples_leaf=best[0], random_state=seed,
                               n_jobs=1)
    rf.fit(X_tr, y_tr)
    return rf.predict(X_te), best


# ---------------------------------------------------------------------------

class LocalLinearForest:
    """Forest-weighted local ridge regression.

    A regression forest defines a data-adaptive kernel

        alpha_i(x) = B^{-1} sum_b 1{X_i in L_b(x)} / |L_b(x)| .

    The local linear forest replaces the weighted average sum_i alpha_i(x) Y_i
    by the intercept of the weighted ridge fit

        min_{a, beta} sum_i alpha_i(x) (Y_i - a - (X_i - x)' beta)^2
                      + lambda ||beta||^2 ,

    so the leaf value is corrected for the local slope rather than shrunk
    toward a within-tree anchor.  It is the closest competitor in spirit to
    SSRF, because both modify what a leaf reports without changing where the
    splits fall.  The ridge penalty is chosen from a grid by validation on a
    held-out fifth of the training sample.
    """

    def __init__(self, n_estimators=500, min_samples_leaf=20,
                 max_features=None, lambdas=(0.01, 0.1, 1.0, 10.0),
                 random_state=0):
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.lambdas = tuple(lambdas)
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float).ravel()
        n, p = X.shape
        mtry = self.max_features or max(1, p // 3)
        rng = np.random.default_rng(self.random_state)
        val = rng.random(n) < 0.2
        if val.sum() < 10 or (~val).sum() < 20:
            val = np.zeros(n, dtype=bool)

        self.rf_ = RandomForestRegressor(
            n_estimators=self.n_estimators, max_features=mtry,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state, n_jobs=1)
        self.rf_.fit(X[~val] if val.any() else X, y[~val] if val.any() else y)
        self.X_ = X[~val] if val.any() else X
        self.y_ = y[~val] if val.any() else y
        self.scale_ = self.X_.std(axis=0, ddof=1)
        self.scale_[self.scale_ < _EPS] = 1.0
        self.leaves_ = self.rf_.apply(self.X_)

        if val.any():
            errs = []
            for lam in self.lambdas:
                pred = self._predict(X[val], lam)
                errs.append(np.mean((pred - y[val]) ** 2))
            self.lambda_ = self.lambdas[int(np.argmin(errs))]
        else:
            self.lambda_ = self.lambdas[len(self.lambdas) // 2]
        return self

    def _weights(self, X_new):
        leaves_new = self.rf_.apply(X_new)
        n_tr = self.X_.shape[0]
        W = np.zeros((X_new.shape[0], n_tr))
        for b in range(leaves_new.shape[1]):
            col_tr = self.leaves_[:, b]
            uniq, inv = np.unique(col_tr, return_inverse=True)
            counts = np.bincount(inv, minlength=uniq.size).astype(float)
            pos = np.searchsorted(uniq, leaves_new[:, b])
            pos = np.clip(pos, 0, uniq.size - 1)
            ok = uniq[pos] == leaves_new[:, b]
            for i in np.flatnonzero(ok):
                mask = inv == pos[i]
                W[i, mask] += 1.0 / counts[pos[i]]
        return W / leaves_new.shape[1]

    def _predict(self, X_new, lam):
        X_new = np.asarray(X_new, float)
        W = self._weights(X_new)
        p = self.X_.shape[1]
        Zs = self.X_ / self.scale_
        out = np.empty(X_new.shape[0])
        pen = lam * np.eye(p + 1)
        pen[0, 0] = 0.0                     # the intercept is left unpenalised
        for i in range(X_new.shape[0]):
            w = W[i]
            nz = np.flatnonzero(w > 0)
            if nz.size < p + 2:
                out[i] = float(np.dot(w, self.y_) / max(w.sum(), _EPS)) \
                    if w.sum() > 0 else float(self.y_.mean())
                continue
            D = np.column_stack([np.ones(nz.size),
                                 Zs[nz] - X_new[i] / self.scale_])
            sw = w[nz]
            A = D.T @ (D * sw[:, None]) + pen
            bvec = D.T @ (sw * self.y_[nz])
            try:
                out[i] = float(np.linalg.solve(A, bvec)[0])
            except np.linalg.LinAlgError:
                out[i] = float(np.dot(sw, self.y_[nz]) / sw.sum())
        return out

    def predict(self, X_new):
        return self._predict(np.asarray(X_new, float), self.lambda_)


# ---------------------------------------------------------------------------

def bart(X_tr, y_tr, X_te, n_trees=None, n_draws=1000, n_burn=1000, seed=0,
         tree_grid=(25, 50, 100, 200)):
    """Bayesian additive regression trees, with the ensemble size validated.

    BART shrinks each tree's contribution through the prior on leaf values, so
    it is the natural Bayesian counterpart to the present method.  Its default
    of two hundred trees is calibrated to problems with more signal than the
    ones studied here; at a low signal-to-noise ratio a smaller sum-of-trees
    model is markedly more accurate, and reporting the default alone would
    understate what BART can do.  The number of trees is therefore chosen on a
    held-out fifth of the training sample with short chains, and the selected
    value is refitted on the whole training sample with the full chain.  Pass
    an explicit ``n_trees`` to skip the selection.

    Returns ``None`` when the backend is not installed, so callers can drop
    the column rather than fail.
    """
    try:
        from bartz.BART import gbart
    except Exception:
        return None
    import contextlib
    import io

    X_tr = np.asarray(X_tr, float)
    y_tr = np.asarray(y_tr, float)
    X_te = np.asarray(X_te, float)

    def _fit(xa, ya, xb, ntree, ndpost, nskip, sd):
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf), contextlib.redirect_stdout(buf):
            f = gbart(x_train=xa, y_train=ya, x_test=xb, ntree=ntree,
                      ndpost=ndpost, nskip=nskip, seed=sd)
            return np.asarray(f.yhat_test_mean, dtype=float)

    try:
        if n_trees is None:
            rng = np.random.default_rng(seed)
            n = X_tr.shape[0]
            val = rng.random(n) < 0.2
            if val.sum() >= 10 and (~val).sum() >= 30:
                errs = []
                for nt in tree_grid:
                    pr = _fit(X_tr[~val], y_tr[~val], X_tr[val], nt,
                              250, 250, seed)
                    errs.append(float(np.mean((pr - y_tr[val]) ** 2))
                                if pr is not None else np.inf)
                n_trees = tree_grid[int(np.argmin(errs))]
            else:
                n_trees = tree_grid[len(tree_grid) // 2]
        yhat = _fit(X_tr, y_tr, X_te, n_trees, n_draws, n_burn, seed)
        return yhat if yhat.shape[0] == X_te.shape[0] else None
    except Exception:
        return None


def ngboost(X_tr, y_tr, X_te, n_estimators=1000, learning_rate=0.03, seed=0):
    """Natural-gradient boosting of a normal predictive distribution.

    NGBoost changes the objective rather than the leaf estimator and returns a
    full conditional distribution with a fitted scale, which makes it the
    strongest available comparison for a method whose gain comes from
    modelling variance.  Running a fixed number of boosting rounds overfits
    badly at a low signal-to-noise ratio, so a fifth of the training sample is
    held out and boosting stops when it stops improving on it.  Returns
    ``(mean, scale)`` or ``None``.
    """
    try:
        from ngboost import NGBRegressor
        from ngboost.distns import Normal
    except Exception:
        return None
    try:
        X_tr = np.asarray(X_tr, float)
        y_tr = np.asarray(y_tr, float)
        rng = np.random.default_rng(seed)
        val = rng.random(X_tr.shape[0]) < 0.2
        if val.sum() < 10 or (~val).sum() < 30:
            val = np.zeros(X_tr.shape[0], dtype=bool)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = NGBRegressor(Dist=Normal, n_estimators=n_estimators,
                             learning_rate=learning_rate, verbose=False,
                             random_state=seed)
            if val.any():
                m.fit(X_tr[~val], y_tr[~val], X_val=X_tr[val],
                      Y_val=y_tr[val], early_stopping_rounds=50)
            else:
                m.fit(X_tr, y_tr)
            d = m.pred_dist(np.asarray(X_te, float))
        return np.asarray(d.loc, float), np.asarray(d.scale, float)
    except Exception:
        return None
