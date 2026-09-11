"""
calibration.py
==============

Estimating the contraction factor at the level at which it acts.

The aggregation result says the factor that minimises forest risk at a point
is

    lambda*(x) = Cov(m, f) / ( Var(m) + V_B(x) ) ,

with m the expectation of the forest prediction, f the regression function and
V_B(x) the sampling variance of the forest prediction at x.  A weight fitted
inside a tree estimates the same ratio with the variance of one leaf mean in
place of V_B(x), which is the B = 1 case, and the aggregation result charges
the difference quadratically.  This module estimates lambda* directly.

Two estimators are provided.

``method="cv"``  (the default)
    The sample is split into K folds; each fold is predicted by a forest of
    the same size grown on the other folds, and the factor is the slope of the
    regression of the response on those predictions.  The slope of the
    response on an out-of-sample prediction estimates Cov(m, f) / Var(fhat),
    which is lambda* exactly, because the noise in the response is
    uncorrelated with a prediction that did not see it.  No variance model
    enters, and the estimator works at any ensemble size.  It costs K forest
    fits, which at the small ensembles where the correction matters is a small
    absolute cost.  The forests used for calibration see (K-1)/K of the data,
    so their predictions are slightly noisier than the deployed forest's and
    the factor is slightly conservative.

``method="ij"``
    One forest, with the out-of-bag predictions supplying the slope and the
    infinitesimal jackknife of Wager, Hastie and Efron supplying the
    variances.  An out-of-bag average uses only the trees that excluded the
    point, so its variance is not V_B; writing s^2(x) for the across-tree
    variance of the per-tree predictions,

        V_oob(x) = V_B(x) - s^2(x)/B + s^2(x)/B_oob(x) ,

    and transporting the slope to the deployed ensemble size uses that
    identity.  This estimator is cheap but it inherits the infinitesimal
    jackknife's requirement that B be large relative to n.  At B = 10 with
    n = 300 the corrected jackknife overstates the true prediction variance
    about fivefold and correlates with it at about 0.44; by B = 1000 the two
    agree to within a quarter and correlate at about 0.90.  The
    ``ij_reliability`` helper reproduces that diagnostic.  Use this route only
    when B is at least of the order of n, which is the regime in which the
    correction has least to offer, so in practice the cross-validated route is
    the one to use.

A pointwise mode is also implemented, which varies the factor with the local
prediction variance.  It is reported in the manuscript because the theory
makes lambda* a function of x, but it does not pay off: the pointwise variance
estimate is noisy enough that multiplying the prediction by a noisy factor
adds more variance than the localisation removes.  Shrinking the pointwise
variance toward its mean reduces but does not remove the penalty.
"""

from __future__ import annotations

import numpy as np

__all__ = ["per_tree_predictions", "ij_variance", "ForestCalibrator",
           "calibrate_by_cv", "ij_reliability"]

_EPS = 1e-12


# ---------------------------------------------------------------------------
#  Ingredients
# ---------------------------------------------------------------------------

def per_tree_predictions(forest, X, rule="rf"):
    """(n, B) matrix of the value each tree reports at each row."""
    X = np.ascontiguousarray(X, dtype=float)
    out = np.empty((X.shape[0], len(forest.trees_)), dtype=float)
    for b, rec in enumerate(forest.trees_):
        node = rec.tree.apply(X)
        j = np.clip(np.searchsorted(rec.leaf_ids, node), 0,
                    rec.leaf_ids.size - 1)
        vals = rec.values[rule][j].astype(float, copy=True)
        miss = rec.leaf_ids[j] != node
        if miss.any():
            vals[miss] = (rec.theta if np.isfinite(rec.theta)
                          else forest.y_mean_)
        out[:, b] = vals
    return out


def ij_variance(forest, X, per_tree=None):
    """Bias-corrected infinitesimal-jackknife variance of the forest mean.

    With N_{b,i} the number of times observation i enters the sample of tree b
    and t_b(x) the value tree b reports at x,

        V_IJ(x) = sum_i [ B^{-1} sum_b (N_{b,i} - Nbar_i)(t_b(x) - tbar(x)) ]^2 ,

    whose Monte-Carlo bias from using finitely many trees is n s^2(x) / B with
    s^2(x) the across-tree variance.  Subtracting that term gives the corrected
    estimator, truncated below because the correction can overshoot.

    Returns ``(V, per_tree, s2)``.
    """
    T = per_tree_predictions(forest, X) if per_tree is None else per_tree
    B = T.shape[1]
    Tc = T - T.mean(axis=1, keepdims=True)
    N = forest.inbag_.astype(float)                     # (B, n)
    Nc = N - N.mean(axis=0, keepdims=True)
    cov = (Tc @ Nc) / B
    raw = (cov ** 2).sum(axis=1)
    s2 = (Tc ** 2).sum(axis=1) / B
    n = N.shape[1]
    return np.maximum(raw - n * s2 / B, s2 / B * 1e-3), T, s2


def ij_reliability(fit_forest, gen, n, p, B_list=(10, 50, 200, 1000),
                   n_test=60, reps=30, seed0=5000):
    """Compare the jackknife variance with the truth across ensemble sizes.

    ``gen(n, p, rng)`` must return ``(X, y, f)``.  The true prediction variance
    is obtained by refitting on independent training samples, which is only
    possible in a simulation; the point of the diagnostic is to say at which
    ensemble sizes the one-sample estimate may be trusted.
    """
    rng0 = np.random.default_rng(seed0 - 1)
    Xt, _, _ = gen(n_test, p, rng0)
    rows = []
    for B in B_list:
        P, I = [], []
        for r in range(reps):
            rng = np.random.default_rng(seed0 + r)
            X, y, _ = gen(n, p, rng)
            F = fit_forest(X, y, B, r)
            V, T, _ = ij_variance(F, Xt)
            P.append(T.mean(axis=1))
            I.append(V)
        P, I = np.asarray(P), np.asarray(I)
        truth = P.var(axis=0, ddof=1)
        rows.append(dict(B=int(B), truth=float(truth.mean()),
                         ij=float(I.mean()),
                         ratio=float(I.mean() / max(truth.mean(), _EPS)),
                         corr=float(np.corrcoef(truth, I.mean(axis=0))[0, 1])))
    return rows


# ---------------------------------------------------------------------------
#  The calibrator
# ---------------------------------------------------------------------------

def calibrate_by_cv(fit_forest, X, y, n_folds=5, random_state=0):
    """Cross-validated contraction factor and anchor.

    ``fit_forest(X_tr, y_tr, seed)`` must return a fitted forest of the size
    that will be deployed.  Returns ``(lambda_hat, theta_hat)``.
    """
    X = np.ascontiguousarray(X, dtype=float)
    y = np.ascontiguousarray(y, dtype=float).ravel()
    n = X.shape[0]
    rng = np.random.default_rng(random_state)
    fold = rng.permutation(n) % n_folds
    pred = np.empty(n)
    for k in range(n_folds):
        tr, te = fold != k, fold == k
        if te.sum() == 0:
            continue
        if tr.sum() < 10:
            pred[te] = y[tr].mean() if tr.sum() else y.mean()
            continue
        F = fit_forest(X[tr], y[tr], int(rng.integers(0, 2 ** 31 - 1)))
        pred[te] = per_tree_predictions(F, X[te]).mean(axis=1)
    theta = float(y.mean())
    v = float(np.var(pred, ddof=1))
    if v <= _EPS:
        return 0.0, theta
    return float(np.cov(pred, y, ddof=1)[0, 1] / v), theta


class ForestCalibrator:
    """Forest-level contraction factor, estimated from the training sample.

    Parameters
    ----------
    method : {"cv", "ij"}
    n_folds : int
        Folds for the cross-validated route.
    clip : (float, float)
        Range the factor is confined to.  Values above one are permitted and
        are meaningful: they are the regime in which the forest is already
        contracted relative to the regression function, so the best linear
        correction is an expansion.
    positive_part : bool
        Cap the factor at one, which makes the rule a contraction and so
        directly comparable with the leaf-level rules.
    local_shrink : float
        Weight placed on the pointwise variance when the pointwise mode is
        used; zero recovers the global factor and one uses the raw pointwise
        estimate.
    """

    def __init__(self, method="cv", n_folds=5, clip=(0.0, 2.0),
                 positive_part=False, local_shrink=0.5, random_state=0):
        self.method = method
        self.n_folds = int(n_folds)
        self.clip = tuple(clip)
        self.positive_part = bool(positive_part)
        self.local_shrink = float(local_shrink)
        self.random_state = random_state

    def fit(self, forest, X, y, fit_forest=None):
        X = np.ascontiguousarray(X, dtype=float)
        y = np.ascontiguousarray(y, dtype=float).ravel()
        self.B_ = len(forest.trees_)

        if self.method == "cv":
            if fit_forest is None:
                raise ValueError('method="cv" needs a fit_forest callable')
            lam, theta = calibrate_by_cv(fit_forest, X, y, self.n_folds,
                                         self.random_state)
            self.theta_ = theta
            self.lambda_ = float(self._clip(np.array([lam]))[0])
            V, _, _ = ij_variance(forest, X)
            self.V_mean_ = float(np.mean(V))
            # the pointwise mode needs a signal scale on the same footing as
            # the variance; invert the global factor for it
            self.signal_ = (self.lambda_ * self.V_mean_
                            / max(1.0 - self.lambda_, 1e-3)
                            if self.lambda_ < 1.0 else np.inf)
            return self

        if self.method != "ij":
            raise ValueError(f"unknown method: {self.method!r}")

        T = per_tree_predictions(forest, X)
        V_B, _, s2 = ij_variance(forest, X, per_tree=T)
        oob = (forest.inbag_.T == 0)
        n_oob = oob.sum(axis=1).astype(float)
        use = n_oob >= max(3, 0.05 * self.B_)
        self.theta_ = float(y.mean())
        self.V_mean_ = float(np.mean(V_B))
        if use.sum() < 10:
            self.lambda_, self.signal_ = 1.0, np.inf
            return self
        f_oob = np.where(oob, T, 0.0).sum(axis=1) / np.maximum(n_oob, 1.0)
        f_oob, y_u = f_oob[use], y[use]
        var_oob = float(np.var(f_oob, ddof=1))
        if var_oob <= _EPS:
            self.lambda_, self.signal_ = 0.0, 0.0
            return self
        cov_mf = float(np.cov(f_oob, y_u, ddof=1)[0, 1])     # = Cov(m, f)
        V_oob = V_B[use] - s2[use] / self.B_ + s2[use] / n_oob[use]
        var_m = max(var_oob - float(np.mean(V_oob)), 0.0)
        self.signal_ = var_m
        self.lambda_ = float(self._clip(
            np.array([cov_mf / max(var_m + self.V_mean_, _EPS)]))[0])
        self.cov_mf_ = cov_mf
        return self

    def _clip(self, lam):
        lam = np.clip(lam, self.clip[0], self.clip[1])
        return np.minimum(lam, 1.0) if self.positive_part else lam

    def lambda_at(self, forest, X, per_tree=None):
        """Pointwise factor, with the local variance shrunk toward its mean."""
        V, _, _ = ij_variance(forest, X, per_tree=per_tree)
        Vs = self.local_shrink * V + (1.0 - self.local_shrink) * V.mean()
        if not np.isfinite(self.signal_):
            return np.full(V.shape, self.lambda_)
        # anchored so that a point of average variance receives the global
        # factor, and the variation about it follows the theory's dependence
        base = self.signal_ + self.V_mean_
        return self._clip(self.lambda_ * base / np.maximum(
            self.signal_ + Vs, _EPS))

    def predict(self, forest, X, mode="global"):
        T = per_tree_predictions(forest, X)
        pred = T.mean(axis=1)
        if mode == "global":
            lam = np.full(pred.shape, self.lambda_)
        elif mode == "local":
            lam = self.lambda_at(forest, X, per_tree=T)
        else:
            raise ValueError(f"unknown mode: {mode!r}")
        return self.theta_ + lam * (pred - self.theta_), lam
