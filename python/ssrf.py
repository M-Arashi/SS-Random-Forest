"""
ssrf.py
=======

Stein-Shrunken Random Forests (SSRF): reference Python implementation.

A bagged regression-tree ensemble in which the value reported by each leaf is

    * the raw leaf sample mean                       (standard Breiman forest),
    * a heteroskedastic empirical-Bayes shrinkage of the leaf means toward a
      within-tree anchor                             (SSRF-EB),
    * a James-Stein shrinkage of the whitened leaf means toward the same
      anchor                                         (SSRF-JS).

The partition is identical in all three cases, so the predictors differ only
in the value stored at each leaf.

Notation.  Within tree b,

    K_b        number of leaves,
    ybar_{b,l} leaf mean computed on the estimation sample,
    v_{b,l}    sampling variance of that mean,
    theta_b    precision-weighted anchor,
    Q_b        = sum_l (ybar_{b,l} - theta_b)^2 / v_{b,l}, Cochran's statistic,
    tau_b^2    between-leaf variance of the leaf-conditional means,
    lambda     shrinkage weight in [0, 1].

Four implementation points differ from a naive transcription of the estimator
and each of them changes the fitted shrinkage weights materially.

1.  Effective leaf size under bootstrap resampling.  A bootstrap leaf holds
    distinct observations with integer multiplicities c_i, and the leaf mean
    is the weighted average sum_i c_i Y_i / C with C = sum_i c_i.  Its
    variance is sigma^2 sum_i c_i^2 / C^2, not sigma^2 / C.  Because in-bag
    multiplicities follow a Poisson(1) law conditioned to be positive,
    sum c_i^2 / C is close to 2, so the raw count overstates the information
    in a leaf by about a factor of two.  The effective size

        n_eff = C^2 / sum_i c_i^2

    is used throughout.  For subsampling without replacement, and for the
    honest estimation half, all c_i equal one and n_eff = C.

2.  Degrees of freedom of the pooled within-leaf variance.  With
    multiplicities, E[sum_i c_i (Y_i - ybar)^2] = sigma^2 (C - sum c_i^2 / C),
    so the pooled estimator divides by sum_l (C_l - sum_i c_i^2 / C_l).
    Dividing by sum_l (C_l - 1) understates sigma^2 by roughly the same
    factor of two, which compounds with the first point.

3.  Estimation of the between-leaf variance.  Leaf-mean variances inside one
    tree differ by orders of magnitude, and the unweighted moment estimator
    is unbiased but very noisy in that regime, so its positive-part
    truncation fires far too often.  The default here is the
    DerSimonian-Laird estimator, which is the inverse-variance-weighted
    moment estimator for exactly this random-effects model,

        tau^2_DL = ( Q - (K - 1) )_+ / ( sum_l u_l - sum_l u_l^2 / sum_l u_l ),
        u_l = 1 / v_l .

    An unweighted moment estimator is retained as an option, corrected for
    unequal weights: with theta_hat a weighted mean,
    E[S^2] = sum_l c_l (tau^2 + v_l) with c_l = 1 - 2 w_l + K w_l^2, so the
    unbiased version divides by sum_l c_l rather than by K - 1.

4.  Honest estimation.  Each honest tree splits its subsample in two; the
    first part chooses the splits and the second supplies the leaf means, the
    leaf sizes and both variance components.  The within-leaf central limit
    approximation the theory uses then applies to independent summands.

One identity is worth keeping in mind while reading the code.  In a tree
whose leaves have equal variances, the empirical-Bayes weight built from the
DerSimonian-Laird estimator is exactly (1 - (K-1)/Q)_+, while the James-Stein
weight is (1 - (K-3)/Q)_+.  The two rules are the same rule with a different
degrees-of-freedom constant, and the Stein constant is the one that carries a
finite-sample dominance guarantee.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from sklearn.tree import DecisionTreeRegressor

__all__ = [
    "ShrunkenForest",
    "leaf_summaries",
    "pooled_within_leaf_variance",
    "robust_within_leaf_variance",
    "tau2_dersimonian_laird",
    "tau2_moment_unweighted",
    "shrink_leaves_eb",
    "shrink_leaves_js",
]

_EPS = 1e-12


# ---------------------------------------------------------------------------
#  Leaf bookkeeping
# ---------------------------------------------------------------------------

def leaf_summaries(node, y, orig_index=None):
    """Per-leaf counts, effective counts, means and residual sums of squares.

    ``node`` holds the terminal-node identifier of every estimation-sample
    row, ``y`` the responses, and ``orig_index`` the index of each row in the
    original training set, which is what makes repeated bootstrap draws
    recognisable.  When ``orig_index`` is ``None`` every row is treated as a
    distinct observation.

    Returns ``(leaf_ids, inv, counts, n_eff, leaf_mean, rss, df)``.
    """
    leaf_ids, inv = np.unique(node, return_inverse=True)
    K = leaf_ids.size
    counts = np.bincount(inv, minlength=K).astype(float)
    sums = np.bincount(inv, weights=y, minlength=K)
    leaf_mean = sums / np.maximum(counts, 1.0)

    if orig_index is None:
        sum_c2 = counts.copy()                      # every multiplicity is one
    else:
        # count how often each (leaf, original observation) pair occurs
        key = inv.astype(np.int64) * (int(orig_index.max()) + 1) \
            + orig_index.astype(np.int64)
        _, mult = np.unique(key, return_counts=True)
        # map each distinct pair back to its leaf
        pair_leaf = np.unique(key) // (int(orig_index.max()) + 1)
        sum_c2 = np.bincount(pair_leaf.astype(int),
                             weights=mult.astype(float) ** 2,
                             minlength=K)

    n_eff = counts ** 2 / np.maximum(sum_c2, _EPS)
    resid = y - leaf_mean[inv]
    rss = float(np.dot(resid, resid))
    df = float(np.sum(counts - sum_c2 / np.maximum(counts, 1.0)))
    return leaf_ids, inv, counts, n_eff, leaf_mean, rss, df


# ---------------------------------------------------------------------------
#  Within-leaf scale
# ---------------------------------------------------------------------------

def pooled_within_leaf_variance(rss, df):
    """Pooled within-leaf residual variance with multiplicity-aware df."""
    if df <= 0:
        return np.nan
    return float(rss / df)


def robust_within_leaf_variance(y, inv, leaf_mean, df, method="mad",
                                huber_k=1.345, max_iter=60):
    """Robust pooled within-leaf scale, squared.

    Heavy-tailed within-leaf noise inflates the pooled sum of squares through
    a handful of observations, which drives sigma^2 up, lambda down, and the
    shrinkage toward the anchor up.  Two robust alternatives are offered, both
    rescaled to agree with the Gaussian scale so that nothing is given away
    when the tails are light.

    ``mad``    1.4826 times the median absolute within-leaf deviation.
    ``huber``  Huber's proposal-2 M-estimate of scale on the pooled residuals.

    The multiplicity-aware degrees-of-freedom correction N / df compensates
    for centring the residuals at estimated leaf means.
    """
    resid = y - leaf_mean[inv]
    N = float(resid.size)
    corr = N / max(df, 1.0)

    med = np.median(resid)
    s = 1.4826 * np.median(np.abs(resid - med))
    if not np.isfinite(s) or s <= 0:
        return np.nan
    if method == "mad":
        return float(corr * s ** 2)
    if method != "huber":
        raise ValueError(f"unknown robust scale method: {method!r}")

    beta = _huber_beta(huber_k)
    for _ in range(max_iter):
        u = np.clip(resid / s, -huber_k, huber_k)
        s_new = s * float(np.sqrt(np.mean(u ** 2) / beta))
        if not np.isfinite(s_new) or s_new <= 0:
            break
        if abs(s_new - s) <= 1e-8 * s:
            s = s_new
            break
        s = s_new
    return float(corr * s ** 2)


def _huber_beta(k):
    """E[min(Z, k)^2] for Z standard normal truncated by the Huber score."""
    from scipy.stats import norm
    phi_k, Phi_k = float(norm.pdf(k)), float(norm.cdf(k))
    return 2.0 * (Phi_k - 0.5) - 2.0 * k * phi_k + 2.0 * k ** 2 * (1.0 - Phi_k)


# ---------------------------------------------------------------------------
#  Between-leaf variance
# ---------------------------------------------------------------------------

def cochran_q(leaf_mean, v_leaf):
    """Cochran's heterogeneity statistic and the precision-weighted anchor."""
    u = 1.0 / np.maximum(v_leaf, _EPS)
    theta = float(np.dot(u, leaf_mean) / u.sum())
    Q = float(np.dot(u, (leaf_mean - theta) ** 2))
    return Q, theta, u


def tau2_dersimonian_laird(leaf_mean, v_leaf):
    """DerSimonian-Laird moment estimator of the between-leaf variance.

        E[Q] = (K - 1) + tau^2 ( sum u_l - sum u_l^2 / sum u_l ),

    which is verified by expanding Q under the random-effects model, so the
    moment estimator is the positive part of the corresponding solution.
    """
    K = leaf_mean.size
    Q, theta, u = cochran_q(leaf_mean, v_leaf)
    den = float(u.sum() - (u ** 2).sum() / u.sum())
    if den <= _EPS:
        return 0.0, theta, Q
    return max((Q - (K - 1)) / den, 0.0), theta, Q


def tau2_moment_unweighted(leaf_mean, v_leaf, weights):
    """Unweighted moment estimator, corrected for unequal anchor weights."""
    K = leaf_mean.size
    theta = float(np.dot(weights, leaf_mean))
    dev = leaf_mean - theta
    c = 1.0 - 2.0 * weights + K * weights ** 2
    tau2 = (float(np.dot(dev, dev)) - float(np.dot(c, v_leaf))) / max(float(c.sum()), _EPS)
    return max(tau2, 0.0), theta


def tau2_robust(leaf_mean, v_leaf, weights):
    """Median-based estimator, used when leaf means themselves have outliers."""
    theta = float(np.dot(weights, leaf_mean))
    dev = leaf_mean - theta
    mad = 1.4826 * float(np.median(np.abs(dev - np.median(dev))))
    return max(mad ** 2 - float(np.median(v_leaf)), 0.0), theta


# ---------------------------------------------------------------------------
#  Shrinkage rules
# ---------------------------------------------------------------------------

def shrink_leaves_eb(leaf_mean, v_leaf, tau2, theta):
    """Heteroskedastic empirical-Bayes rule.

        mu_hat_l = theta + lambda_l (ybar_l - theta),
        lambda_l = tau^2 / (tau^2 + v_l).
    """
    lam = np.zeros_like(leaf_mean) if tau2 <= 0.0 else tau2 / (tau2 + v_leaf)
    return theta + lam * (leaf_mean - theta), lam


def shrink_leaves_js(leaf_mean, v_leaf, theta, Q):
    """James-Stein rule in the whitened coordinates z_l = ybar_l / sqrt(v_l).

    Writing u_l = v_l^{-1/2}, the hypothesis that all leaf-conditional means
    coincide places the mean vector in span(u).  The orthogonal projection of
    z on that line has l-th coordinate theta / sqrt(v_l) with theta the
    precision-weighted mean, and the squared length of the residual is exactly
    Q.  Shrinking toward the line with the Stein constant K - 3 and
    transforming back gives a single factor shared by all leaves of the tree,

        mu_hat_l = theta + (1 - (K-3)/Q)_+ (ybar_l - theta).

    The rule requires K >= 4.
    """
    K = leaf_mean.size
    if K < 4 or Q <= _EPS:
        return leaf_mean.copy(), np.ones_like(leaf_mean)
    lam = max(1.0 - (K - 3.0) / Q, 0.0)
    return theta + lam * (leaf_mean - theta), np.full(K, lam)


# ---------------------------------------------------------------------------
#  Per-tree record
# ---------------------------------------------------------------------------

@dataclass
class _TreeRecord:
    tree: DecisionTreeRegressor
    leaf_ids: np.ndarray
    leaf_mean: np.ndarray
    leaf_size: np.ndarray          # raw counts
    n_eff: np.ndarray              # effective counts
    v_leaf: np.ndarray
    theta: float
    tau2: float
    sigma2: float
    Q: float
    values: dict = field(default_factory=dict)
    lambdas: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
#  The forest
# ---------------------------------------------------------------------------

class ShrunkenForest:
    """Bagged regression trees with optional Stein shrinkage of leaf values.

    Parameters
    ----------
    n_estimators : int
    max_features : {"third", int, float, None}
    min_samples_leaf : int
    honest : bool
        Split each tree's sample into a structure half and an estimation half.
    bootstrap : bool
        Resample with replacement (Breiman) or subsample without replacement.
    max_samples : float
        Subsample fraction when ``bootstrap`` is False.
    anchor : {"tree", "ancestor"}
        Shrink toward the whole tree's anchor, or toward the anchor of the
        group of leaves sharing an ancestor at ``anchor_depth``.  The second
        option is for partitions whose leaves describe distinct regimes.
    anchor_depth : int
    sigma_method : {"pooled", "mad", "huber"}
    tau_method : {"dl", "mom", "robust"}
    min_leaf_eff : float
        Leaves whose effective size falls below this are still shrunk, but
        their variance is floored so that a single stray observation cannot
        dominate the anchor.
    random_state : int or None
    """

    def __init__(self, n_estimators=500, max_features="third",
                 min_samples_leaf=20, honest=False, bootstrap=True,
                 max_samples=0.5, anchor="tree", anchor_depth=2,
                 sigma_method="pooled", tau_method="dl", min_leaf_eff=1.0,
                 effective_size=True, random_state=None):
        self.n_estimators = int(n_estimators)
        self.max_features = max_features
        self.min_samples_leaf = int(min_samples_leaf)
        self.honest = bool(honest)
        self.bootstrap = bool(bootstrap)
        self.max_samples = float(max_samples)
        self.anchor = anchor
        self.anchor_depth = int(anchor_depth)
        self.sigma_method = sigma_method
        self.tau_method = tau_method
        self.min_leaf_eff = float(min_leaf_eff)
        # effective_size=False reproduces the naive accounting in which a
        # bootstrap leaf is credited with its raw count and the pooled variance
        # is divided by sum_l (C_l - 1).  It is retained only so that the
        # magnitude of that error can be reported, and should not be used.
        self.effective_size = bool(effective_size)
        self.random_state = random_state

    # -- fitting ------------------------------------------------------------

    def fit(self, X, y):
        X = np.ascontiguousarray(X, dtype=float)
        y = np.ascontiguousarray(y, dtype=float).ravel()
        n, p = X.shape
        rng = np.random.default_rng(self.random_state)
        mtry = self._resolve_mtry(p)

        self.y_mean_ = float(y.mean())
        self.n_features_in_ = p
        self.trees_ = []
        inbag = []
        n_sub = n if self.bootstrap else max(4 * self.min_samples_leaf,
                                             int(round(self.max_samples * n)))
        n_sub = min(n_sub, n)

        for _ in range(self.n_estimators):
            seed = int(rng.integers(0, 2 ** 31 - 1))
            if self.bootstrap:
                idx = rng.integers(0, n, size=n_sub)
            else:
                idx = rng.choice(n, size=n_sub, replace=False)

            if self.honest:
                perm = rng.permutation(idx.size)
                half = idx.size // 2
                idx_split, idx_est = idx[perm[:half]], idx[perm[half:]]
            else:
                idx_split = idx_est = idx

            tree = DecisionTreeRegressor(max_features=mtry,
                                         min_samples_leaf=self.min_samples_leaf,
                                         random_state=seed)
            tree.fit(X[idx_split], y[idx_split])
            rec = self._summarise_tree(tree, X[idx_est], y[idx_est], idx_est)
            if rec is not None:
                self.trees_.append(rec)
                inbag.append(np.bincount(idx, minlength=n).astype(np.float32))

        if not self.trees_:
            raise RuntimeError("no usable trees were grown")
        # in-bag multiplicities, needed by the infinitesimal-jackknife variance
        self.inbag_ = np.vstack(inbag)
        self._apply_shrinkage()
        return self

    def _resolve_mtry(self, p):
        if self.max_features == "third":
            return max(1, p // 3)
        if self.max_features is None:
            return p
        if isinstance(self.max_features, float):
            return max(1, int(round(self.max_features * p)))
        return int(self.max_features)

    def _summarise_tree(self, tree, Xe, ye, idx_est):
        node = tree.apply(Xe)
        multiplicities = idx_est if self.bootstrap and not self.honest else None
        (leaf_ids, inv, counts, n_eff,
         leaf_mean, rss, df) = leaf_summaries(node, ye, multiplicities)

        if not self.effective_size:
            n_eff = counts.copy()
            df = float(np.sum(np.maximum(counts - 1.0, 0.0)))
        if self.sigma_method == "pooled":
            sigma2 = pooled_within_leaf_variance(rss, df)
        else:
            sigma2 = robust_within_leaf_variance(ye, inv, leaf_mean, df,
                                                 method=self.sigma_method)
        if not np.isfinite(sigma2) or sigma2 <= 0:
            sigma2 = float(np.var(ye)) if ye.size > 1 else _EPS
        sigma2 = max(sigma2, _EPS)

        v_leaf = sigma2 / np.maximum(n_eff, self.min_leaf_eff)
        return _TreeRecord(tree=tree, leaf_ids=leaf_ids, leaf_mean=leaf_mean,
                           leaf_size=counts, n_eff=n_eff, v_leaf=v_leaf,
                           theta=np.nan, tau2=np.nan, sigma2=sigma2, Q=np.nan)

    def _leaf_groups(self, rec):
        """Split the leaves of one tree into anchoring groups."""
        K = rec.leaf_ids.size
        if self.anchor == "tree":
            return [np.arange(K)]
        if self.anchor != "ancestor":
            raise ValueError(f"unknown anchor: {self.anchor!r}")

        t = rec.tree.tree_
        left, right = t.children_left, t.children_right
        anc, depth, stack = {0: 0}, {0: 0}, [0]
        while stack:
            nid = stack.pop()
            d = depth[nid]
            for child in (left[nid], right[nid]):
                if child == -1:
                    continue
                depth[child] = d + 1
                anc[child] = child if d + 1 <= self.anchor_depth else anc[nid]
                stack.append(child)

        keys = np.array([anc.get(int(nid), 0) for nid in rec.leaf_ids])
        groups = [np.flatnonzero(keys == k) for k in np.unique(keys)]
        big = [g for g in groups if g.size >= 4]
        small = [g for g in groups if g.size < 4]
        if small:
            big.append(np.concatenate(small))
        return big if big else [np.arange(K)]

    def _apply_shrinkage(self):
        for rec in self.trees_:
            raw = rec.leaf_mean
            eb, js = raw.copy(), raw.copy()
            lam_eb, lam_js = np.ones_like(raw), np.ones_like(raw)
            thetas, tau2s, Qs, wts = [], [], [], []

            for g in self._leaf_groups(rec):
                if g.size < 3:
                    continue
                mean_g, v_g = raw[g], rec.v_leaf[g]
                w_g = rec.n_eff[g] / rec.n_eff[g].sum()

                if self.tau_method == "dl":
                    tau2, theta, Q = tau2_dersimonian_laird(mean_g, v_g)
                elif self.tau_method == "mom":
                    tau2, theta = tau2_moment_unweighted(mean_g, v_g, w_g)
                    Q, _, _ = cochran_q(mean_g, v_g)
                elif self.tau_method == "robust":
                    tau2, theta = tau2_robust(mean_g, v_g, w_g)
                    Q, _, _ = cochran_q(mean_g, v_g)
                else:
                    raise ValueError(f"unknown tau_method: {self.tau_method!r}")

                _, theta_p, Q_p = tau2_dersimonian_laird(mean_g, v_g)
                eb[g], lam_eb[g] = shrink_leaves_eb(mean_g, v_g, tau2, theta)
                js[g], lam_js[g] = shrink_leaves_js(mean_g, v_g, theta_p, Q_p)

                thetas.append(theta)
                tau2s.append(tau2)
                Qs.append(Q_p)
                wts.append(rec.n_eff[g].sum())

            if wts:
                wv = np.asarray(wts, float)
                wv /= wv.sum()
                rec.theta = float(np.dot(wv, thetas))
                rec.tau2 = float(np.dot(wv, tau2s))
                rec.Q = float(np.dot(wv, Qs))
            rec.values = {"rf": raw, "ssrf": eb, "ssrf_js": js}
            rec.lambdas = {"rf": np.ones_like(raw), "ssrf": lam_eb,
                           "ssrf_js": lam_js}

    # -- prediction ---------------------------------------------------------

    def predict(self, X, rule="ssrf"):
        X = np.ascontiguousarray(X, dtype=float)
        out = np.zeros(X.shape[0], dtype=float)
        for rec in self.trees_:
            node = rec.tree.apply(X)
            j = np.clip(np.searchsorted(rec.leaf_ids, node), 0,
                        rec.leaf_ids.size - 1)
            miss = rec.leaf_ids[j] != node
            vals = rec.values[rule][j]
            if miss.any():                    # leaf empty in the estimation half
                vals = vals.copy()
                fill = rec.theta if np.isfinite(rec.theta) else self.y_mean_
                vals[miss] = fill
            out += vals
        return out / len(self.trees_)

    def per_tree(self, X, rule="rf"):
        """(n, B) matrix of the value each tree reports at each row."""
        X = np.ascontiguousarray(X, dtype=float)
        out = np.empty((X.shape[0], len(self.trees_)), dtype=float)
        for b, rec in enumerate(self.trees_):
            node = rec.tree.apply(X)
            j = np.clip(np.searchsorted(rec.leaf_ids, node), 0,
                        rec.leaf_ids.size - 1)
            vals = rec.values[rule][j].astype(float, copy=True)
            miss = rec.leaf_ids[j] != node
            if miss.any():
                vals[miss] = (rec.theta if np.isfinite(rec.theta)
                              else self.y_mean_)
            out[:, b] = vals
        return out

    def oob_error(self, X, y, rule="ssrf_js"):
        """Mean squared out-of-bag error under a given leaf rule.

        Each training row is predicted by the trees whose sample excluded it,
        so the score is honest for choices made after the forest was grown,
        such as the anchoring depth.
        """
        T = self.per_tree(X, rule)
        oob = (self.inbag_.T == 0)
        cnt = oob.sum(axis=1)
        use = cnt > 0
        if not use.any():
            return np.nan
        pred = np.where(oob, T, 0.0).sum(axis=1)[use] / cnt[use]
        return float(np.mean((pred - np.asarray(y, float).ravel()[use]) ** 2))

    def select_anchor(self, X, y, candidates=(("tree", 0), ("ancestor", 1),
                                              ("ancestor", 2), ("ancestor", 3)),
                      rule="ssrf_js"):
        """Choose the anchoring depth by out-of-bag error, after growing.

        Shrinking every leaf toward one tree-level anchor is a poor choice when
        the partition separates distinct regimes, and shrinking toward a
        subtree anchor is a poor choice when the subtrees hold too few leaves
        for the between-leaf variance to be estimable.  Which of the two binds
        is a property of the data, so it is selected rather than assumed.  The
        candidates are scored on the same fitted forest, because the anchoring
        depth changes only the leaf values, so no tree is regrown and the whole
        search costs one pass over the leaves per candidate.
        """
        best, best_err = None, np.inf
        for (kind, depth) in candidates:
            self.anchor, self.anchor_depth = kind, int(depth)
            self._apply_shrinkage()
            err = self.oob_error(X, y, rule)
            if np.isfinite(err) and err < best_err:
                best, best_err = (kind, int(depth)), err
        if best is None:
            best = ("tree", 0)
        self.anchor, self.anchor_depth = best
        self._apply_shrinkage()
        self.anchor_selected_ = best
        self.anchor_oob_error_ = float(best_err)
        return best

    # -- diagnostics --------------------------------------------------------

    def mean_lambda(self, rule="ssrf"):
        return np.asarray([float(np.dot(r.n_eff / r.n_eff.sum(), r.lambdas[rule]))
                           for r in self.trees_])

    def leaf_counts(self):
        return np.asarray([r.leaf_ids.size for r in self.trees_], dtype=float)

    def effective_leaf_sizes(self):
        return np.concatenate([r.n_eff for r in self.trees_])

    def variance_components(self):
        return (np.asarray([r.sigma2 for r in self.trees_]),
                np.asarray([r.tau2 for r in self.trees_]))

    def permutation_importance(self, X, y, rule="ssrf", n_repeats=5,
                               random_state=0):
        """Permutation importance computed for the requested leaf rule.

        The importance of a feature is the increase in mean squared error when
        that column is permuted, averaged over repeats.  Computing it
        separately for each rule is the only way to say whether shrinkage
        changes which features matter; carrying the standard forest's
        importances across and rescaling them would answer nothing.
        """
        X = np.asarray(X, float)
        y = np.asarray(y, float).ravel()
        rng = np.random.default_rng(random_state)
        base = float(np.mean((self.predict(X, rule) - y) ** 2))
        imp = np.zeros(X.shape[1])
        for j in range(X.shape[1]):
            acc = 0.0
            for _ in range(n_repeats):
                Xp = X.copy()
                Xp[:, j] = Xp[rng.permutation(X.shape[0]), j]
                acc += float(np.mean((self.predict(Xp, rule) - y) ** 2))
            imp[j] = acc / n_repeats - base
        return imp
