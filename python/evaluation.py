"""
evaluation.py
=============

Accuracy measures, interval scores and tests of equal predictive accuracy.

Three points of method deserve comment, because each was a weakness of the
first version of the numerical study.

1.  Standard error of a median.  The asymptotic formula 1.253 s / sqrt(R) is
    valid for a normal sampling distribution and is badly behaved when the
    replication-level losses are heavy-tailed, which is exactly the regime
    the simulations are designed to sit in.  It produced standard errors
    larger than the medians themselves.  The nonparametric bootstrap of the
    median used here has no such failure mode.

2.  Diebold-Mariano tests.  The plain statistic is oversized in small
    samples, so the Harvey, Leybourne and Newbold correction is applied and
    the reference distribution is Student's t with T - 1 degrees of freedom.
    A Newey-West long-run variance with the usual h - 1 truncation is used
    for multi-step losses; for one-step-ahead forecasts the truncation is
    zero and the estimator reduces to the sample variance.

3.  Prediction intervals.  Out-of-bag residual quantiles borrowed from one
    method and applied to all of them are not calibrated for any of them.
    Split conformal intervals are calibrated per method, have finite-sample
    coverage guarantees under exchangeability, and are scored here by the
    interval score of Gneiting and Raftery, which rewards narrow intervals
    and penalises misses.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = ["mse", "mae", "bootstrap_median_se", "dm_test", "holm",
           "split_conformal", "interval_score", "wilcoxon_win",
           "directional_accuracy", "net_pnl"]


def mse(pred, y):
    return float(np.mean((np.asarray(pred) - np.asarray(y)) ** 2))


def mae(pred, y):
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(y))))


def directional_accuracy(pred, y):
    p, t = np.sign(np.asarray(pred)), np.sign(np.asarray(y))
    return float(np.mean(p == t) * 100.0)


# ---------------------------------------------------------------------------

def bootstrap_median_se(x, n_boot=2000, seed=0):
    """Nonparametric bootstrap standard error of the median."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    return float(np.std(np.median(x[idx], axis=1), ddof=1))


# ---------------------------------------------------------------------------

def dm_test(loss_a, loss_b, h=1, two_sided=True):
    """Diebold-Mariano test of equal predictive accuracy, HLN-corrected.

    The loss differential is d_t = loss_a - loss_b, so a positive mean
    favours model b.  Returns ``(statistic, p_value)``.  The statistic is
    referred to a t distribution on T - 1 degrees of freedom after the
    Harvey-Leybourne-Newbold small-sample correction

        HLN = sqrt( [T + 1 - 2h + h(h-1)/T] / T ) .
    """
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    d = d[np.isfinite(d)]
    T = d.size
    if T < 8:
        return np.nan, np.nan
    dbar = d.mean()
    dc = d - dbar
    gamma0 = float(np.dot(dc, dc) / T)
    lrv = gamma0
    for k in range(1, h):
        gk = float(np.dot(dc[k:], dc[:-k]) / T)
        lrv += 2.0 * (1.0 - k / h) * gk
    if lrv <= 0:
        return np.nan, np.nan
    stat = dbar / np.sqrt(lrv / T)
    hln = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    stat *= hln
    p = 2 * stats.t.sf(abs(stat), df=T - 1) if two_sided else stats.t.sf(stat, df=T - 1)
    return float(stat), float(p)


def holm(pvals):
    """Holm step-down adjustment, returned in the input order."""
    p = np.asarray(pvals, float)
    ok = np.isfinite(p)
    out = np.full(p.shape, np.nan)
    q = p[ok]
    m = q.size
    if m == 0:
        return out
    order = np.argsort(q)
    adj = np.empty(m)
    running = 0.0
    for rank, j in enumerate(order):
        val = (m - rank) * q[j]
        running = max(running, val)
        adj[j] = min(running, 1.0)
    out[ok] = adj
    return out


def wilcoxon_win(diff):
    """One-sided Wilcoxon signed-rank test that ``diff`` is centred above zero.

    Applied across replications to the per-replication loss difference
    (competitor minus SSRF), this is the appropriate replication-level test.
    Averaging or taking medians of per-replication p-values, as the first
    version did, has no interpretation as a test.
    """
    d = np.asarray(diff, float)
    d = d[np.isfinite(d) & (d != 0)]
    if d.size < 6:
        return np.nan, np.nan, np.nan
    res = stats.wilcoxon(d, alternative="greater")
    return float(res.statistic), float(res.pvalue), float(np.mean(d > 0))


# ---------------------------------------------------------------------------

def split_conformal(pred_cal, y_cal, pred_test, alpha=0.05):
    """Split conformal prediction intervals with absolute-residual scores.

    Returns ``(lower, upper)``.  Coverage is at least 1 - alpha in finite
    samples when the calibration and test points are exchangeable.
    """
    r = np.abs(np.asarray(y_cal, float) - np.asarray(pred_cal, float))
    n = r.size
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        q = float(np.max(r))
    else:
        q = float(np.sort(r)[k - 1])
    return pred_test - q, pred_test + q


def interval_score(lower, upper, y, alpha=0.05):
    """Mean interval score of Gneiting and Raftery (negatively oriented).

        IS = (u - l) + (2/alpha)(l - y) 1{y < l} + (2/alpha)(y - u) 1{y > u}
    """
    lower, upper, y = map(lambda a: np.asarray(a, float), (lower, upper, y))
    width = upper - lower
    below = np.maximum(lower - y, 0.0)
    above = np.maximum(y - upper, 0.0)
    return float(np.mean(width + (2.0 / alpha) * (below + above)))


# ---------------------------------------------------------------------------

def net_pnl(pred, y, cost=0.02):
    """Sign-based strategy profit and loss, net of round-trip trading costs.

    A position flip costs ``cost``, in the units of ``y``, charged on the day
    of the flip.  The targets used here are daily log returns in percent, so
    two basis points is ``0.02``; passing ``2`` would charge two percent.
    Charging the cost once and then dividing it by the length of the sample,
    as the first version did, made the strategy effectively costless and
    overstated the net result.  Returns the cumulative net series and the
    annualised Sharpe ratio computed on the net daily series.
    """
    pos = np.sign(np.asarray(pred, float))
    pos[pos == 0] = 1.0
    y = np.asarray(y, float)
    gross = pos * y
    turn = np.abs(np.diff(pos, prepend=pos[0])) / 2.0     # 1 on a flip, else 0
    net = gross - cost * turn
    sd = net.std(ddof=1)
    sharpe = float(net.mean() / sd * np.sqrt(252.0)) if sd > 0 else np.nan
    return np.cumsum(net), sharpe, float(net.mean()), float(turn.sum())
