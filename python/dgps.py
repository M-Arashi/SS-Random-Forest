"""
dgps.py
=======

Data-generating processes for the simulation study.

All three processes are deliberately low signal-to-noise, in the range that
daily commodity and currency returns occupy, because that is the regime in
which leaf-level estimation variance dominates and shrinkage can matter.
The realised signal-to-noise ratio is returned alongside the sample so that
it can be reported rather than asserted.

DGP 1  smooth weak mean, autoregressive Gaussian features, conditional
       variance increasing in x1, Student t_5 innovations.
DGP 2  sparse weak mean on three uniform coordinates, variance modulated by
       a fourth, and highly collinear Gaussian noise coordinates.
DGP 3  the heavy-tail stress case.  Student t_{2.5} innovations, so the
       fourth moment does not exist and the within-leaf central limit
       approximation is at its least accurate.  This process exists to test
       the robustness claims rather than to flatter the method.
"""

from __future__ import annotations

import numpy as np

__all__ = ["dgp1", "dgp2", "dgp3", "DGPS", "signal_to_noise"]


def _ar1_gaussian(rng, n, p, rho=0.6):
    """Gaussian design with correlation rho^{|j-k|}, drawn by an AR recursion."""
    X = np.empty((n, p))
    X[:, 0] = rng.standard_normal(n)
    s = np.sqrt(1.0 - rho ** 2)
    for j in range(1, p):
        X[:, j] = rho * X[:, j - 1] + s * rng.standard_normal(n)
    return X


def _unit_t(rng, n, df):
    """Student t scaled to unit variance (requires df > 2)."""
    return rng.standard_t(df, size=n) / np.sqrt(df / (df - 2.0))


def dgp1(n, p, rng, noise_scale=0.9):
    X = _ar1_gaussian(rng, n, p, rho=0.6)
    f = 0.5 * np.tanh(X[:, 0]) + 0.3 * X[:, 1] * X[:, 2] / (1.0 + X[:, 3] ** 2)
    if p >= 5:
        f = f + 0.2 * np.sin(X[:, 4])
    sd = np.sqrt(1.0 + 0.5 * X[:, 0] ** 2)
    eps = noise_scale * sd * _unit_t(rng, n, 5)
    return X, f + eps, f


def dgp2(n, p, rng, noise_scale=1.0):
    X = np.empty((n, p))
    X[:, :3] = rng.random((n, 3))
    X[:, 3] = rng.uniform(-1.0, 1.0, n)
    if p > 4:
        common = rng.standard_normal(n)
        rho = 0.8
        X[:, 4:] = (np.sqrt(rho) * common[:, None]
                    + np.sqrt(1.0 - rho) * rng.standard_normal((n, p - 4)))
    f = 0.6 * (np.exp(-4.0 * (X[:, 0] - 0.3) ** 2)
               - np.exp(-4.0 * (X[:, 1] - 0.7) ** 2)) + 0.4 * (X[:, 2] > 0.5)
    sd = np.sqrt(0.5 + X[:, 3] ** 2)
    eps = noise_scale * sd * (0.9 * _unit_t(rng, n, 4)
                              + 0.1 * rng.standard_normal(n))
    return X, f + eps, f


def dgp3(n, p, rng, noise_scale=1.2):
    """Heavy-tail stress case: t_{2.5} innovations, infinite fourth moment."""
    X = _ar1_gaussian(rng, n, p, rho=0.3)
    f = 0.4 * np.sign(X[:, 0]) * np.sqrt(np.abs(X[:, 0]))
    if p >= 3:
        f = f + 0.25 * X[:, 1] * (X[:, 2] > 0)
    sd = np.sqrt(0.5 + 0.5 * np.abs(X[:, 0]))
    eps = noise_scale * sd * _unit_t(rng, n, 2.5)
    return X, f + eps, f


DGPS = {"dgp1": dgp1, "dgp2": dgp2, "dgp3": dgp3}


def signal_to_noise(y, f):
    """Var(f) / Var(y - f), reported for each configuration."""
    resid = np.asarray(y) - np.asarray(f)
    vr = float(np.var(resid))
    return float(np.var(f) / vr) if vr > 0 else np.nan
