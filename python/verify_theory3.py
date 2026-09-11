"""
Third block of numerical checks, covering the aggregation results and the
practical rule that applies a different weight in every tree.

(A) The risk identity for a forest contracted by a single factor toward a
    fixed anchor, and the interval of factors that improve on it.
(B) The exact decomposition of the heterogeneous rule into that contracted
    forest plus an across-tree covariance term, and the bound on the term.
(C) The crossover ensemble size.
(D) The calibration estimator of the forest-level optimal factor.

Each block compares a closed-form expression against a Monte-Carlo average
taken over independent replications of the whole sampling mechanism.
"""

import numpy as np

rng = np.random.default_rng(20260911)


def banner(t):
    print("\n" + "=" * 76)
    print(t)
    print("=" * 76)


# ---------------------------------------------------------------------------
banner("A.  Risk of a forest contracted by a single factor")

# A synthetic ensemble with a known mean, a known between-tree correlation and
# a known bias, so that every quantity in the theorem is available exactly.
B, M = 40, 400_000
vbar, rho = 0.9, 0.25
f_true, theta = 1.30, 0.0
m = 1.10                      # E[forest prediction]; m - f is the bias
g, Delta = m - theta, m - f_true
V_B = vbar * (rho + (1 - rho) / B)

common = rng.normal(0.0, np.sqrt(rho * vbar), size=M)
idio = rng.normal(0.0, np.sqrt((1 - rho) * vbar), size=(M, B))
fhat = m + common + idio.mean(axis=1)

print(f"  Var(fhat) mc = {fhat.var(ddof=1):.6f}   "
      f"vbar(rho + (1-rho)/B) = {V_B:.6f}")

lam_star = g * (f_true - theta) / (V_B + g ** 2)
print(f"  lambda* (theory) = {lam_star:.6f}")
print(f"  {'lambda':>8} {'risk mc':>12} {'risk theory':>12} {'gap formula':>13}")
best_mc, best_lam = np.inf, None
for lam in (0.5, 0.8, lam_star, 1.0, 1.2, 1.4):
    risk_mc = float(np.mean((theta + lam * (fhat - theta) - f_true) ** 2))
    risk_th = lam ** 2 * V_B + (Delta + (lam - 1) * g) ** 2
    gap = (V_B + g ** 2) * (lam - lam_star) ** 2
    risk_min = lam_star ** 2 * V_B + (Delta + (lam_star - 1) * g) ** 2
    print(f"  {lam:8.4f} {risk_mc:12.6f} {risk_th:12.6f} "
          f"{risk_min + gap:13.6f}")
    if risk_mc < best_mc:
        best_mc, best_lam = risk_mc, lam

lo, hi = sorted((2 * lam_star - 1, 1.0))
print(f"  improvement window = ({lo:.4f}, {hi:.4f})")
mid = 0.5 * (lo + hi)
for lam in (lo - 0.02, lo + 0.002, mid, hi - 0.002, hi + 0.02):
    r1 = float(np.mean((fhat - f_true) ** 2))
    rl = float(np.mean((theta + lam * (fhat - theta) - f_true) ** 2))
    inside = lo < lam < hi
    ok = (rl < r1) == inside
    print(f"    lambda = {lam:6.3f}  inside = {str(inside):5s}  "
          f"improves = {str(rl < r1):5s}  {'ok' if ok else 'MISMATCH'}")


# ---------------------------------------------------------------------------
banner("B.  Heterogeneous weights: exact decomposition and its bound")

# f_SL = mean_b [theta_b + lam_b (ybar_b - theta_b)]
#      = f_lambar + C,  C = mean_b (lam_b - lambar)(D_b - Dbar),  D_b = ybar_b - theta_b
Bh, Mh = 25, 200_000
ybar = rng.normal(1.0, 0.8, size=(Mh, Bh))
theta_b = rng.normal(0.2, 0.15, size=(Mh, Bh))
lam_b = rng.uniform(0.55, 0.95, size=(Mh, Bh))

f_SL = (theta_b + lam_b * (ybar - theta_b)).mean(axis=1)
a = theta_b.mean(axis=1)
D = ybar - theta_b
Dbar = D.mean(axis=1)
lambar = lam_b.mean(axis=1)
f_lambar = a + lambar * Dbar
C = ((lam_b - lambar[:, None]) * (D - Dbar[:, None])).mean(axis=1)

print(f"  max |f_SL - (f_lambar + C)| = {np.abs(f_SL - (f_lambar + C)).max():.3e}")
# f_lambar must equal the contracted forest with anchor a
fhat_rf = ybar.mean(axis=1)
print(f"  max |f_lambar - (a + lambar (fhat_RF - a))| = "
      f"{np.abs(f_lambar - (a + lambar * (fhat_rf - a))).max():.3e}")

s_lam = lam_b.std(axis=1, ddof=0)
s_D = D.std(axis=1, ddof=0)
viol = float(np.mean(np.abs(C) > s_lam * s_D + 1e-12))
print(f"  fraction of draws violating |C| <= s_lambda * s_D : {viol:.4f}")
print(f"  mean |C| = {np.abs(C).mean():.5f},  mean bound = "
      f"{(s_lam * s_D).mean():.5f},  mean |f_SL| = {np.abs(f_SL).mean():.5f}")

# with a weight that does not vary across trees the correction vanishes
lam_c = np.full((Mh, Bh), 0.75)
f_SLc = (theta_b + lam_c * (ybar - theta_b)).mean(axis=1)
Cc = ((lam_c - lam_c.mean(axis=1)[:, None])
      * (D - Dbar[:, None])).mean(axis=1)
print(f"  constant-weight case: max |C| = {np.abs(Cc).max():.3e}")


# ---------------------------------------------------------------------------
banner("C.  Crossover ensemble size")

# parameters chosen so that the denominator of B* is positive for some of the
# weights below and negative for others, which is what makes the test bite
for (vbar_c, rho_c, g_c) in ((1.0, 0.02, 1.20), (1.0, 0.10, 0.55)):
  print(f"  vbar = {vbar_c}, rho = {rho_c}, g = {g_c}")
  for lam in (0.60, 0.70, 0.80, 0.90):
    num = (1 + lam) * (1 - rho_c) * vbar_c
    den = (1 - lam) * g_c ** 2 - (1 + lam) * rho_c * vbar_c
    Bstar = num / den if den > 0 else np.inf
    # scan B and record the largest B at which the factor still improves,
    # comparing the exact risks rather than the interval, so the check is
    # independent of the algebra it is testing
    last_ok = None
    for Bx in range(1, 5000):
        V = vbar_c * (rho_c + (1 - rho_c) / Bx)
        r1 = V + 0.0 ** 2
        rl = lam ** 2 * V + ((lam - 1) * g_c) ** 2
        if rl < r1:
            last_ok = Bx
    print(f"    lambda = {lam:.2f}:  B* formula = "
          f"{('inf' if not np.isfinite(Bstar) else f'{Bstar:9.2f}'):>9}   "
          f"largest improving B, exact risks = "
          f"{('>=4999' if last_ok == 4999 else str(last_ok)):>8}")


# ---------------------------------------------------------------------------
banner("D.  Calibrating the forest-level factor from held-out predictions")

# The best linear predictor of f from fhat has slope Cov(m, f) / Var(fhat),
# and regressing Y rather than f on fhat estimates the same slope because the
# noise in Y is uncorrelated with fhat when fhat is out of sample.
Md = 200_000
sig_eps = 1.0
f_x = rng.normal(0.0, 1.0, size=Md)          # the regression function at x
kappa = 0.65                                  # the forest contracts f toward 0
V_pred = 0.40
m_x = kappa * f_x
fhat_x = m_x + rng.normal(0.0, np.sqrt(V_pred), size=Md)
Y = f_x + rng.normal(0.0, sig_eps, size=Md)

lam_true = float(np.cov(m_x, f_x)[0, 1] / np.var(fhat_x))
slope_f = float(np.cov(fhat_x, f_x)[0, 1] / np.var(fhat_x))
slope_Y = float(np.cov(fhat_x, Y)[0, 1] / np.var(fhat_x))
print(f"  Cov(m,f)/Var(fhat)          = {lam_true:.5f}")
print(f"  regression of f on fhat     = {slope_f:.5f}")
print(f"  regression of Y on fhat     = {slope_Y:.5f}")
print("  (the last two agree, so the response may be used in place of the "
      "unobserved\n   regression function when the prediction is out of sample)")

# transporting a slope fitted at one prediction variance to another
V_small, V_large = 1.10, 0.40
fhat_s = m_x + rng.normal(0.0, np.sqrt(V_small), size=Md)
slope_s = float(np.cov(fhat_s, Y)[0, 1] / np.var(fhat_s))
transported = slope_s * np.var(fhat_s) / (np.var(fhat_s) - V_small + V_large)
print(f"  slope at the larger variance = {slope_s:.5f}")
print(f"  transported to the smaller   = {transported:.5f}   "
      f"target = {slope_Y:.5f}")

print("\nAll checks complete.")
