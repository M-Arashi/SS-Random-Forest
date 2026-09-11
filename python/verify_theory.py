"""
Numerical verification of the finite-sample identities used in the
Stein-Shrunken Random Forest theory.

Each block below is an independent Monte-Carlo check of one algebraic
identity in the manuscript.  The blocks are self-contained and are run
before any of the identities are used elsewhere.
"""
import numpy as np

rng = np.random.default_rng(20260910)


def banner(txt):
    print("\n" + "=" * 74)
    print(txt)
    print("=" * 74)


# ---------------------------------------------------------------------------
# 1.  Method-of-moments estimator of the between-leaf variance tau^2 under
#     UNEQUAL leaf-size weights.
#
#     ybar_l = theta + u_l + e_l,   u_l ~ (0, tau^2),  e_l ~ (0, v_l)
#     theta_hat = sum_l w_l ybar_l,  S^2 = sum_l (ybar_l - theta_hat)^2
#
#     Claim:  E[S^2] = sum_l c_l (tau^2 + v_l),  c_l = 1 - 2 w_l + K w_l^2.
# ---------------------------------------------------------------------------
banner("1.  E[S^2] = sum_l c_l (tau^2 + v_l),  c_l = 1 - 2 w_l + K w_l^2")

K = 7
n_leaf = np.array([12, 40, 200, 25, 90, 8, 400], dtype=float)
sigma2, tau2 = 2.3, 0.45
v = sigma2 / n_leaf
w = n_leaf / n_leaf.sum()
c = 1.0 - 2.0 * w + K * w ** 2

M = 400_000
u = rng.normal(0.0, np.sqrt(tau2), size=(M, K))
e = rng.normal(0.0, 1.0, size=(M, K)) * np.sqrt(v)
ybar = u + e
theta_hat = ybar @ w
S2 = ((ybar - theta_hat[:, None]) ** 2).sum(axis=1)

pred_new = (c * (tau2 + v)).sum()
pred_old = (K - 1) * tau2 + v.sum()          # estimator used in the first draft
print(f"  Monte-Carlo  E[S^2]              = {S2.mean():.6f}  "
      f"(mc se {S2.std(ddof=1)/np.sqrt(M):.6f})")
print(f"  corrected    sum_l c_l(tau^2+v_l) = {pred_new:.6f}")
print(f"  first draft  (K-1)tau^2 + sum v_l = {pred_old:.6f}")
print(f"  => first-draft tau^2 estimator bias factor: "
      f"{(pred_new - v.sum()) / ((K - 1) * tau2):.4f} (1.0 would be unbiased)")

# corrected plug-in estimator, checked for unbiasedness of the numerator
tau2_hat_new = (S2 - (c * v).sum()) / c.sum()
tau2_hat_old = (S2 - v.sum()) / (K - 1)
print(f"  E[tau2_hat] corrected = {tau2_hat_new.mean():.6f}   (target {tau2:.6f})")
print(f"  E[tau2_hat] as drafted = {tau2_hat_old.mean():.6f}   (target {tau2:.6f})")


# ---------------------------------------------------------------------------
# 2.  Exact oracle risk improvement in the precision-weighted loss
#     L(m) = sum_l (m_l - mu_l)^2 / v_l .
#
#     Claim:  E[L(ybar)] = K  and  E[L(oracle)] = sum_l lambda_l,
#     so the improvement is exactly sum_l (1 - lambda_l).
# ---------------------------------------------------------------------------
banner("2.  E[L(ybar)] = K ;  E[L(oracle)] = sum_l lambda_l")

lam = tau2 / (tau2 + v)
mu = 0.0 + u                                  # theta = 0 without loss of generality
oracle = 0.0 + lam * (ybar - 0.0)
L_raw = (((ybar - mu) ** 2) / v).sum(axis=1)
L_orc = (((oracle - mu) ** 2) / v).sum(axis=1)
print(f"  E[L(ybar)]   mc = {L_raw.mean():.5f}      theory = {K}")
print(f"  E[L(oracle)] mc = {L_orc.mean():.5f}      theory = {lam.sum():.5f}")
print(f"  improvement  mc = {(L_raw - L_orc).mean():.5f}  theory = {(1-lam).sum():.5f}")


# ---------------------------------------------------------------------------
# 3.  The bound asserted in the first draft,
#         improvement  >=  (K-2) tau^4 / [ (tau^2 + vbar) sum_l v_l ],
#     is checked against the exact improvement sum_l (1 - lambda_l).
# ---------------------------------------------------------------------------
banner("3.  Status of the improvement bound asserted in the first draft")

for tau2_t in [0.01, 0.1, 0.45, 5.0, 50.0]:
    lam_t = tau2_t / (tau2_t + v)
    exact = (1.0 - lam_t).sum()
    drafted = (K - 2) * tau2_t ** 2 / ((tau2_t + v.mean()) * v.sum())
    flag = "HOLDS" if drafted <= exact + 1e-12 else "FAILS"
    print(f"  tau^2 = {tau2_t:7.2f}:  exact = {exact:10.5f}   "
          f"drafted bound = {drafted:12.5f}   {flag}")


# ---------------------------------------------------------------------------
# 4.  James-Stein shrinkage toward the precision-weighted mean, applied in the
#     whitened coordinates z_l = ybar_l / sqrt(v_l).
#
#     Claim (frequentist, no prior):  for every fixed mu and K >= 4,
#         E[L(mu_JS)] = K - (K-3)^2 E[1/Q],   Q = sum_l (ybar_l-theta_hat)^2/v_l
#     with theta_hat the precision-weighted mean.  Note that when
#     v_l = sigma^2/n_l the precision weights coincide with the leaf-size
#     weights w_l used throughout the paper.
# ---------------------------------------------------------------------------
banner("4.  Frequentist risk of JS toward the precision-weighted mean")

prec = 1.0 / v
w_prec = prec / prec.sum()
print(f"  precision weights vs leaf-size weights, max abs diff = "
      f"{np.abs(w_prec - w).max():.2e}")

mu_fixed = np.array([0.9, -0.4, 0.15, 2.1, -1.3, 0.0, 0.55])   # arbitrary, fixed
Mf = 600_000
ybar_f = mu_fixed + rng.normal(0.0, 1.0, size=(Mf, K)) * np.sqrt(v)
theta_f = ybar_f @ w_prec
d = ybar_f - theta_f[:, None]
Q = ((d ** 2) / v).sum(axis=1)

for cshr, name in [(K - 3, "c = K-3"), (K - 2, "c = K-2 (too large here)")]:
    shrink = 1.0 - cshr / Q
    js = theta_f[:, None] + shrink[:, None] * d
    jsp = theta_f[:, None] + np.maximum(shrink, 0.0)[:, None] * d
    L_js = (((js - mu_fixed) ** 2) / v).sum(axis=1).mean()
    L_jsp = (((jsp - mu_fixed) ** 2) / v).sum(axis=1).mean()
    pred = K - cshr * (2 * (K - 3) - cshr) * np.mean(1.0 / Q)
    print(f"  {name:26s} risk mc = {L_js:8.5f}  SURE prediction = {pred:8.5f}"
          f"   positive part = {L_jsp:8.5f}   (unshrunken = {K})")


# ---------------------------------------------------------------------------
# 5.  Balanced-leaf Bayes risk of the JS rule:  K - (K-3)(1 - lambda).
# ---------------------------------------------------------------------------
banner("5.  Bayes risk of the JS rule with balanced leaves")

Kb = 12
vb = 0.30
tau2b = 0.20
lamb = tau2b / (tau2b + vb)
Mb = 400_000
mub = rng.normal(0.0, np.sqrt(tau2b), size=(Mb, Kb))
yb = mub + rng.normal(0.0, np.sqrt(vb), size=(Mb, Kb))
tb = yb.mean(axis=1)
db = yb - tb[:, None]
Qb = (db ** 2).sum(axis=1) / vb
shr = np.maximum(1.0 - (Kb - 3) / Qb, 0.0)
jsb = tb[:, None] + shr[:, None] * db
print(f"  E[L(ybar)]  mc = {(((yb-mub)**2)/vb).sum(axis=1).mean():.5f}   theory {Kb}")
print(f"  E[L(JS+)]   mc = {(((jsb-mub)**2)/vb).sum(axis=1).mean():.5f}   "
      f"theory (no pos. part) {Kb - (Kb-3)*(1-lamb):.5f}")
print(f"  E[L(oracle)] mc = "
      f"{(((lamb*yb - mub)**2)/vb).sum(axis=1).mean():.5f}   theory {Kb*lamb:.5f}")


# ---------------------------------------------------------------------------
# 6.  Effect of a misspecified within-leaf variance.
#
#     If sigma^2 is replaced by sigma^2 (1 + delta), the fraction of the oracle
#     risk reduction that is lost is exactly
#         [ lambda delta / (1 + (1 - lambda) delta) ]^2 ,
#     which is strictly below 1 for every delta > -1.
# ---------------------------------------------------------------------------
banner("6.  Cost of a misspecified within-leaf variance")

v0, tau0 = 0.5, 0.5
lam0 = tau0 / (tau0 + v0)
print(f"  lambda = {lam0:.3f}")
print(f"  {'delta':>8} {'risk(shrunk)':>14} {'risk(raw)':>10} "
      f"{'frac. of gain lost':>20} {'formula':>10}")
for delta in [-0.9, -0.5, -0.2, 0.0, 0.5, 2.0, 10.0]:
    lam_d = tau0 / (tau0 + v0 * (1 + delta))
    r_c = lam_d ** 2 + (1 - lam_d) ** 2 * tau0 / v0       # weighted-loss risk
    r_1 = 1.0
    r_lam = lam0
    frac = (r_c - r_lam) / (r_1 - r_lam)
    formula = (lam0 * delta / (1 + (1 - lam0) * delta)) ** 2
    print(f"  {delta:8.2f} {r_c:14.6f} {r_1:10.6f} {frac:20.6f} {formula:10.6f}")


# ---------------------------------------------------------------------------
# 7.  Per-leaf safety region:  the shrunken leaf beats the raw leaf mean iff
#     (mu_l - theta)^2 < 2 tau^2 + v_l.
# ---------------------------------------------------------------------------
banner("7.  Per-leaf safety region  (mu-theta)^2 < 2 tau^2 + v")

vv, tt = 0.4, 0.25
ll = tt / (tt + vv)
thr = 2 * tt + vv
for gap2 in [0.0, 0.5 * thr, 0.99 * thr, 1.01 * thr, 3 * thr]:
    risk_shrunk = ll ** 2 * vv + (1 - ll) ** 2 * gap2
    better = risk_shrunk < vv
    print(f"  (mu-theta)^2 = {gap2:7.4f} (threshold {thr:.4f}):  "
          f"risk {risk_shrunk:.6f} vs {vv:.6f}  -> "
          f"{'improves' if better else 'does not improve'}")

print("\nAll checks complete.")
