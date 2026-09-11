"""
Second block of numerical checks.

(a) Sampling variance of a bootstrap leaf mean, and the correct degrees of
    freedom for the pooled within-leaf variance when observations carry
    integer multiplicities.
(b) The DerSimonian-Laird estimator of the between-leaf variance under
    strong heteroskedasticity of the leaf-mean variances.
(c) The algebraic identity linking the empirical-Bayes weight built from the
    DerSimonian-Laird estimator to the James-Stein weight.
"""
import numpy as np

rng = np.random.default_rng(4242)


def banner(t):
    print("\n" + "=" * 74)
    print(t)
    print("=" * 74)


# ---------------------------------------------------------------------------
banner("A.  Bootstrap multiplicities: Var(leaf mean) and the pooled-variance df")

m_distinct, sigma2 = 40, 2.0
M = 200_000
var_nom, var_eff, num, den_nom, den_eff = [], [], [], [], []
for _ in range(4000):
    c = rng.poisson(1.0, size=m_distinct)
    c = c[c > 0]
    if c.size < 5:
        continue
    Cs, C2 = c.sum(), (c ** 2).sum()
    y = rng.normal(0.0, np.sqrt(sigma2), size=(200, c.size))
    ybar = (y * c).sum(axis=1) / Cs
    var_nom.append(sigma2 / Cs)
    var_eff.append(sigma2 * C2 / Cs ** 2)
    num.append(np.mean(((y - ybar[:, None]) ** 2 * c).sum(axis=1)))
    den_nom.append(Cs - 1)
    den_eff.append(Cs - C2 / Cs)
    if len(var_nom) == 1:
        emp = ybar.var(ddof=1)
mean_num = float(np.mean(num))
print(f"  E[sum_i c_i (y_i - ybar)^2] / sigma^2 : mc = {mean_num/sigma2:8.4f}")
print(f"     nominal df  (C - 1)               :      {np.mean(den_nom):8.4f}")
print(f"     effective df (C - sum c^2 / C)    :      {np.mean(den_eff):8.4f}")
print(f"  ratio of nominal to effective leaf size, C / (C^2/sum c^2) = "
      f"{np.mean([a/b for a, b in zip(var_eff, var_nom)]):.4f}")
print("  (a ratio near 2 means the nominal leaf count overstates the "
      "information by about a factor of two)")

# direct check of Var(ybar) on one multiplicity pattern
c = rng.poisson(1.0, size=m_distinct)
c = c[c > 0]
Cs, C2 = c.sum(), (c ** 2).sum()
y = rng.normal(0.0, np.sqrt(sigma2), size=(400_000, c.size))
ybar = (y * c).sum(axis=1) / Cs
print(f"  Var(ybar) mc = {ybar.var(ddof=1):.6f}   "
      f"sigma^2 sum c^2 / C^2 = {sigma2*C2/Cs**2:.6f}   "
      f"sigma^2 / C = {sigma2/Cs:.6f}")


# ---------------------------------------------------------------------------
banner("B.  DerSimonian-Laird estimator of tau^2 under heteroskedastic v_l")

K = 9
v = np.array([0.02, 0.05, 0.4, 0.01, 1.5, 0.08, 0.03, 0.9, 0.2])
tau2 = 0.12
u = 1.0 / v
den_dl = u.sum() - (u ** 2).sum() / u.sum()
M = 300_000
mu = rng.normal(0.0, np.sqrt(tau2), size=(M, K))
yb = mu + rng.normal(0.0, 1.0, size=(M, K)) * np.sqrt(v)
th_u = (yb * u).sum(axis=1) / u.sum()
Q = (u * (yb - th_u[:, None]) ** 2).sum(axis=1)
print(f"  E[Q] mc = {Q.mean():9.5f}    theory (K-1) + tau^2 * den = "
      f"{(K-1) + tau2*den_dl:9.5f}")
tau_dl = np.maximum((Q - (K - 1)) / den_dl, 0.0)
w_un = np.full(K, 1.0 / K)
c_un = 1.0 - 2 * w_un + K * w_un ** 2
th_w = yb.mean(axis=1)
S2 = ((yb - th_w[:, None]) ** 2).sum(axis=1)
tau_mom = np.maximum((S2 - (c_un * v).sum()) / c_un.sum(), 0.0)
print(f"  E[tau2_DL ] = {tau_dl.mean():.5f}   sd = {tau_dl.std():.5f}   "
      f"(target {tau2})")
print(f"  E[tau2_MoM] = {tau_mom.mean():.5f}   sd = {tau_mom.std():.5f}   "
      f"(target {tau2})")
print("  (the unweighted moment estimator is unbiased before truncation but "
      "far noisier\n   when the v_l differ by two orders of magnitude, so "
      "truncation at zero bites hard)")


# ---------------------------------------------------------------------------
banner("C.  Balanced case: EB weight from DL equals 1 - (K-1)/Q ; JS uses K-3")

Kb, vb = 15, 0.25
tau_true = 0.18
Mb = 200_000
mub = rng.normal(0.0, np.sqrt(tau_true), size=(Mb, Kb))
ybb = mub + rng.normal(0.0, np.sqrt(vb), size=(Mb, Kb))
thb = ybb.mean(axis=1)
Qb = ((ybb - thb[:, None]) ** 2).sum(axis=1) / vb
ub = np.full(Kb, 1.0 / vb)
den_b = ub.sum() - (ub ** 2).sum() / ub.sum()
tau_dl_b = np.maximum((Qb - (Kb - 1)) / den_b, 0.0)
lam_eb = np.where(tau_dl_b > 0, tau_dl_b / (tau_dl_b + vb), 0.0)
lam_cf = np.maximum(1.0 - (Kb - 1) / Qb, 0.0)
lam_js = np.maximum(1.0 - (Kb - 3) / Qb, 0.0)
print(f"  max |lambda_EB(DL) - (1 - (K-1)/Q)_+|  = "
      f"{np.abs(lam_eb - lam_cf).max():.3e}")
print(f"  mean lambda_EB = {lam_eb.mean():.5f}   "
      f"mean lambda_JS = {lam_js.mean():.5f}")
L_raw = (((ybb - mub) ** 2) / vb).sum(axis=1).mean()
L_eb = ((((thb[:, None] + lam_eb[:, None]*(ybb - thb[:, None])) - mub) ** 2)
        / vb).sum(axis=1).mean()
L_js = ((((thb[:, None] + lam_js[:, None]*(ybb - thb[:, None])) - mub) ** 2)
        / vb).sum(axis=1).mean()
lam_or = tau_true / (tau_true + vb)
L_or = ((((lam_or*ybb) - mub) ** 2) / vb).sum(axis=1).mean()
print(f"  weighted Bayes risk:  raw {L_raw:7.4f}   JS {L_js:7.4f}   "
        f"EB {L_eb:7.4f}   oracle {L_or:7.4f}   (K = {Kb})")
print(f"  JS theory without positive part: K - (K-3)(1-lambda) = "
      f"{Kb - (Kb-3)*(1-lam_or):.4f}")
print("\nAll checks complete.")
