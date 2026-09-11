"""
aggregate.py
============

Turns the replication-level output of simulation.py and realdata.py into the
tables, figures and macro definitions that the manuscript reads.

Two conventions are followed throughout.

Medians carry a nonparametric bootstrap standard error.  The asymptotic
formula 1.253 s / sqrt(R) presumes a normal sampling distribution and
misbehaves when the replication-level losses are heavy-tailed, which is the
regime these processes were built to sit in.

Every number quoted in the manuscript text is written here into
``sim_macros.tex`` and read back by name, so a figure in a sentence cannot
drift away from the table it came from.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

import evaluation as E

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SIM = os.path.join(ROOT, "outputs", "sim")
REAL = os.path.join(ROOT, "outputs", "real")

DGP_LABEL = {"dgp1": "DGP~1", "dgp2": "DGP~2", "dgp3": "DGP~3"}


_NUM_WORDS = {0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five",
              6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"}


def numword(x):
    """Spell an integer with letters only.

    A LaTeX control sequence may not contain digits, so a macro named after a
    numeric setting has to spell the number out.
    """
    return "".join(_NUM_WORDS[int(c)] for c in str(int(x)))


def fmt(x, d=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "---"
    return f"{x:.{d}f}"


def table(body, header, caption, label, align, note="", wide_at=8):
    """Assemble one LaTeX table.

    Tables with many columns are wrapped in a box scaled to the text width,
    because at this page size a ten-column table otherwise runs into the
    margin and the overfull box is what a reader sees.
    """
    ncol = sum(1 for c in align if c in "lrc")
    inner = (f"\\begin{{tabular}}{{{align}}}\n\\toprule\n"
             f"{header} \\\\\n\\midrule\n"
             + "\n".join(body)
             + "\n\\bottomrule\n\\end{tabular}")
    if ncol > wide_at:
        inner = "\\resizebox{\\textwidth}{!}{%\n" + inner + "}"
    return (
        "\\begin{table}[htbp]\n\\centering\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        "\\small\n\\setlength{\\tabcolsep}{4pt}\n"
        + inner + "\n"
        + (f"\\\\[2pt]\n\\parbox{{\\textwidth}}{{\\footnotesize {note}}}\n"
           if note else "")
        + "\\end{table}\n")


# ---------------------------------------------------------------------------
#  Experiment A: the substitution surface
# ---------------------------------------------------------------------------

def substitution(macros):
    path = os.path.join(SIM, "sim_substitution.csv")
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path)
    w = d.pivot_table(index=["dgp", "min_leaf", "B", "rep"], columns="method",
                      values="mse_true").reset_index()
    w["g_js"] = 100 * (w["SSRF-JS"] / w["RF"] - 1)
    w["g_eb"] = 100 * (w["SSRF"] / w["RF"] - 1)

    rows = []
    for (dgp, nl, B), grp in w.groupby(["dgp", "min_leaf", "B"]):
        stat, pv, frac = E.wilcoxon_win(grp["RF"] - grp["SSRF-JS"])
        rows.append(dict(dgp=dgp, min_leaf=nl, B=B,
                         gain_js=grp["g_js"].median(),
                         gain_eb=grp["g_eb"].median(),
                         se=E.bootstrap_median_se(grp["g_js"].values),
                         win=100 * float(np.mean(grp["g_js"] < 0)),
                         p=pv, reps=len(grp)))
    g = pd.DataFrame(rows)
    g["p_holm"] = np.nan
    for nl, idx in g.groupby("min_leaf").groups.items():
        g.loc[idx, "p_holm"] = E.holm(g.loc[idx, "p"].values)
    g.to_csv(os.path.join(SIM, "substitution_summary.csv"), index=False)

    Bs = sorted(g.B.unique())
    body = []
    for nl in sorted(g.min_leaf.unique()):
        for dgp in sorted(g.dgp.unique()):
            s = g[(g.min_leaf == nl) & (g.dgp == dgp)].set_index("B").reindex(Bs)
            cells = " & ".join(fmt(v, 1) for v in s.gain_js)
            body.append(f"{DGP_LABEL[dgp]} & {nl} & {cells} \\\\")
        if nl != sorted(g.min_leaf.unique())[-1]:
            body.append("\\addlinespace")
    reps = int(g.reps.max())
    txt = table(
        body,
        "Process & $n_0$ & " + " & ".join(f"$B{{=}}{b}$" for b in Bs),
        ("Median percentage change in test mean squared error against the "
         "true conditional mean when the leaf means of a fitted forest are "
         "replaced by their James--Stein shrunken values. Negative entries "
         f"favour shrinkage. Sample size $n=600$, $p=10$, $R={reps}$ "
         "replications, minimum leaf size $n_0$. Each replication grows one "
         "forest and truncates it to each ensemble size, so the comparison "
         "across $B$ is paired."),
        "tab:substitution",
        "ll" + "r" * len(Bs))
    open(os.path.join(SIM, "Table_Sim_Substitution.tex"), "w").write(txt)

    body = []
    for nl in sorted(g.min_leaf.unique()):
        for dgp in sorted(g.dgp.unique()):
            s = g[(g.min_leaf == nl) & (g.dgp == dgp)].set_index("B").reindex(Bs)
            cells = " & ".join(fmt(v, 0) for v in s.win)
            body.append(f"{DGP_LABEL[dgp]} & {nl} & {cells} \\\\")
        if nl != sorted(g.min_leaf.unique())[-1]:
            body.append("\\addlinespace")
    txt = table(
        body,
        "Process & $n_0$ & " + " & ".join(f"$B{{=}}{b}$" for b in Bs),
        ("Percentage of replications in which the shrunken forest attains a "
         "lower test mean squared error than the forest it was built from, "
         "for the configurations of Table~\\ref{tab:substitution}."),
        "tab:substitution_wins",
        "ll" + "r" * len(Bs))
    open(os.path.join(SIM, "Table_Sim_SubstitutionWins.tex"), "w").write(txt)

    def pick(nl, B, dgp=None):
        s = g[(g.min_leaf == nl) & (g.B == B)]
        if dgp:
            s = s[s.dgp == dgp]
        return float(s.gain_js.median())

    # the two shrinkage rules against each other, configuration by
    # configuration; negative means the empirical-Bayes rule is the more
    # accurate of the two
    w["eb_vs_js"] = 100 * (w["SSRF"] / w["SSRF-JS"] - 1)
    ebjs = w.groupby(["dgp", "min_leaf", "B"])["eb_vs_js"].median()

    macros.update({
        "SimReps": str(reps),
        "EbVsJsMedian": fmt(-float(ebjs.median()), 2),
        "EbVsJsBest": fmt(-float(ebjs.min()), 1),
        "EbVsJsWorst": fmt(float(ebjs.max()), 1),
        "EbVsJsConfigs": str(int(len(ebjs))),
        "EbVsJsEbWins": str(int((ebjs < 0).sum())),
        "GainSmallLeafBOne": fmt(-pick(5, 1), 1),
        "GainSmallLeafBTen": fmt(-pick(5, 10), 1),
        "GainSmallLeafBFiveHundred": fmt(-pick(5, 500), 1),
        "GainLargeLeafBOne": fmt(-pick(20, 1), 1),
        "GainLargeLeafBTen": fmt(-pick(20, 10), 1),
        "GainLargeLeafBFiveHundred": fmt(pick(20, 500), 1),
        "HeavyTailPenalty": fmt(pick(20, 500, "dgp3"), 1),
        "MeanLambdaSmallLeaf": fmt(float(d[d.min_leaf == 5].lam_js.mean()), 3),
        "MeanLambdaLargeLeaf": fmt(float(d[d.min_leaf == 20].lam_js.mean()), 3),
        "MedianSNR": fmt(float(d.snr.median()), 3),
    })
    return g


# ---------------------------------------------------------------------------
#  Experiment B: the accuracy grid
# ---------------------------------------------------------------------------

GRID_METHODS = ["RF", "SSRF-JS", "SSRF", "SSRF-rob", "RF-H", "SSRF-JS-H",
                "RF-tuned", "SSRF-tuned", "RidgeRF", "LassoRF", "GBM"]


def grid(macros):
    path = os.path.join(SIM, "sim_grid.csv")
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path)
    agg = (d.groupby(["dgp", "n", "p", "B", "method"])
             .agg(mse_true=("mse_true", "median"),
                  mse=("mse", "median"),
                  coverage=("coverage", "median"),
                  isc=("interval_score", "median"),
                  rt=("runtime", "mean"),
                  reps=("rep", "nunique"))
             .reset_index())
    agg.to_csv(os.path.join(SIM, "grid_summary.csv"), index=False)

    for B in sorted(agg.B.unique()):
        blk = agg[agg.B == B]
        present = [m for m in GRID_METHODS if m in set(blk.method)]
        body = []
        for dgp in sorted(blk.dgp.unique()):
            for (n, p) in sorted({(r.n, r.p) for r in blk.itertuples()}):
                s = blk[(blk.dgp == dgp) & (blk.n == n) & (blk.p == p)]
                s = s.set_index("method").reindex(present)
                v = s.mse_true.values
                cells = [fmt(x, 4) for x in v]
                if np.isfinite(v).any():
                    j = int(np.nanargmin(v))
                    cells[j] = "\\textbf{" + cells[j] + "}"
                body.append(f"{DGP_LABEL[dgp]} & $({n},{p})$ & "
                            + " & ".join(cells) + " \\\\")
            body.append("\\addlinespace")
        reps = int(blk.reps.max())
        txt = table(
            body[:-1],
            "Process & $(n,p)$ & "
            + " & ".join(m.replace("-", "--") for m in present),
            (f"Median test mean squared error against the true conditional "
             f"mean, ensembles of $B={B}$ trees, minimum leaf size $5$, "
             f"$R={reps}$ replications. Bold marks the smallest entry in each "
             "row. Columns marked H are honest, subsampled forests."),
            f"tab:grid_B{B}", "ll" + "r" * len(present))
        open(os.path.join(SIM, f"Table_Sim_Grid_B{B}.tex"), "w").write(txt)

    # interval score and coverage at the small ensemble
    Bmin = int(agg.B.min())
    blk = agg[agg.B == Bmin]
    present = [m for m in GRID_METHODS if m in set(blk.method)]
    body = []
    for dgp in sorted(blk.dgp.unique()):
        for (n, p) in sorted({(r.n, r.p) for r in blk.itertuples()}):
            s = blk[(blk.dgp == dgp) & (blk.n == n) & (blk.p == p)]
            s = s.set_index("method").reindex(present)
            body.append(f"{DGP_LABEL[dgp]} & $({n},{p})$ & "
                        + " & ".join(fmt(x, 3) for x in s["isc"]) + " \\\\")
        body.append("\\addlinespace")
    txt = table(
        body[:-1],
        "Process & $(n,p)$ & "
        + " & ".join(m.replace("-", "--") for m in present),
        (f"Median interval score of nominal $95\\%$ split-conformal "
         f"prediction intervals, ensembles of $B={Bmin}$ trees. The score is "
         "negatively oriented, so smaller is better; it rewards narrow "
         "intervals and penalises misses."),
        "tab:grid_interval", "ll" + "r" * len(present))
    open(os.path.join(SIM, "Table_Sim_Interval.tex"), "w").write(txt)

    # coverage summary macro
    for m in ("RF", "SSRF-JS"):
        s = agg[agg.method == m]
        if len(s):
            macros[f"Coverage{m.replace('-','')}"] = fmt(
                float(s["coverage"].median()), 3)
    if "SSRF-JS" in set(agg.method) and "RF" in set(agg.method):
        for B in sorted(agg.B.unique()):
            a = agg[(agg.B == B) & (agg.method == "RF")].mse_true.median()
            b = agg[(agg.B == B) & (agg.method == "SSRF-JS")].mse_true.median()
            macros[f"GridGainB{numword(B)}"] = fmt(-100 * (b / a - 1), 1)
    return agg


# ---------------------------------------------------------------------------
#  Experiment C: expensive competitors
# ---------------------------------------------------------------------------

def extended(macros):
    path = os.path.join(SIM, "sim_extended.csv")
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path)
    order = ["RF", "SSRF-JS", "SSRF", "SSRF-rob", "RF-tuned", "SSRF-tuned",
             "LLF", "BART", "NGBoost", "GBM", "LassoRF"]
    present = [m for m in order if m in set(d.method)]
    body = []
    for dgp in sorted(d.dgp.unique()):
        s = d[d.dgp == dgp].groupby("method").agg(
            mse=("mse_true", "median"), isc=("interval_score", "median"),
            rt=("runtime", "mean")).reindex(present)
        v = s.mse.values
        cells = [fmt(x, 4) for x in v]
        if np.isfinite(v).any():
            j = int(np.nanargmin(v))
            cells[j] = "\\textbf{" + cells[j] + "}"
        body.append(f"{DGP_LABEL[dgp]} & " + " & ".join(cells) + " \\\\")
    for dgp in sorted(d.dgp.unique()):
        s = d[d.dgp == dgp].groupby("method").agg(
            rt=("runtime", "mean")).reindex(present)
        body.append(f"{DGP_LABEL[dgp]} runtime (s) & "
                    + " & ".join(fmt(x, 2) for x in s.rt) + " \\\\")
    B = int(d.B.iloc[0])
    n = int(d.n.iloc[0])
    p = int(d.p.iloc[0])
    reps = int(d.rep.nunique())
    txt = table(
        body,
        "Process & " + " & ".join(m.replace("-", "--") for m in present),
        (f"Median test mean squared error against the true conditional mean "
         f"at $n={n}$, $p={p}$, with ensembles of $B={B}$ trees and $R={reps}$ "
         "replications, against the competitors that change the leaf rule or "
         "the objective. Every competitor is tuned before it is compared: the "
         "number of sum-of-trees components for BART and the number of "
         "boosting rounds for NGBoost and for the gradient-boosted ensemble "
         "are chosen on a held-out fifth of the training sample, the ridge "
         "penalty of the local linear forest likewise, and the leaf size and "
         "split width of the tuned forest by cross-validation. Mean "
         "wall-clock time per replication is reported below the accuracy "
         "block, and is as much a part of the comparison as the accuracy."),
        "tab:extended", "l" + "r" * len(present))
    open(os.path.join(SIM, "Table_Sim_Extended.tex"), "w").write(txt)
    return d


# ---------------------------------------------------------------------------
#  Experiment D: tree equivalence
# ---------------------------------------------------------------------------

def equivalence(macros):
    path = os.path.join(SIM, "sim_equivalence.csv")
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path)
    a = (d.groupby(["dgp", "B", "method"]).mse_true.mean()
          .unstack("method").reset_index())
    rows = []
    for dgp, s in a.groupby("dgp"):
        s = s.sort_values("B")
        logB, rf = np.log(s.B.values), s["RF"].values
        for B, target in zip(s.B.values, s["SSRF-JS"].values):
            if target >= rf[0]:
                factor = np.nan
            elif target <= rf[-1]:
                factor = np.inf
            else:
                # rf is decreasing in B, so -rf is increasing and can be used
                # directly as the abscissa of the interpolation
                factor = float(np.exp(np.interp(-target, -rf, logB)) / B)
            rows.append(dict(dgp=dgp, B=int(B), rf=float(rf[list(s.B).index(B)]),
                             ssrf=float(target), factor=factor))
    eq = pd.DataFrame(rows)
    eq.to_csv(os.path.join(SIM, "equivalence_summary.csv"), index=False)

    Bs = sorted(eq.B.unique())
    body = []
    for dgp in sorted(eq.dgp.unique()):
        s = eq[eq.dgp == dgp].set_index("B").reindex(Bs)
        cells = []
        for f in s.factor:
            cells.append("$\\infty$" if (f is not None and np.isinf(f))
                         else fmt(f, 1))
        body.append(f"{DGP_LABEL[dgp]} & " + " & ".join(cells) + " \\\\")
    txt = table(
        body,
        "Process & " + " & ".join(f"$B{{=}}{b}$" for b in Bs),
        ("Tree-equivalence factor: the number of unshrunken trees whose risk "
         "matches that of $B$ shrunken ones, divided by $B$, obtained by "
         "interpolating the standard forest's risk curve in $\\log B$. An "
         "entry of $2$ means that shrinkage buys what doubling the ensemble "
         "would have bought, and an entry below $1$ means the shrunken forest "
         "is matched by fewer unshrunken trees, so the shrinkage has done "
         "harm. An entry of $\\infty$ means the shrunken forest of that size "
         "is more accurate than the unshrunken forest at every size examined, "
         "up to $600$ trees. Minimum leaf size $5$, $n=600$, $p=10$."),
        "tab:equivalence", "l" + "r" * len(Bs))
    open(os.path.join(SIM, "Table_Sim_Equivalence.tex"), "w").write(txt)

    small = eq[(eq.B <= 25) & np.isfinite(eq.factor)]
    if len(small):
        macros["MedianEquivFactor"] = fmt(float(small.factor.median()), 1)
    return eq


# ---------------------------------------------------------------------------
#  Experiment E: the best single contraction factor
# ---------------------------------------------------------------------------

def level(macros):
    path = os.path.join(SIM, "sim_level.csv")
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path)
    Bs = sorted(d.B.unique())
    body = []
    for dgp in sorted(d.dgp.unique()):
        for n in sorted(d[d.dgp == dgp].n.unique()):
            s = d[(d.dgp == dgp) & (d.n == n)].set_index("B").reindex(Bs)
            body.append(f"{DGP_LABEL[dgp]} & {n} & "
                        + " & ".join(fmt(v, 3) for v in s.lam_best)
                        + " & " + fmt(float(s.lam_tree.mean()), 3) + " \\\\")
        body.append("\\addlinespace")
    reps = int(d.reps.max())
    txt = table(
        body[:-1],
        "Process & $n$ & "
        + " & ".join(f"$B{{=}}{b}$" for b in Bs)
        + " & tree-level $\\bar{\\hat\\lambda}$",
        ("The single contraction factor that minimises forest risk against "
         "the true conditional mean, found by a grid search on "
         "$[0, 1.4]$, alongside the leaf-size-weighted mean of the factors "
         "the tree-level rule produces. Minimum leaf size $5$, $p=10$, "
         f"$R={reps}$ replications. A best factor at or above one means no "
         "contraction of any size improves the forest, which is the third "
         "branch of the improvement window."),
        "tab:level", "ll" + "r" * len(Bs) + "r")
    open(os.path.join(SIM, "Table_Sim_Level.tex"), "w").write(txt)

    small = d[(d.n == d.n.min()) & (d.B == d.B.min())]
    large = d[(d.n == d.n.max()) & (d.B == d.B.max())]
    macros["LamBestSmall"] = fmt(float(small.lam_best.median()), 2)
    macros["LamBestLarge"] = fmt(float(large.lam_best.median()), 2)
    macros["LamTreeTypical"] = fmt(float(d.lam_tree.median()), 2)
    return d


# ---------------------------------------------------------------------------
#  Benchmark suite
# ---------------------------------------------------------------------------

def _dataset_shapes():
    """Marginal response variance, sample size and dimension per dataset."""
    import glob
    out = {}
    for path in glob.glob(os.path.join(ROOT, "outputs", "data", "*.csv")):
        name = os.path.splitext(os.path.basename(path))[0]
        df = pd.read_csv(path).dropna()
        if "y" in df.columns:
            out[name] = dict(var=float(np.var(df["y"].to_numpy(float), ddof=1)),
                             n=int(df.shape[0]), p=int(df.shape[1] - 1))
    try:
        from sklearn.datasets import load_diabetes
        d = load_diabetes()
        out["diabetes"] = dict(var=float(np.var(d.target, ddof=1)),
                               n=int(d.data.shape[0]), p=int(d.data.shape[1]))
    except Exception:
        pass
    return out


def benchmarks(macros):
    """Merge the two benchmark runs into the tables the manuscript reads.

    The gain columns are the paired median percentage change from the forest
    the shrinkage was applied to.  Two diagnostic columns accompany them,
    because without those the table records an outcome without recording the
    condition the theory says decides it: the share of response variance the
    forest explains, which stands in for the forest-level signal-to-noise
    ratio, and the mean fitted shrinkage weight.
    """
    paths = [os.path.join(ROOT, "outputs", "real", "bench_raw.csv"),
             os.path.join(ROOT, "outputs", "real_py", "bench_raw.csv")]
    frames = []
    for p in paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            # the two runs name the fitted weight differently; harmonise before
            # concatenating, or the rename produces two columns of one name
            df = df.rename(columns={"lambda": "lam", "lambda_naive": "lam_naive",
                                    "split": "rep"})
            df["source"] = "R" if os.sep + "real" + os.sep in p else "Python"
            frames.append(df)
    if not frames:
        return None
    d = pd.concat(frames, ignore_index=True)
    if "rep" not in d.columns:
        d["rep"] = 0
    shapes = _dataset_shapes()

    rows = []
    for (src, ds, B), g in d.groupby(["source", "dataset", "B"]):
        w = g.pivot_table(index="rep", columns="method", values="mse")
        if "RF" not in w or "SSRF-JS" not in w:
            continue
        w = w.dropna(subset=["RF", "SSRF-JS"])
        if len(w) < 5:
            continue
        diff = (w["RF"] - w["SSRF-JS"]).to_numpy(float)
        _, pv, _ = E.wilcoxon_win(diff)
        sh = shapes.get(ds, {})
        vy = sh.get("var", np.nan)
        rows.append(dict(
            source=src, dataset=ds, B=int(B),
            n=int(sh.get("n", 0)), p=int(sh.get("p", 0)),
            r2=1.0 - float(w["RF"].median()) / vy if np.isfinite(vy) else np.nan,
            lam=float(pd.to_numeric(g["lam"], errors="coerce").mean())
            if "lam" in g else np.nan,
            gain=float(np.median(100 * (w["SSRF-JS"] / w["RF"] - 1))),
            win=100 * float(np.mean(diff > 0)), pval=pv, reps=len(w)))
    s = pd.DataFrame(rows)
    if s.empty:
        return None
    s["p_holm"] = np.nan
    for key, idx in s.groupby(["source", "dataset"]).groups.items():
        s.loc[idx, "p_holm"] = E.holm(s.loc[idx, "pval"].values)
    s.to_csv(os.path.join(REAL, "bench_summary_merged.csv"), index=False)

    src = "R" if (s.source == "R").any() else "Python"
    blk = s[s.source == src]
    Bs = sorted(blk.B.unique())
    body = []
    for ds in sorted(blk.dataset.unique(),
                     key=lambda z: blk[blk.dataset == z].n.iloc[0]):
        r = blk[blk.dataset == ds].set_index("B").reindex(Bs)
        n = int(blk[blk.dataset == ds].n.iloc[0])
        body.append(
            f"\\texttt{{{ds}}} & {n} & {fmt(float(r.r2.median()), 2)} & "
            f"{fmt(float(r.lam.mean()), 2)} & "
            + " & ".join(fmt(v, 1) for v in r.gain)
            + f" & {fmt(float(np.nanmin(r.p_holm.values)), 3)} \\\\")
    reps = int(blk.reps.max())
    txt = table(
        body,
        "Dataset & $n$ & $R^2$ & $\\bar{\\hat\\lambda}$ & "
        + " & ".join(f"$B{{=}}{b}$" for b in Bs) + " & Holm $p$",
        (f"Public benchmark suite, {reps} random 70/30 splits per dataset with "
         "the same splits used by every method. The gain columns give the "
         "median percentage change in out-of-sample mean squared error when "
         "the leaf means are replaced by their James--Stein shrunken values, "
         "so negative entries favour shrinkage. $R^2$ is the share of the "
         "response variance the unshrunken forest explains and stands in for "
         "the forest-level signal-to-noise ratio; $\\bar{\\hat\\lambda}$ is the "
         "mean fitted shrinkage weight. The last column is the smallest "
         "Holm-adjusted $p$-value across ensemble sizes for the one-sided "
         "paired Wilcoxon signed-rank test that shrinkage lowers the error."),
        "tab:bench_gain", "lrrr" + "r" * len(Bs) + "r")
    open(os.path.join(REAL, "Table_Bench_Gain.tex"), "w").write(txt)

    macros["BenchCount"] = str(int(blk.dataset.nunique()))
    macros["BenchSplits"] = str(reps)
    macros["BenchMedianR"] = fmt(float(blk.r2.median()), 2)
    macros["BenchWinsSmallB"] = str(int((blk[blk.B == min(Bs)].gain < 0).sum()))
    macros["BenchWinsLargeB"] = str(int((blk[blk.B == max(Bs)].gain < 0).sum()))
    macros["BenchLambda"] = fmt(float(blk.lam.mean()), 2)
    return s


# ---------------------------------------------------------------------------
#  Experiments F to H: calibration, jackknife reliability, anchoring
# ---------------------------------------------------------------------------

CAL_METHODS = ["RF", "SSRF-JS", "Cal-CV", "Cal-CV-local", "Cal-IJ"]


def calibration(macros):
    path = os.path.join(SIM, "sim_calibration.csv")
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path)
    present = [m for m in CAL_METHODS if m in set(d.method)]
    body = []
    for dgp in sorted(d.dgp.unique()):
        for n in sorted(d.n.unique()):
            for B in sorted(d.B.unique()):
                s = d[(d.dgp == dgp) & (d.n == n) & (d.B == B)]
                if s.empty:
                    continue
                v = [float(s[s.method == m].mse_true.median()) for m in present]
                cells = [fmt(x, 4) for x in v]
                if np.isfinite(v).any():
                    cells[int(np.nanargmin(v))] = \
                        "\\textbf{" + cells[int(np.nanargmin(v))] + "}"
                lt = float(s.lam_tree.mean())
                lc = float(s.lam_cv.mean())
                lo = float(s.lam_oracle.mean())
                body.append(f"{DGP_LABEL[dgp]} & {n} & {B} & "
                            + " & ".join(cells)
                            + f" & {fmt(lt, 2)} & {fmt(lc, 2)} & {fmt(lo, 2)}"
                            + " \\\\")
        body.append("\\addlinespace")
    reps = int(d.rep.nunique())
    txt = table(
        body[:-1],
        "Process & $n$ & $B$ & "
        + " & ".join(m.replace("-", "--") for m in present)
        + " & $\\bar{\\hat\\lambda}_{\\mathrm{tree}}$ & $\\hat\\lambda_{\\mathrm{cv}}$"
          " & $\\lambda^\\star$",
        (f"Median test mean squared error against the true conditional mean, "
         f"$R={reps}$ replications, minimum leaf size $5$, $p=10$. The three "
         "rightmost columns give the mean factor the leaf-level rule produces, "
         "the mean cross-validated estimate of the forest-level factor, and "
         "the factor a grid search against the true conditional mean would "
         "have chosen. Cal--CV and Cal--IJ contract the forest prediction by a "
         "single calibrated factor, estimated by cross-validation and by the "
         "out-of-bag route; Cal--CV--local varies that factor with the local "
         "prediction variance."),
        "tab:calibration", "lrr" + "r" * len(present) + "rrr")
    open(os.path.join(SIM, "Table_Sim_Calibration.tex"), "w").write(txt)

    small = d[d.B == d.B.min()]
    large = d[d.B == d.B.max()]
    for tag, blk in (("Small", small), ("Large", large)):
        rf = blk[blk.method == "RF"].mse_true.median()
        for m, key in (("SSRF-JS", "Js"), ("Cal-CV", "Cal")):
            if m in set(blk.method):
                v = blk[blk.method == m].mse_true.median()
                macros[f"CalGain{key}B{tag}"] = fmt(-100 * (v / rf - 1), 1)
    # how the calibrated rule compares with the leaf-level rule, configuration
    # by configuration, at the smallest ensemble
    rel, wins = [], 0
    for (dgp, n), blk in small.groupby(["dgp", "n"]):
        a = blk[blk.method == "SSRF-JS"].mse_true.median()
        c = blk[blk.method == "Cal-CV"].mse_true.median()
        if np.isfinite(a) and np.isfinite(c) and a > 0:
            rel.append(100 * (c / a - 1))
            wins += int(c < a)
    if rel:
        macros["CalVsJsSmallB"] = fmt(-float(np.median(rel)), 1)
        macros["CalWinsSmallB"] = str(wins)
        macros["CalConfigsSmallB"] = str(len(rel))
    macros["LamCvSmallB"] = fmt(float(small.lam_cv.mean()), 2)
    macros["LamCvLargeB"] = fmt(float(large.lam_cv.mean()), 2)
    macros["LamTreeTypicalTwo"] = fmt(float(d.lam_tree.mean()), 2)
    return d


def ij(macros):
    path = os.path.join(SIM, "sim_ij.csv")
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path)
    Bs = sorted(d.B.unique())
    body = []
    for dgp in sorted(d.dgp.unique()):
        s = d[d.dgp == dgp].set_index("B").reindex(Bs)
        body.append(f"{DGP_LABEL[dgp]} ratio & "
                    + " & ".join(fmt(v, 2) for v in s.ratio) + " \\\\")
        body.append(f"{DGP_LABEL[dgp]} correlation & "
                    + " & ".join(fmt(v, 2) for v in s["corr"]) + " \\\\")
    n = int(d.n.iloc[0])
    txt = table(
        body,
        "Process & " + " & ".join(f"$B{{=}}{b}$" for b in Bs),
        ("Reliability of the bias-corrected infinitesimal jackknife as the "
         f"ensemble grows, at $n={n}$, $p=10$. The ratio is the mean jackknife "
         "estimate divided by the variance of the forest prediction over "
         "independent training samples, so one is exact; the correlation is "
         "taken across test points between the two. The estimate is unusable "
         "at small ensembles and becomes trustworthy as $B$ approaches $n$, "
         "which is why the cross-validated calibration is the one recommended "
         "in Section~\\ref{subsec:calibration}."),
        "tab:ij", "l" + "r" * len(Bs))
    open(os.path.join(SIM, "Table_Sim_IJ.tex"), "w").write(txt)
    small = d[d.B == min(Bs)]
    large = d[d.B == max(Bs)]
    macros["IjRatioSmall"] = fmt(float(small.ratio.median()), 1)
    macros["IjCorrSmall"] = fmt(float(small["corr"].median()), 2)
    macros["IjRatioLarge"] = fmt(float(large.ratio.median()), 2)
    macros["IjCorrLarge"] = fmt(float(large["corr"].median()), 2)
    return d


ANCHOR_ORDER = ["RF", "tree", "depth 1", "depth 2", "depth 3", "selected",
                "oracle depth"]


def anchor(macros):
    path = os.path.join(SIM, "sim_anchor.csv")
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path)
    present = [m for m in ANCHOR_ORDER if m in set(d.method)]
    body = []
    for dgp in sorted(d.dgp.unique()):
        for n in sorted(d.n.unique()):
            for B in sorted(d.B.unique()):
                s = d[(d.dgp == dgp) & (d.n == n) & (d.B == B)]
                if s.empty:
                    continue
                v = [float(s[s.method == m].mse_true.median()) for m in present]
                cells = [fmt(x, 4) for x in v]
                # the oracle column is not attainable, so it is excluded from
                # the comparison the bold face marks
                cmp_v = [x if m != "oracle depth" else np.inf
                         for x, m in zip(v, present)]
                if np.isfinite(cmp_v).any():
                    cells[int(np.nanargmin(cmp_v))] = \
                        "\\textbf{" + cells[int(np.nanargmin(cmp_v))] + "}"
                body.append(f"{DGP_LABEL[dgp]} & {n} & {B} & "
                            + " & ".join(cells) + " \\\\")
        body.append("\\addlinespace")
    reps = int(d.rep.nunique())
    txt = table(
        body[:-1],
        "Process & $n$ & $B$ & "
        + " & ".join(m.replace("depth ", "depth~") for m in present),
        (f"Anchoring ablation, $R={reps}$ replications, minimum leaf size $5$, "
         "$p=10$, median test mean squared error against the true conditional "
         "mean. The column marked tree shrinks every leaf toward one "
         "tree-level anchor; the depth columns shrink toward the anchor of the "
         "group of leaves sharing an ancestor at that depth. The selected "
         "column chooses among the same candidates by out-of-bag error, which "
         "costs one pass over the leaves per candidate because no tree is "
         "regrown. The oracle column takes the best depth in hindsight and is "
         "excluded from the bold face, being unattainable."),
        "tab:anchor", "lrr" + "r" * len(present))
    open(os.path.join(SIM, "Table_Sim_Anchor.tex"), "w").write(txt)

    tree = d[d.method == "tree"].mse_true.median()
    sel = d[d.method == "selected"].mse_true.median()
    orc = d[d.method == "oracle depth"].mse_true.median()
    macros["AnchorSelGain"] = fmt(-100 * (sel / tree - 1), 1)
    macros["AnchorOracleGain"] = fmt(-100 * (orc / tree - 1), 1)
    if "chosen" in d.columns:
        ch = d[d.method == "selected"].chosen.dropna()
        if len(ch):
            macros["AnchorModal"] = str(ch.mode().iloc[0]).replace(
                "depth ", "depth~")
    return d


def heterogeneity(macros):
    path = os.path.join(SIM, "sim_heterogeneity.csv")
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path)
    Bs = sorted(d.B.unique())
    body = []
    for rule in sorted(d.rule.unique()):
        for n in sorted(d.n.unique()):
            s = (d[(d.rule == rule) & (d.n == n)]
                 .groupby("B")[["rel_C", "s_lambda", "mse_true",
                                "mse_common", "identity_err"]].mean()
                 .reindex(Bs))
            gap = 100 * (s.mse_true / s.mse_common - 1)
            body.append(f"{rule} & {n} & "
                        + " & ".join(fmt(100 * v, 2) for v in s.rel_C)
                        + " & " + " & ".join(fmt(v, 3) for v in s.s_lambda)
                        + " & " + " & ".join(fmt(v, 2) for v in gap)
                        + " \\\\")
        body.append("\\addlinespace")
    k = len(Bs)
    group = ("& & \\multicolumn{%d}{c}{$100\\,|C|$ relative to the deviation} "
             "& \\multicolumn{%d}{c}{across-tree $s_\\lambda$} "
             "& \\multicolumn{%d}{c}{risk gap (\\%%)} \\\\\n"
             "\\cmidrule(lr){3-%d}\\cmidrule(lr){%d-%d}\\cmidrule(lr){%d-%d}\n"
             % (k, k, k, 2 + k, 3 + k, 2 + 2 * k, 3 + 2 * k, 2 + 3 * k))
    txt = table(
        body[:-1],
        group
        + "Rule & $n$ & "
        + " & ".join(f"$B{{=}}{b}$" for b in Bs)
        + " & " + " & ".join(f"$B{{=}}{b}$" for b in Bs)
        + " & " + " & ".join(f"$B{{=}}{b}$" for b in Bs),
        ("How far the rule in use sits from the common-factor forest of "
         "Theorem~\\ref{tab:substitution}. The first block gives the "
         "across-tree covariance term of Proposition~\\ref{prop:hetero} as a "
         "percentage of the prediction's own distance from the anchor; the "
         "second gives the across-tree standard deviation of the fitted "
         "weight; the third gives the percentage difference in risk between "
         "the heterogeneous rule and its common-factor counterpart. The "
         "decomposition itself held to machine precision in every replication."),
        "tab:hetero", "lr" + "r" * (3 * len(Bs)))
    open(os.path.join(SIM, "Table_Sim_Heterogeneity.tex"), "w").write(txt)
    js = d[d.rule == "JS"]
    eb = d[d.rule == "EB"]
    macros["HeteroRelJs"] = fmt(100 * float(js.rel_C.mean()), 1)
    macros["HeteroRelEb"] = fmt(100 * float(eb.rel_C.mean()), 1)
    macros["HeteroSlamJs"] = fmt(float(js.s_lambda.mean()), 3)
    macros["HeteroSlamEb"] = fmt(float(eb.s_lambda.mean()), 3)
    macros["HeteroRiskGapJs"] = fmt(
        100 * float((js.mse_true / js.mse_common - 1).mean()), 2)
    macros["HeteroRiskGapEb"] = fmt(
        100 * float((eb.mse_true / eb.mse_common - 1).mean()), 2)
    return d


# ---------------------------------------------------------------------------

def write_macros(macros):
    # emitted so that a later file may redefine a name an earlier one set,
    # which happens where the R and Python runs both report a quantity
    lines = [f"\\providecommand{{\\{k}}}{{}}\\renewcommand{{\\{k}}}{{{v}}}"
             for k, v in sorted(macros.items())]
    open(os.path.join(SIM, "sim_macros.tex"), "w").write("\n".join(lines) + "\n")
    os.makedirs(REAL, exist_ok=True)
    print(f"wrote {len(lines)} macros")


if __name__ == "__main__":
    os.makedirs(SIM, exist_ok=True)
    macros = {}
    for fn in (substitution, grid, extended, equivalence, level,
               calibration, ij, anchor, heterogeneity, benchmarks):
        try:
            fn(macros)
            print(f"  {fn.__name__}: done")
        except FileNotFoundError:
            print(f"  {fn.__name__}: input missing, skipped")
        except Exception as exc:                       # keep going, report
            print(f"  {fn.__name__}: {type(exc).__name__}: {exc}")
    write_macros(macros)
