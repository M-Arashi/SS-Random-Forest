"""
realdata.py
===========

Empirical evaluation of Stein-shrunken leaf estimation, Python side.

Part 1  A suite of public regression datasets. Nine of them are exported as
        CSV by ``R/realdata_code.R`` into ``outputs/data``, so that the two
        implementations are evaluated on exactly the same matrices, and the
        diabetes data distributed with scikit-learn is added. Each dataset is
        evaluated over repeated random splits and over a grid of ensemble
        sizes, which is the axis along which the theory makes a prediction.

Part 2  The gold-return panel. The live download path uses the same tickers
        as the R script. When no market data is reachable the script builds a
        series calibrated to the published moments of daily gold log returns
        and labels every output it writes as such, so that a table built from
        the substitute cannot be mistaken for one built from market data.

Usage
-----
    python realdata.py --splits 30
    python realdata.py --splits 30 --with-expensive     # adds BART, NGBoost
"""

from __future__ import annotations

import argparse
import glob
import os
import time
import warnings

import numpy as np
import pandas as pd

from ssrf import ShrunkenForest
import competitors as C
import evaluation as E

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "outputs", "data")
OUT_DIR = os.path.join(ROOT, "outputs", "real_py")
os.makedirs(OUT_DIR, exist_ok=True)

B_LIST = (5, 10, 25, 100, 500)
_NUM_WORDS = {0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five",
              6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"}


def numword(x):
    """Spell an integer with letters only.

    A LaTeX control sequence may not contain digits, so a macro named after a
    numeric setting has to spell the number out.
    """
    return "".join(_NUM_WORDS[int(c)] for c in str(int(x)))


# ---------------------------------------------------------------------------
#  Part 1: benchmark suite
# ---------------------------------------------------------------------------

def load_datasets():
    """Datasets exported by the R script, plus the scikit-learn diabetes data."""
    out = {}
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*.csv"))):
        name = os.path.splitext(os.path.basename(path))[0]
        df = pd.read_csv(path).dropna()
        if "y" not in df.columns or df.shape[0] < 25:
            continue
        y = df["y"].to_numpy(float)
        X = df.drop(columns=["y"]).to_numpy(float)
        keep = X.std(axis=0) > 0
        out[name] = (X[:, keep], y)
    try:
        from sklearn.datasets import load_diabetes
        d = load_diabetes()
        out["diabetes"] = (np.asarray(d.data, float), np.asarray(d.target, float))
    except Exception:
        pass
    return out


def bench_one(X, y, B, min_leaf, rng, with_expensive=False, train_frac=0.7):
    n = X.shape[0]
    idx = rng.permutation(n)
    n_tr = max(12, int(round(train_frac * n)))
    tr, te = idx[:n_tr], idx[n_tr:]
    if te.size < 5:
        return []
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    seed = int(rng.integers(0, 2 ** 31 - 1))

    preds, times = {}, {}
    t0 = time.perf_counter()
    F = ShrunkenForest(n_estimators=B, min_samples_leaf=min_leaf,
                       random_state=seed).fit(Xtr, ytr)
    tf = time.perf_counter() - t0
    preds["RF"] = F.predict(Xte, "rf")
    preds["SSRF-JS"] = F.predict(Xte, "ssrf_js")
    preds["SSRF"] = F.predict(Xte, "ssrf")
    for k in ("RF", "SSRF-JS", "SSRF"):
        times[k] = tf
    lam = float(np.mean(F.mean_lambda("ssrf_js")))
    leaves = float(np.mean(F.leaf_counts()))
    n_eff = float(np.mean(F.effective_leaf_sizes()))

    # the estimator as specified before the leaf-size and degrees-of-freedom
    # corrections, kept so that the size of that error can be reported
    t0 = time.perf_counter()
    Fn = ShrunkenForest(n_estimators=B, min_samples_leaf=min_leaf,
                        tau_method="mom", effective_size=False,
                        random_state=seed).fit(Xtr, ytr)
    preds["SSRF-naive"] = Fn.predict(Xte, "ssrf")
    times["SSRF-naive"] = time.perf_counter() - t0
    lam_naive = float(np.mean(Fn.mean_lambda("ssrf")))

    t0 = time.perf_counter()
    Fr = ShrunkenForest(n_estimators=B, min_samples_leaf=min_leaf,
                        sigma_method="huber", random_state=seed).fit(Xtr, ytr)
    preds["SSRF-rob"] = Fr.predict(Xte, "ssrf_js")
    times["SSRF-rob"] = time.perf_counter() - t0

    if B >= 25 and Xtr.shape[1] >= 2 and n_tr >= 40:
        t0 = time.perf_counter()
        pr_t, best = C.rf_tuned(Xtr, ytr, Xte, B, seed)
        preds["RF-tuned"] = pr_t
        times["RF-tuned"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        Ft = ShrunkenForest(n_estimators=B, min_samples_leaf=best[0],
                            max_features=best[1], random_state=seed).fit(Xtr, ytr)
        preds["SSRF-tuned"] = Ft.predict(Xte, "ssrf_js")
        times["SSRF-tuned"] = times["RF-tuned"] + (time.perf_counter() - t0)
        for name, fn in (("LassoRF", C.lasso_rf),):
            t0 = time.perf_counter()
            preds[name] = fn(Xtr, ytr, Xte, B, min_leaf, seed)
            times[name] = time.perf_counter() - t0
        t0 = time.perf_counter()
        preds["GBM"] = C.gbm(Xtr, ytr, Xte, max(B, 300), seed)
        times["GBM"] = time.perf_counter() - t0
        if with_expensive:
            t0 = time.perf_counter()
            llf = C.LocalLinearForest(n_estimators=max(100, B),
                                      min_samples_leaf=max(min_leaf, 5),
                                      random_state=seed).fit(Xtr, ytr)
            preds["LLF"] = llf.predict(Xte)
            times["LLF"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            pb = C.bart(Xtr, ytr, Xte, n_draws=1000, n_burn=1000,
                        seed=seed)
            if pb is not None:
                preds["BART"], times["BART"] = pb, time.perf_counter() - t0
            t0 = time.perf_counter()
            pn = C.ngboost(Xtr, ytr, Xte, n_estimators=400, seed=seed)
            if pn is not None:
                preds["NGBoost"] = pn[0]
                times["NGBoost"] = time.perf_counter() - t0

    return [dict(method=k, mse=E.mse(v, yte), mae=E.mae(v, yte),
                 runtime=times.get(k, np.nan), lam=lam, lam_naive=lam_naive,
                 leaves=leaves, n_eff=n_eff, n_train=n_tr)
            for k, v in preds.items()]


def run_benchmarks(splits, min_leaf=5, with_expensive=False, seed0=7):
    data = load_datasets()
    if not data:
        raise SystemExit(
            f"no benchmark CSVs found in {DATA_DIR}; run R/realdata_code.R first")
    rows = []
    for name, (X, y) in data.items():
        t0 = time.perf_counter()
        for s in range(splits):
            rng = np.random.default_rng(seed0 + 9176 * s + len(name))
            for B in B_LIST:
                for r in bench_one(X, y, B, min_leaf, rng,
                                   with_expensive and B == max(B_LIST)):
                    rows.append(dict(dataset=name, n=X.shape[0], p=X.shape[1],
                                     split=s, B=B, **r))
        print(f"  {name:12s} n={X.shape[0]:4d} p={X.shape[1]:2d} "
              f"in {time.perf_counter()-t0:5.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(
            os.path.join(OUT_DIR, "bench_raw.csv"), index=False)
    return pd.DataFrame(rows)


def summarise_benchmarks(df):
    w = df.pivot_table(index=["dataset", "n", "p", "B", "split"],
                       columns="method", values="mse").reset_index()
    rows = []
    for (ds, n, p, B), g in w.groupby(["dataset", "n", "p", "B"]):
        if "SSRF-JS" not in g or "RF" not in g:
            continue
        d = (g["RF"] - g["SSRF-JS"]).to_numpy(float)
        _, pv, _ = E.wilcoxon_win(d)
        rows.append(dict(
            dataset=ds, n=int(n), n_features=int(p), B=int(B),
            gain_js=float(np.median(100 * (g["SSRF-JS"] / g["RF"] - 1))),
            gain_eb=float(np.median(100 * (g["SSRF"] / g["RF"] - 1)))
            if "SSRF" in g else np.nan,
            gain_naive=float(np.median(100 * (g["SSRF-naive"] / g["RF"] - 1)))
            if "SSRF-naive" in g else np.nan,
            win=100 * float(np.mean(d > 0)), pval=pv, splits=len(g)))
    s = pd.DataFrame(rows)
    s["p_holm"] = np.nan
    for ds, idx in s.groupby("dataset").groups.items():
        s.loc[idx, "p_holm"] = E.holm(s.loc[idx, "pval"].values)
    s.to_csv(os.path.join(OUT_DIR, "bench_summary.csv"), index=False)
    return s


def write_bench_tables(s, df):
    def fmt(x, d=1):
        return "---" if x is None or not np.isfinite(x) else f"{x:.{d}f}"

    Bs = sorted(s.B.unique())
    body = []
    for ds in sorted(s.dataset.unique(), key=lambda z: s[s.dataset == z].n.iloc[0]):
        r = s[s.dataset == ds].set_index("B").reindex(Bs)
        n = int(s[s.dataset == ds].n.iloc[0])
        p = int(s[s.dataset == ds].n_features.iloc[0])
        best_p = np.nanmin(r.p_holm.values) if np.isfinite(r.p_holm).any() else np.nan
        body.append(f"\\texttt{{{ds}}} & {n} & {p} & "
                    + " & ".join(fmt(v) for v in r.gain_js)
                    + f" & {fmt(best_p, 3)} \\\\")
    splits = int(s.splits.max())
    txt = ("\\begin{table}[htbp]\n\\centering\n"
           "\\caption{Median percentage change in out-of-sample mean squared "
           "error on public regression datasets when the leaf means of a "
           "fitted forest are replaced by their James--Stein shrunken values, "
           f"over {splits} random 70/30 splits per dataset with the same "
           "splits used by every method. Negative entries favour shrinkage. "
           "The last column is the smallest Holm-adjusted $p$-value across "
           "ensemble sizes for the one-sided paired Wilcoxon signed-rank test "
           "that shrinkage lowers the error.}\n"
           "\\label{tab:bench_gain}\n\\small\n\\setlength{\\tabcolsep}{4pt}\n"
           "\\begin{tabular}{lrr" + "r" * len(Bs) + "r}\n\\toprule\n"
           "Dataset & $n$ & $p$ & "
           + " & ".join(f"$B{{=}}{b}$" for b in Bs)
           + " & Holm $p$ \\\\\n\\midrule\n"
           + "\n".join(body)
           + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    open(os.path.join(OUT_DIR, "Table_Bench_Gain.tex"), "w").write(txt)

    # method-level comparison at the smallest and largest ensemble
    order = ["RF", "SSRF-JS", "SSRF", "SSRF-rob", "SSRF-naive", "RF-tuned",
             "SSRF-tuned", "LassoRF", "GBM", "LLF", "BART", "NGBoost"]
    present = [m for m in order if m in set(df.method)]
    body = []
    for B in (min(B_LIST), max(B_LIST)):
        for ds in sorted(df.dataset.unique(),
                         key=lambda z: df[df.dataset == z].n.iloc[0]):
            g = df[(df.dataset == ds) & (df.B == B)]
            base = g[g.method == "RF"].mse.median()
            if not np.isfinite(base):
                continue
            vals = [g[g.method == m].mse.median() / base if (g.method == m).any()
                    else np.nan for m in present]
            cells = [fmt(100 * (v - 1), 1) if np.isfinite(v) else "---"
                     for v in vals]
            if np.isfinite(vals).any():
                j = int(np.nanargmin(vals))
                cells[j] = "\\textbf{" + cells[j] + "}"
            body.append(f"{B} & \\texttt{{{ds}}} & " + " & ".join(cells) + " \\\\")
        body.append("\\addlinespace")
    txt = ("\\begin{table}[htbp]\n\\centering\n"
           "\\caption{Median out-of-sample mean squared error on the "
           "benchmark suite, as a percentage change from the standard forest "
           "of the same size. Negative entries beat the standard forest; bold "
           "marks the best method in each row. Entries are omitted where a "
           "method is not defined for the dataset, which happens for the "
           "penalised and local linear competitors when the design has a "
           "single column or very few rows.}\n"
           "\\label{tab:bench_methods}\n\\small\n\\setlength{\\tabcolsep}{3pt}\n"
           "\\begin{tabular}{rl" + "r" * len(present) + "}\n\\toprule\n"
           "$B$ & Dataset & "
           + " & ".join(m.replace("-", "--") for m in present)
           + " \\\\\n\\midrule\n" + "\n".join(body[:-1])
           + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    open(os.path.join(OUT_DIR, "Table_Bench_Methods.tex"), "w").write(txt)

    macros = {
        "BenchCount": str(int(s.dataset.nunique())),
        "BenchSplits": str(splits),
        "BenchWinsSmallB": str(int((s[s.B == min(B_LIST)].gain_js < 0).sum())),
        "BenchWinsLargeB": str(int((s[s.B == max(B_LIST)].gain_js < 0).sum())),
        "BenchMedianGainSmallB": fmt(-s[s.B == min(B_LIST)].gain_js.median(), 1),
        "BenchMedianGainLargeB": fmt(-s[s.B == max(B_LIST)].gain_js.median(), 1),
        "BenchLambdaNaive": fmt(float(df.lam_naive.mean()), 3),
        "BenchLambdaCorrected": fmt(float(df.lam.mean()), 3),
    }
    open(os.path.join(OUT_DIR, "bench_macros.tex"), "w").write(
        "\n".join(
            f"\\providecommand{{\\{k}}}{{}}\\renewcommand{{\\{k}}}{{{v}}}"
            for k, v in macros.items()) + "\n")
    return macros


# ---------------------------------------------------------------------------
#  Part 2: gold-return panel
# ---------------------------------------------------------------------------

def download_gold_panel():
    """Live download, mirroring the tickers used by the R script."""
    try:
        import yfinance as yf
    except Exception:
        return None
    try:
        g = yf.download("GC=F", start="2005-01-03", progress=False)
        fx = yf.download("EURUSD=X", start="2005-01-03", progress=False)
        if g is None or len(g) == 0 or fx is None or len(fx) == 0:
            return None
        df = pd.DataFrame({
            "date": g.index,
            "gold": np.asarray(g["Close"]).ravel(),
            "high": np.asarray(g["High"]).ravel(),
            "low": np.asarray(g["Low"]).ravel(),
        }).set_index("date")
        df["eur"] = pd.Series(np.asarray(fx["Close"]).ravel(),
                              index=fx.index).reindex(df.index).ffill()
        df["dgs10"] = np.nan
        df["dgs2"] = np.nan
        return df.dropna().reset_index()
    except Exception:
        return None


def simulate_gold_panel(n_days=5023, seed=20240101):
    """Series calibrated to the published moments of daily gold log returns.

    Excess kurtosis near four, mild negative skewness, annualised volatility
    near eighteen per cent and GARCH(1,1) persistence near 0.99. Any output
    built from this series is labelled as a substitute.
    """
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2005-01-03", periods=n_days)
    a0, a1, b1 = 0.02, 0.07, 0.92
    s2 = np.empty(n_days)
    s2[0] = a0 / (1 - a1 - b1)
    z = rng.standard_t(5, n_days) / np.sqrt(5 / 3)
    r = np.empty(n_days)
    r[0] = np.sqrt(s2[0]) * z[0]
    for t in range(1, n_days):
        s2[t] = a0 + a1 * r[t - 1] ** 2 + b1 * s2[t - 1]
        r[t] = np.sqrt(s2[t]) * z[t]
    target = 0.18 / np.sqrt(252)
    r *= target / r.std()
    r[r < -3 * target] *= 1.15
    r = 0.95 * r + 0.05 * np.concatenate([[0.0], r[:-1]])
    e = rng.normal(0, 0.07 / np.sqrt(252), n_days)
    e = -0.30 * r + np.sqrt(1 - 0.09) * e
    e *= (0.07 / np.sqrt(252)) / e.std()
    P = 435 * np.exp(np.cumsum(r))
    rng_band = 1.5 * np.sqrt(s2)
    return pd.DataFrame(dict(
        date=days, gold=P, high=P * np.exp(0.5 * rng_band),
        low=P * np.exp(-0.5 * rng_band), eur=1.32 * np.exp(np.cumsum(e)),
        dgs10=np.maximum(0.1, np.cumsum(rng.normal(0, 0.05, n_days)) + 4),
        dgs2=np.maximum(0.1, np.cumsum(rng.normal(0, 0.05, n_days)) + 2)))


def build_features(df):
    """The twenty-four features described in the manuscript."""
    df = df.sort_values("date").reset_index(drop=True)
    r = np.concatenate([[np.nan], np.diff(np.log(df.gold.to_numpy(float)))])
    re = np.concatenate([[np.nan], np.diff(np.log(df.eur.to_numpy(float)))])
    rg = r - re
    s = pd.Series(r)

    def roll(w, fn):
        return s.rolling(w, min_periods=int(np.ceil(0.8 * w))).apply(
            fn, raw=True).to_numpy()

    sig21 = roll(21, np.std)
    feats = pd.DataFrame({
        "r_lag0": r, "r_lag1": np.roll(r, 1), "r_lag2": np.roll(r, 2),
        "r_lag3": np.roll(r, 3), "r_lag4": np.roll(r, 4),
        "sig5": roll(5, np.std), "sig10": roll(10, np.std),
        "sig21": sig21, "sig63": roll(63, np.std),
        "s_eur0": re, "s_eur1": np.roll(re, 1), "s_eur2": np.roll(re, 2),
        "r_eur_g": rg, "inter_rvol": r * sig21, "inter_svol": re * sig21,
        "dgs10_diff": np.concatenate([[np.nan],
                                      np.diff(df.dgs10.to_numpy(float))]),
        "dgs_spread": df.dgs10.to_numpy(float) - df.dgs2.to_numpy(float),
        "parkinson": (np.log(df.high.to_numpy(float))
                      - np.log(df.low.to_numpy(float))) ** 2 / (4 * np.log(2)),
        "range_norm": (df.high.to_numpy(float) - df.low.to_numpy(float))
        / df.gold.to_numpy(float),
        "skew21": roll(21, lambda z: float(pd.Series(z).skew())),
        "kurt21": roll(21, lambda z: float(pd.Series(z).kurt())),
    })
    for k in (1, 2, 3, 4):
        feats.iloc[:k, list(feats.columns).index(f"r_lag{k}")] = np.nan
    dow = pd.to_datetime(df.date).dt.dayofweek
    feats["dow_tue"] = (dow == 1).astype(int)
    feats["dow_wed"] = (dow == 2).astype(int)
    feats["dow_thu"] = (dow == 3).astype(int)
    feats["y_gold"] = 100 * np.concatenate([r[1:], [np.nan]])
    feats["y_bic"] = 100 * np.concatenate([rg[1:], [np.nan]])
    feats["date"] = df.date.to_numpy()
    return feats.dropna().reset_index(drop=True)


def rolling_origin(dat, cols, y_name, B, min_leaf=5, n_blocks=10,
                   min_train_frac=0.5, seed=0):
    """Successive test blocks, each predicted from all data preceding it.

    A single chronological split gives one draw from the distribution of
    out-of-sample performance and no way to separate a difference between
    methods from sampling variation.
    """
    n = len(dat)
    edges = np.linspace(int(min_train_frac * n), n, n_blocks + 1).astype(int)
    preds, truth, dates = {}, [], []
    for k in range(n_blocks):
        tr = np.arange(edges[k])
        te = np.arange(edges[k], edges[k + 1])
        if te.size < 5:
            continue
        Xtr, ytr = dat.loc[tr, cols].to_numpy(float), dat.loc[tr, y_name].to_numpy(float)
        Xte, yte = dat.loc[te, cols].to_numpy(float), dat.loc[te, y_name].to_numpy(float)
        F = ShrunkenForest(n_estimators=B, min_samples_leaf=min_leaf,
                           random_state=seed + k).fit(Xtr, ytr)
        blk = {"RF": F.predict(Xte, "rf"),
               "SSRF-JS": F.predict(Xte, "ssrf_js"),
               "SSRF": F.predict(Xte, "ssrf"),
               "LassoRF": C.lasso_rf(Xtr, ytr, Xte, B, min_leaf, seed + k),
               "GBM": C.gbm(Xtr, ytr, Xte, max(B, 300), seed + k)}
        for m, v in blk.items():
            preds.setdefault(m, []).append(v)
        truth.append(yte)
        dates.append(dat.loc[te, "date"].to_numpy())
    return ({m: np.concatenate(v) for m, v in preds.items()},
            np.concatenate(truth), np.concatenate(dates))


def gold_table(preds, y, caption, label, note=""):
    def fmt(x, d=4):
        return "---" if x is None or not np.isfinite(x) else f"{x:.{d}f}"
    half = np.arange(len(y) // 2)
    rest = np.arange(len(y) // 2, len(y))
    ref = (preds["SSRF-JS"] - y) ** 2
    others = [m for m in preds if m != "SSRF-JS"]
    pv = E.holm([E.dm_test((preds[m] - y) ** 2, ref, h=1)[1] for m in others])
    pmap = dict(zip(others, pv))
    body = []
    for m, pr in preds.items():
        lo, hi = E.split_conformal(pr[half], y[half], pr[rest])
        isc = E.interval_score(lo, hi, y[rest])
        _, sharpe, _, _ = E.net_pnl(pr, y)
        body.append(f"{m.replace('-', '--')} & {fmt(E.mse(pr, y))} & "
                    f"{fmt(E.mae(pr, y))} & {fmt(isc, 3)} & "
                    f"{fmt(sharpe, 3)} & "
                    f"{'(ref.)' if m == 'SSRF-JS' else fmt(pmap.get(m), 3)} \\\\")
    return ("\\begin{table}[htbp]\n\\centering\n"
            f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
            "\\small\n\\setlength{\\tabcolsep}{5pt}\n"
            "\\begin{tabular}{lrrrrr}\n\\toprule\n"
            "Method & MSE & MAE & Interval score & Sharpe & Holm $p$ \\\\\n"
            "\\midrule\n" + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n"
            + (f"\\\\[2pt]\n\\parbox{{\\textwidth}}{{\\footnotesize {note}}}\n"
               if note else "") + "\\end{table}\n")


def run_gold():
    print("Attempting the market-data download ... ", end="", flush=True)
    panel = download_gold_panel()
    if panel is None or "dgs10" not in panel or panel["dgs10"].isna().all():
        print("unavailable.")
        panel = simulate_gold_panel()
        source = "calibrated substitute"
        note = ("These figures were produced from a series calibrated to the "
                "published moments of daily gold log returns, because the "
                "market-data download was unavailable in the environment that "
                "ran the script. They are not estimates from the market data.")
    else:
        print("done.")
        source = "Yahoo Finance"
        note = ""
    dat = build_features(panel)
    cols = [c for c in dat.columns if c not in ("y_gold", "y_bic", "date")]
    print(f"Panel: {len(panel)} days, features: n={len(dat)}, p={len(cols)}, "
          f"source: {source}")

    macros = {"GoldSource": source, "GoldDays": str(len(panel))}
    for B in (10, 500):
        t0 = time.perf_counter()
        preds, y, _ = rolling_origin(dat, cols, "y_gold", B)
        open(os.path.join(OUT_DIR, f"Table_Real_Gold_B{B}.tex"), "w").write(
            gold_table(preds, y,
                       "Rolling-origin out-of-sample performance for the "
                       "next-day gold log return, in basis points, with "
                       f"ensembles of $B={B}$ trees. The Sharpe ratio is that "
                       "of a sign-based strategy charged two basis points on "
                       "the day a position flips. The last column reports "
                       "Holm-adjusted Diebold--Mariano $p$-values against "
                       "SSRF--JS.", f"tab:gold_B{B}", note))
        macros[f"GoldGainB{numword(B)}"] = (
            f"{100 * (E.mse(preds['SSRF-JS'], y) / E.mse(preds['RF'], y) - 1):.2f}")
        pb, yb, _ = rolling_origin(dat, cols, "y_bic", B)
        open(os.path.join(OUT_DIR,
                          f"Table_Real_Bicurrency_B{B}.tex"), "w").write(
            gold_table(pb, yb,
                       "Rolling-origin out-of-sample performance for the "
                       "Gold/USD--EUR bicurrency spread, in basis points, "
                       f"with ensembles of $B={B}$ trees.",
                       f"tab:bic_B{B}", note))
        print(f"  B={B} done in {time.perf_counter()-t0:.0f}s", flush=True)
    return macros


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=30)
    ap.add_argument("--min-leaf", type=int, default=5)
    ap.add_argument("--with-expensive", action="store_true")
    ap.add_argument("--skip-gold", action="store_true")
    a = ap.parse_args()

    print("===== Part 1: public benchmark suite =====")
    df = run_benchmarks(a.splits, a.min_leaf, a.with_expensive)
    s = summarise_benchmarks(df)
    macros = write_bench_tables(s, df)
    print(s.to_string(index=False))

    if not a.skip_gold:
        print("\n===== Part 2: gold-return panel =====")
        macros.update(run_gold())

    with open(os.path.join(OUT_DIR, "real_macros.tex"), "w") as fh:
        fh.write("\n".join(
            f"\\providecommand{{\\{k}}}{{}}\\renewcommand{{\\{k}}}{{{v}}}"
            for k, v in macros.items()) + "\n")
    print(f"\nOutputs written to {OUT_DIR}")
