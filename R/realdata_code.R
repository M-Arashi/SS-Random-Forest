################################################################################
##  realdata_code.R
##
##  Empirical evaluation of Stein-shrunken leaf estimation.
##
##  Part 1  A suite of nine public regression datasets, all distributed with
##          base R or with mlbench and MASS, so the analysis runs offline and
##          reproduces exactly.  Sample sizes run from 30 to 506, which covers
##          the range in which the substitution between ensemble averaging and
##          leaf-level shrinkage is predicted to bind and the range in which it
##          is predicted to be exhausted.  Every dataset is evaluated over
##          repeated random splits and over a grid of ensemble sizes.
##
##  Part 2  The gold-return application.  The panel is the daily gold spot
##          price (Yahoo ticker "GC=F"), the USD/EUR rate ("EURUSD=X") and the
##          ten- and two-year Treasury constant-maturity yields (FRED series
##          "DGS10" and "DGS2"), from January 2005 onward, downloaded with
##          quantmod.  When the download is unavailable the script substitutes
##          a series calibrated to the published moments of daily gold log
##          returns and says so in every output it writes, so that a table
##          built from the substitute can never be mistaken for one built from
##          the market data.
##
##  REQUIREMENTS
##      randomForest, glmnet, gbm, matrixStats, mlbench, MASS,
##      ggplot2, dplyr, tidyr, scales, RColorBrewer, corrplot
##      quantmod (optional, only for the live download in Part 2)
##
##  USAGE
##      Rscript realdata_code.R
##      SSRF_SPLITS=100 Rscript realdata_code.R      # production run
##
##  OUTPUTS (./outputs/real/)
##      bench_raw.csv, bench_summary.csv, Table_Bench_*.tex,
##      Figure_Bench_*.pdf, gold_results.rds, Table_Real_*.tex,
##      Figure_Real_*.pdf, real_macros.tex
################################################################################

.script_dir <- function() {
  a <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(a)) return(dirname(sub("^--file=", "", a[1])))
  for (cand in c("R", ".")) if (file.exists(file.path(cand, "ssrf_core.R")))
    return(cand)
  "."
}
source(file.path(.script_dir(), "ssrf_core.R"))

suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(tidyr); library(scales)
  library(mlbench); library(MASS)
})

OUT_DIR <- file.path("outputs", "real")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(file.path("outputs", "data"), showWarnings = FALSE, recursive = TRUE)

SPLITS <- as.integer(Sys.getenv("SSRF_SPLITS", "50"))
set.seed(as.integer(Sys.getenv("SSRF_SEED", "20260910")))


## ============================================================================
##  PART 1.  Public benchmark suite
## ============================================================================

##' Load the nine benchmark datasets as numeric design matrices and responses.
##' Factors are expanded to indicator columns and incomplete rows dropped, so
##' that every method sees exactly the same matrix.
load_benchmarks <- function() {
  numify <- function(df, yname) {
    df <- df[stats::complete.cases(df), , drop = FALSE]
    y <- as.numeric(df[[yname]])
    X <- df[, setdiff(names(df), yname), drop = FALSE]
    X <- stats::model.matrix(~ . - 1, data = X)
    keep <- apply(X, 2, function(z) stats::sd(z) > 0)
    list(X = X[, keep, drop = FALSE], y = y)
  }
  out <- list()

  data(BostonHousing, package = "mlbench", envir = environment())
  out$boston <- numify(BostonHousing, "medv")

  data(Ozone, package = "mlbench", envir = environment())
  oz <- Ozone
  names(oz) <- paste0("v", seq_len(ncol(oz)))
  oz <- oz[, vapply(oz, function(z) !is.factor(z) || nlevels(z) < 15, TRUE)]
  out$ozone <- numify(oz, "v4")

  data(Servo, package = "mlbench", envir = environment())
  out$servo <- numify(Servo, "Class")

  out$cpus <- numify(MASS::cpus[, c("syct", "mmin", "mmax", "cach",
                                    "chmin", "chmax", "perf")], "perf")
  out$mcycle <- numify(MASS::mcycle, "accel")
  out$airquality <- numify(datasets::airquality, "Ozone")
  out$swiss <- numify(datasets::swiss, "Fertility")
  out$savings <- numify(datasets::LifeCycleSavings, "sr")
  out$attitude <- numify(datasets::attitude, "rating")

  ## the same matrices are written out so that the Python scripts evaluate
  ## exactly the same data
  for (nm in names(out))
    utils::write.csv(cbind(as.data.frame(out[[nm]]$X), y = out[[nm]]$y),
                     file.path("outputs", "data", paste0(nm, ".csv")),
                     row.names = FALSE)
  out
}

##' One train and test split of one dataset, at one ensemble size.
bench_one <- function(dat, B, nodesize, train_frac = 0.7) {
  n <- length(dat$y)
  idx <- sample.int(n, max(10, floor(train_frac * n)))
  te <- setdiff(seq_len(n), idx)
  Xtr <- dat$X[idx, , drop = FALSE]; ytr <- dat$y[idx]
  Xte <- dat$X[te, , drop = FALSE];  yte <- dat$y[te]

  M  <- fit_ssrf(Xtr, ytr, B = B, nodesize = nodesize, rule = "js")
  ME <- ssrf_leaf_values(M$rf, Xtr, ytr, rule = "eb")
  ML <- ssrf_leaf_values(M$rf, Xtr, ytr, rule = "eb", tau_method = "mom",
                         effective_size = FALSE)

  preds <- list(
    RF            = ssrf_predict(M$rf, M$leafvals, Xte, "raw"),
    `SSRF-JS`     = ssrf_predict(M$rf, M$leafvals, Xte, "shrunk"),
    SSRF          = ssrf_predict(M$rf, ME,         Xte, "shrunk"),
    `SSRF-naive`  = ssrf_predict(M$rf, ML,         Xte, "shrunk"))
  if (B >= 25) {
    tu <- fit_rf_tuned(Xtr, ytr, Xte, B)
    preds$`RF-tuned` <- tu$pred
    MT <- fit_ssrf(Xtr, ytr, B = B, nodesize = tu$nodesize, mtry = tu$mtry,
                   rule = "js")
    preds$`SSRF-tuned` <- ssrf_predict(MT$rf, MT$leafvals, Xte, "shrunk")
    preds$GBM     <- fit_gbm(Xtr, ytr, Xte, n_trees = max(B, 300))
    preds$LassoRF <- fit_lasso_rf(Xtr, ytr, Xte, B, nodesize)
    preds$LLF     <- fit_llf(Xtr, ytr, Xte, B = max(B, 100),
                             nodesize = max(nodesize, 5))
  }
  do.call(rbind, lapply(names(preds), function(nm)
    data.frame(method = nm, mse = mse_of(preds[[nm]], yte),
               mae = mae_of(preds[[nm]], yte),
               lambda = M$mean_lambda, lambda_naive = mean(ssrf_mean_lambda(ML)),
               leaves = M$leaves, n_eff = M$n_eff, n_train = length(idx))))
}

run_benchmarks <- function(datasets, splits = SPLITS,
                           B_list = c(5, 10, 25, 100, 500),
                           nodesize = 5) {
  rows <- list(); k <- 0
  for (nm in names(datasets)) {
    t0 <- proc.time()
    for (s in seq_len(splits)) for (B in B_list) {
      r <- bench_one(datasets[[nm]], B, nodesize)
      k <- k + 1
      rows[[k]] <- cbind(dataset = nm, split = s, B = B,
                         nodesize = nodesize, r)
    }
    cat(sprintf("  %-11s n=%4d p=%2d done in %.0fs\n", nm,
                length(datasets[[nm]]$y), ncol(datasets[[nm]]$X),
                (proc.time() - t0)["elapsed"]))
  }
  do.call(rbind, rows)
}

cat("===== Part 1: public benchmark suite =====\n")
benchmarks <- load_benchmarks()
bench <- run_benchmarks(benchmarks)
write.csv(bench, file.path(OUT_DIR, "bench_raw.csv"), row.names = FALSE)

## Paired comparison of each shrunken rule against the forest it was built on.
bench_gain <- bench %>%
  dplyr::select(dataset, B, split, method, mse) %>%
  tidyr::pivot_wider(names_from = method, values_from = mse) %>%
  dplyr::group_by(dataset, B) %>%
  dplyr::summarise(
    n_train = NA_real_,
    gain_js = median(100 * (`SSRF-JS` / RF - 1)),
    gain_eb = median(100 * (SSRF / RF - 1)),
    gain_naive = median(100 * (`SSRF-naive` / RF - 1)),
    win_js = 100 * mean(`SSRF-JS` < RF),
    wilcox_p = tryCatch(stats::wilcox.test(RF, `SSRF-JS`, paired = TRUE,
                                           alternative = "greater")$p.value,
                        error = function(e) NA_real_),
    .groups = "drop")
bench_gain$holm_p <- stats::p.adjust(bench_gain$wilcox_p, method = "holm")
sizes <- vapply(benchmarks, function(d) length(d$y), numeric(1))
bench_gain$n <- as.numeric(sizes[bench_gain$dataset])
write.csv(bench_gain, file.path(OUT_DIR, "bench_summary.csv"), row.names = FALSE)

fmt <- function(x, d = 2) ifelse(is.na(x), "---", sprintf(paste0("%.", d, "f"), x))

Bs <- sort(unique(bench_gain$B))
rows <- vapply(sort(unique(bench_gain$dataset)), function(ds) {
  g <- bench_gain[bench_gain$dataset == ds, ]
  g <- g[match(Bs, g$B), ]
  paste0("\\texttt{", ds, "} & ", sizes[[ds]], " & ",
         paste(fmt(g$gain_js, 1), collapse = " & "), " & ",
         fmt(min(g$holm_p, na.rm = TRUE), 3), " \\\\")
}, character(1))
writeLines(paste0(
  "\\begin{table}[htbp]\n\\centering\n\\caption{Median percentage change in ",
  "out-of-sample mean squared error on nine public regression datasets when ",
  "the leaf means of a fitted forest are replaced by their James--Stein ",
  "shrunken values, over ", SPLITS, " random 70/30 splits per dataset. ",
  "Negative entries favour shrinkage. The last column is the smallest ",
  "Holm-adjusted $p$-value across ensemble sizes for the one-sided paired ",
  "Wilcoxon signed-rank test that shrinkage lowers the error.}\n",
  "\\label{tab:bench_gain}\n\\small\n\\setlength{\\tabcolsep}{4pt}\n",
  "\\begin{tabular}{lr", paste(rep("r", length(Bs)), collapse = ""), "r}\n",
  "\\toprule\nDataset & $n$ & ",
  paste(sprintf("$B{=}%d$", Bs), collapse = " & "), " & Holm $p$ \\\\\n",
  "\\midrule\n", paste(rows, collapse = "\n"),
  "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"),
  file.path(OUT_DIR, "Table_Bench_Gain.tex"))

theme_pub <- theme_bw(base_size = 11) +
  theme(panel.grid.minor = element_blank(),
        strip.background = element_rect(fill = "grey92", colour = NA),
        legend.position = "bottom")

pb <- ggplot(bench_gain, aes(x = B, y = gain_js, colour = reorder(dataset, n))) +
  geom_hline(yintercept = 0, colour = "grey40", linewidth = 0.4) +
  geom_line(linewidth = 0.6) + geom_point(size = 1.8) +
  scale_x_log10(breaks = Bs) +
  labs(x = "Number of trees B (log scale)",
       y = "Change in out-of-sample MSE from shrinkage (%)",
       colour = NULL) + theme_pub
ggsave(file.path(OUT_DIR, "Figure_Bench_Gain.pdf"), pb, width = 8, height = 4.2)


## ============================================================================
##  PART 2.  Gold-return application
## ============================================================================

download_gold_panel <- function(from = "2005-01-03", to = Sys.Date()) {
  if (!requireNamespace("quantmod", quietly = TRUE)) return(NULL)
  g <- tryCatch(quantmod::getSymbols("GC=F", src = "yahoo", from = from,
                                     to = to, auto.assign = FALSE),
                error = function(e) NULL)
  if (is.null(g)) return(NULL)
  fx <- tryCatch(quantmod::getSymbols("EURUSD=X", src = "yahoo", from = from,
                                      to = to, auto.assign = FALSE),
                 error = function(e) NULL)
  d10 <- tryCatch(quantmod::getSymbols("DGS10", src = "FRED",
                                       auto.assign = FALSE),
                  error = function(e) NULL)
  d2 <- tryCatch(quantmod::getSymbols("DGS2", src = "FRED",
                                      auto.assign = FALSE),
                 error = function(e) NULL)
  if (is.null(fx) || is.null(d10) || is.null(d2)) return(NULL)
  al <- na.omit(merge(g, fx, d10, d2))
  ## quantmod replaces "=" in a ticker by ".", so the column prefixes are
  ## "GC.F" and "EURUSD.X"; the Cl/Hi/Lo accessors find the columns by suffix
  ## and are therefore insensitive to how the prefix was sanitised
  gc <- al[, grep("^GC", colnames(al))]
  data.frame(date = zoo::index(al),
             gold = as.numeric(quantmod::Cl(gc)),
             high = as.numeric(quantmod::Hi(gc)),
             low  = as.numeric(quantmod::Lo(gc)),
             eur  = as.numeric(quantmod::Cl(al[, grep("^EURUSD", colnames(al))])),
             dgs10 = as.numeric(al[, "DGS10"]),
             dgs2  = as.numeric(al[, "DGS2"]))
}

##' Substitute series calibrated to the published moments of daily gold log
##' returns: excess kurtosis near four, mild negative skewness, annualised
##' volatility near eighteen per cent, and GARCH(1,1) persistence near 0.99.
##' Any table built from this substitute is labelled as such.
simulate_gold_panel <- function(n_days = 5023) {
  start <- as.Date("2005-01-03")
  all_days <- seq(start, by = "day", length.out = ceiling(n_days * 1.45))
  dates <- all_days[!format(all_days, "%u") %in% c("6", "7")][seq_len(n_days)]

  a0 <- 0.02; a1 <- 0.07; b1 <- 0.92
  s2 <- numeric(n_days); s2[1] <- a0 / (1 - a1 - b1)
  z <- rt(n_days, 5) / sqrt(5 / 3)
  r <- numeric(n_days); r[1] <- sqrt(s2[1]) * z[1]
  for (t in 2:n_days) {
    s2[t] <- a0 + a1 * r[t - 1]^2 + b1 * s2[t - 1]
    r[t] <- sqrt(s2[t]) * z[t]
  }
  target <- 0.18 / sqrt(252)
  r <- r * target / sd(r)
  r[r < -3 * target] <- r[r < -3 * target] * 1.15
  r <- 0.95 * r + 0.05 * c(0, r[-n_days])

  e <- rnorm(n_days, sd = 0.07 / sqrt(252))
  e <- -0.30 * r + sqrt(1 - 0.09) * e
  e <- e * (0.07 / sqrt(252)) / sd(e)

  P <- 435 * exp(cumsum(r)); Pe <- 1.32 * exp(cumsum(e))
  rng <- 1.5 * sqrt(s2)
  data.frame(date = dates, gold = P, high = P * exp(0.5 * rng),
             low = P * exp(-0.5 * rng), eur = Pe,
             dgs10 = pmax(0.1, cumsum(rnorm(n_days, sd = 0.05)) + 4),
             dgs2  = pmax(0.1, cumsum(rnorm(n_days, sd = 0.05)) + 2))
}

##' The twenty-four features described in the manuscript: lagged returns,
##' multi-horizon rolling volatilities, bicurrency exposures, interactions,
##' macroeconomic indicators, higher rolling moments, range-based proxies and
##' calendar effects.
build_features <- function(df) {
  df <- df[order(df$date), ]
  n <- nrow(df)
  r <- c(NA, diff(log(df$gold)))
  re <- c(NA, diff(log(df$eur)))
  rg <- r - re

  roll <- function(x, w, f) {
    s <- rep(NA_real_, length(x))
    for (i in w:length(x)) {
      xi <- x[(i - w + 1):i]; xi <- xi[!is.na(xi)]
      if (length(xi) >= ceiling(0.8 * w)) s[i] <- f(xi)
    }
    s
  }
  skew <- function(z) { m <- mean(z); v <- mean((z - m)^2)
                        if (v > 0) mean((z - m)^3) / v^1.5 else NA }
  kurt <- function(z) { m <- mean(z); v <- mean((z - m)^2)
                        if (v > 0) mean((z - m)^4) / v^2 - 3 else NA }
  lag_n <- function(x, k) c(rep(NA_real_, k), x[seq_len(length(x) - k)])

  sig21 <- roll(r, 21, sd)
  feats <- data.frame(
    r_lag0 = r, r_lag1 = lag_n(r, 1), r_lag2 = lag_n(r, 2),
    r_lag3 = lag_n(r, 3), r_lag4 = lag_n(r, 4),
    sig5 = roll(r, 5, sd), sig10 = roll(r, 10, sd),
    sig21 = sig21, sig63 = roll(r, 63, sd),
    s_eur0 = re, s_eur1 = lag_n(re, 1), s_eur2 = lag_n(re, 2),
    r_eur_g = rg, inter_rvol = r * sig21, inter_svol = re * sig21,
    dgs10_diff = c(NA, diff(df$dgs10)), dgs_spread = df$dgs10 - df$dgs2,
    parkinson = (log(df$high) - log(df$low))^2 / (4 * log(2)),
    range_norm = (df$high - df$low) / df$gold,
    skew21 = roll(r, 21, skew), kurt21 = roll(r, 21, kurt))
  dow <- format(df$date, "%u")
  feats$dow_tue <- as.integer(dow == "2")
  feats$dow_wed <- as.integer(dow == "3")
  feats$dow_thu <- as.integer(dow == "4")

  out <- cbind(feats,
               y_gold = 100 * c(r[-1], NA),
               y_bic  = 100 * c(rg[-1], NA),
               date = df$date)
  out[stats::complete.cases(out), ]
}

cat("\n===== Part 2: gold-return application =====\n")
cat("Attempting the market-data download ... ")
live <- download_gold_panel()
if (is.null(live)) {
  cat("unavailable.\n")
  panel <- simulate_gold_panel()
  DATA_SOURCE <- "calibrated substitute"
  SOURCE_NOTE <- paste0("These figures were produced from a series ",
    "calibrated to the published moments of daily gold log returns, because ",
    "the market-data download was unavailable in the environment that ran ",
    "the script. They are not estimates from the market data.")
} else {
  cat("done.\n")
  panel <- live
  DATA_SOURCE <- "Yahoo Finance and FRED"
  SOURCE_NOTE <- ""
}
cat(sprintf("Panel: %d trading days, %s to %s, source: %s\n",
            nrow(panel), min(panel$date), max(panel$date), DATA_SOURCE))

dat <- build_features(panel)
feature_cols <- setdiff(names(dat), c("y_gold", "y_bic", "date"))
cat(sprintf("After feature construction: n = %d, p = %d\n",
            nrow(dat), length(feature_cols)))

##' Rolling-origin evaluation.  A single chronological split gives one draw
##' from the distribution of out-of-sample performance and no way to judge how
##' much of the difference between methods is sampling variation.  The sample
##' is instead cut into successive test blocks, each predicted from all data
##' that precede it, and the blocks are pooled.
rolling_origin <- function(dat, y_name, B, nodesize = 5, n_blocks = 10,
                           min_train_frac = 0.5) {
  n <- nrow(dat)
  start <- floor(min_train_frac * n)
  edges <- round(seq(start, n, length.out = n_blocks + 1))
  preds <- list(); truth <- numeric(0); dates <- as.Date(character(0))
  for (k in seq_len(n_blocks)) {
    tr <- seq_len(edges[k])
    te <- (edges[k] + 1):edges[k + 1]
    if (length(te) < 5) next
    Xtr <- as.matrix(dat[tr, feature_cols]); ytr <- dat[[y_name]][tr]
    Xte <- as.matrix(dat[te, feature_cols]); yte <- dat[[y_name]][te]
    M  <- fit_ssrf(Xtr, ytr, B = B, nodesize = nodesize, rule = "js")
    ME <- ssrf_leaf_values(M$rf, Xtr, ytr, rule = "eb")
    blk <- list(RF        = ssrf_predict(M$rf, M$leafvals, Xte, "raw"),
                `SSRF-JS` = ssrf_predict(M$rf, M$leafvals, Xte, "shrunk"),
                SSRF      = ssrf_predict(M$rf, ME,         Xte, "shrunk"),
                LassoRF   = fit_lasso_rf(Xtr, ytr, Xte, B, nodesize),
                GBM       = fit_gbm(Xtr, ytr, Xte, n_trees = max(B, 300)))
    for (nm in names(blk)) preds[[nm]] <- c(preds[[nm]], blk[[nm]])
    truth <- c(truth, yte); dates <- c(dates, dat$date[te])
  }
  list(preds = preds, y = truth, dates = dates,
       lambda = mean(ssrf_mean_lambda(M$leafvals)))
}

gold_results <- list()
for (B in c(10, 50, 500)) {
  cat(sprintf("  rolling-origin evaluation, B = %d ... ", B))
  t0 <- proc.time()
  gold_results[[as.character(B)]] <-
    list(gold = rolling_origin(dat, "y_gold", B),
         bic  = rolling_origin(dat, "y_bic",  B))
  cat(sprintf("%.0fs\n", (proc.time() - t0)["elapsed"]))
}
saveRDS(list(results = gold_results, source = DATA_SOURCE, panel = panel),
        file.path(OUT_DIR, "gold_results.rds"))

##' Out-of-sample table: accuracy, interval score, and the Diebold-Mariano
##' test of SSRF against each benchmark with the Harvey-Leybourne-Newbold
##' correction and a Holm adjustment across the benchmarks.
gold_table <- function(res, caption, label, note = "") {
  meth <- names(res$preds)
  y <- res$y
  mse <- vapply(res$preds, function(p) mse_of(p, y), numeric(1))
  mae <- vapply(res$preds, function(p) mae_of(p, y), numeric(1))
  half <- seq_len(floor(length(y) / 2))
  isc <- vapply(res$preds, function(p) {
    ci <- split_conformal(p[half], y[half], p[-half])
    interval_score(ci$lower, ci$upper, y[-half])
  }, numeric(1))
  pnl <- lapply(res$preds, function(p) net_pnl(p, y))
  others <- setdiff(meth, "SSRF-JS")
  dmp <- vapply(others, function(m)
    dm_test((res$preds[[m]] - y)^2, (res$preds$`SSRF-JS` - y)^2)["p"],
    numeric(1))
  dmp <- stats::p.adjust(dmp, method = "holm")
  rows <- vapply(meth, function(m) {
    paste0(gsub("-", "--", m), " & ", fmt(mse[[m]], 4), " & ",
           fmt(mae[[m]], 4), " & ", fmt(isc[[m]], 3), " & ",
           fmt(pnl[[m]]$sharpe, 3), " & ",
           if (m == "SSRF-JS") "(ref.)" else fmt(dmp[[m]], 3), " \\\\")
  }, character(1))
  paste0("\\begin{table}[htbp]\n\\centering\n\\caption{", caption, "}\n",
         "\\label{", label, "}\n\\small\n\\setlength{\\tabcolsep}{5pt}\n",
         "\\begin{tabular}{lrrrrr}\n\\toprule\n",
         "Method & MSE & MAE & Interval score & Sharpe & Holm $p$ \\\\\n",
         "\\midrule\n", paste(rows, collapse = "\n"),
         "\n\\bottomrule\n\\end{tabular}\n",
         if (nzchar(note))
           paste0("\\\\[2pt]\n\\parbox{\\textwidth}{\\footnotesize ", note, "}\n")
         else "",
         "\\end{table}\n")
}

for (B in names(gold_results)) {
  writeLines(gold_table(gold_results[[B]]$gold,
    sprintf(paste0("Rolling-origin out-of-sample performance for the ",
                   "next-day gold log return, in basis points, with ",
                   "ensembles of $B=%s$ trees. The Sharpe ratio is that of a ",
                   "sign-based strategy net of two basis points per position ",
                   "flip. The last column reports Holm-adjusted ",
                   "Diebold--Mariano $p$-values against SSRF--JS."), B),
    sprintf("tab:gold_B%s", B), SOURCE_NOTE),
    file.path(OUT_DIR, sprintf("Table_Real_Gold_B%s.tex", B)))
  writeLines(gold_table(gold_results[[B]]$bic,
    sprintf(paste0("Rolling-origin out-of-sample performance for the ",
                   "Gold/USD--EUR bicurrency spread, in basis points, with ",
                   "ensembles of $B=%s$ trees."), B),
    sprintf("tab:bic_B%s", B), SOURCE_NOTE),
    file.path(OUT_DIR, sprintf("Table_Real_Bicurrency_B%s.tex", B)))
}

## Descriptive figures
y_all <- 100 * diff(log(panel$gold))
f1 <- ggplot(data.frame(r = y_all), aes(x = r)) +
  geom_histogram(aes(y = after_stat(density)), bins = 60, fill = "#377EB8",
                 colour = "white", alpha = 0.85) +
  stat_function(fun = dnorm, args = list(mean = mean(y_all), sd = sd(y_all)),
                colour = "red", linewidth = 0.7, linetype = 2) +
  labs(x = "Daily gold log return (basis points)", y = "Density") + theme_pub
ggsave(file.path(OUT_DIR, "Figure_Real_Histogram.pdf"), f1,
       width = 6.5, height = 4.2)

## Genuine permutation importance under each leaf rule.  Carrying the standard
## forest's importances across and rescaling them would say nothing about
## whether shrinkage changes which features matter.
perm_importance <- function(rf, leafvals, X, y, values, n_repeats = 5) {
  base <- mse_of(ssrf_predict(rf, leafvals, X, values), y)
  vapply(seq_len(ncol(X)), function(j) {
    acc <- 0
    for (rp in seq_len(n_repeats)) {
      Xp <- X; Xp[, j] <- Xp[sample.int(nrow(X)), j]
      acc <- acc + mse_of(ssrf_predict(rf, leafvals, Xp, values), y)
    }
    acc / n_repeats - base
  }, numeric(1))
}

ntr <- floor(0.8 * nrow(dat))
Xtr <- as.matrix(dat[seq_len(ntr), feature_cols]); ytr <- dat$y_gold[seq_len(ntr)]
Xte <- as.matrix(dat[-seq_len(ntr), feature_cols]); yte <- dat$y_gold[-seq_len(ntr)]
Mv <- fit_ssrf(Xtr, ytr, B = 200, nodesize = 5, rule = "js")
imp <- data.frame(
  feature = colnames(Xtr),
  RF   = perm_importance(Mv$rf, Mv$leafvals, Xte, yte, "raw"),
  SSRF = perm_importance(Mv$rf, Mv$leafvals, Xte, yte, "shrunk"))
write.csv(imp, file.path(OUT_DIR, "gold_importance.csv"), row.names = FALSE)
impl <- imp %>% tidyr::pivot_longer(-feature, names_to = "rule",
                                    values_to = "importance") %>%
  dplyr::group_by(rule) %>%
  dplyr::mutate(importance = importance / max(importance) * 100) %>%
  dplyr::ungroup()
top <- imp$feature[order(imp$RF, decreasing = TRUE)][1:15]
impl <- impl[impl$feature %in% top, ]
impl$feature <- factor(impl$feature, levels = top)
f2 <- ggplot(impl, aes(x = feature, y = importance, fill = rule)) +
  geom_col(position = "dodge", colour = "black", linewidth = 0.2) +
  scale_fill_brewer(palette = "Set1") +
  labs(x = NULL, y = "Permutation importance (scaled to 100)", fill = NULL) +
  theme_pub + theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggsave(file.path(OUT_DIR, "Figure_Real_Importance.pdf"), f2,
       width = 9, height = 4.5)

## Macros for the manuscript, so that quoted numbers cannot drift from tables
gm <- gold_results[["500"]]$gold
gm10 <- gold_results[["10"]]$gold
## emitted so that a later macro file may redefine a name an earlier one set,
## which happens where the R and Python runs both report a quantity
nc <- function(name, value)
  sprintf("\\providecommand{\\%s}{}\\renewcommand{\\%s}{%s}",
          name, name, value)
mac <- c(
  nc("GoldSource", DATA_SOURCE),
  nc("GoldDays", nrow(panel)),
  nc("GoldGainBFiveHundred",
     sprintf("%.2f", 100 * (mse_of(gm$preds$`SSRF-JS`, gm$y) /
                              mse_of(gm$preds$RF, gm$y) - 1))),
  nc("GoldGainBTen",
     sprintf("%.2f", 100 * (mse_of(gm10$preds$`SSRF-JS`, gm10$y) /
                              mse_of(gm10$preds$RF, gm10$y) - 1))))
writeLines(mac, file.path(OUT_DIR, "real_macros.tex"))

cat("\n=================== EMPIRICAL ANALYSIS COMPLETE ===================\n")
cat("Gold data source:", DATA_SOURCE, "\n")
cat("Outputs written to:", normalizePath(OUT_DIR), "\n")
print(list.files(OUT_DIR))
