################################################################################
##  simulation_code.R
##
##  Simulation study for Stein-shrunken leaf estimation in random forests.
##
##  The study is organised around one question.  Ensemble averaging and
##  leaf-level shrinkage both act on the same quantity, the sampling variance
##  of a leaf mean, so how much does the second still buy once the first has
##  been applied?
##
##  Experiment A (substitution)  the shrinkage gain as a function of the
##      ensemble size B and the minimum leaf size, with the process and the
##      sample size held fixed.  Because a shrunken forest of B trees is
##      obtained by truncating one fitted forest, the comparison across B is
##      paired.
##  Experiment B (grid)          accuracy across sample sizes and dimensions at
##      a small and a large ensemble size, against the competing predictors.
##  Experiment C (equivalence)   the number of unshrunken trees whose risk
##      matches that of a given number of shrunken ones.
##
##  Risk is reported both against the observed responses and against the true
##  conditional mean.  The theory speaks about the second, and the first is
##  what a practitioner measures; reporting only the first hides differences
##  under the irreducible noise.
##
##  REQUIREMENTS (all available from CRAN)
##      randomForest, glmnet, gbm, matrixStats, ggplot2, dplyr, tidyr,
##      RColorBrewer, scales
##
##  USAGE
##      Rscript simulation_code.R              # reference run
##      SSRF_REPS=100 Rscript simulation_code.R   # production run
##
##  OUTPUTS (./outputs/sim/)
##      sim_substitution.csv, sim_grid.csv, sim_equivalence.csv
##      Table_Sim_*.tex, Figure_Sim_*.pdf, sim_macros.tex
################################################################################

## Locate ssrf_core.R next to this script, whether the script is run with
## Rscript, sourced, or executed from the project root.
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
})

OUT_DIR <- file.path("outputs", "sim_R")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

REPS  <- as.integer(Sys.getenv("SSRF_REPS", "30"))
set.seed(as.integer(Sys.getenv("SSRF_SEED", "20260910")))


## ============================================================================
##  1.  Data-generating processes
## ============================================================================
##  All three are deliberately low signal-to-noise, in the range that daily
##  commodity and currency returns occupy, because that is where leaf-level
##  estimation variance dominates.  The realised ratio is recorded for each
##  replication rather than asserted.
##
##  DGP 1  smooth weak mean, autoregressive Gaussian design, variance rising
##         in x1, Student t_5 innovations.
##  DGP 2  sparse weak mean on three uniform coordinates, variance modulated
##         by a fourth, remaining coordinates collinear Gaussian noise.
##  DGP 3  the heavy-tail stress case.  Student t_2.5 innovations, so the
##         fourth moment does not exist and the within-leaf central limit
##         approximation is at its least accurate.  It is included to test the
##         robustness claims, not to flatter the method.
## ----------------------------------------------------------------------------

unit_t <- function(n, df) rt(n, df) / sqrt(df / (df - 2))

ar1_design <- function(n, p, rho = 0.6) {
  X <- matrix(0, n, p)
  X[, 1] <- rnorm(n)
  s <- sqrt(1 - rho^2)
  for (j in seq_len(p)[-1]) X[, j] <- rho * X[, j - 1] + s * rnorm(n)
  X
}

dgp1 <- function(n, p, noise = 0.9) {
  X <- ar1_design(n, p, 0.6)
  f <- 0.5 * tanh(X[, 1]) + 0.3 * X[, 2] * X[, 3] / (1 + X[, 4]^2)
  if (p >= 5) f <- f + 0.2 * sin(X[, 5])
  y <- f + noise * sqrt(1 + 0.5 * X[, 1]^2) * unit_t(n, 5)
  list(X = X, y = y, f = f)
}

dgp2 <- function(n, p, noise = 1.0) {
  X <- matrix(0, n, p)
  X[, 1:3] <- runif(n * 3)
  X[, 4] <- runif(n, -1, 1)
  if (p > 4) {
    common <- rnorm(n); rho <- 0.8
    X[, 5:p] <- sqrt(rho) * common + sqrt(1 - rho) * matrix(rnorm(n * (p - 4)), n)
  }
  f <- 0.6 * (exp(-4 * (X[, 1] - 0.3)^2) - exp(-4 * (X[, 2] - 0.7)^2)) +
       0.4 * (X[, 3] > 0.5)
  y <- f + noise * sqrt(0.5 + X[, 4]^2) *
       (0.9 * unit_t(n, 4) + 0.1 * rnorm(n))
  list(X = X, y = y, f = f)
}

dgp3 <- function(n, p, noise = 1.2) {
  X <- ar1_design(n, p, 0.3)
  f <- 0.4 * sign(X[, 1]) * sqrt(abs(X[, 1]))
  if (p >= 3) f <- f + 0.25 * X[, 2] * (X[, 3] > 0)
  y <- f + noise * sqrt(0.5 + 0.5 * abs(X[, 1])) * unit_t(n, 2.5)
  list(X = X, y = y, f = f)
}

DGPS <- list(dgp1 = dgp1, dgp2 = dgp2, dgp3 = dgp3)
snr_of <- function(y, f) var(f) / var(y - f)


## ============================================================================
##  2.  Experiment A: the substitution surface
## ============================================================================
##  One forest of the largest size is grown per replication and truncated to
##  each smaller B.  Per-tree shrinkage does not depend on how many other trees
##  the forest contains, so truncation gives exactly the forest that would have
##  been grown with that many trees, and the comparison across B is paired.
## ----------------------------------------------------------------------------

experiment_substitution <- function(reps = REPS, n = 600, p = 10,
                                    n_test = 1500,
                                    B_list = c(1, 3, 5, 10, 25, 50, 100, 250, 500),
                                    leaf_list = c(2, 5, 20)) {
  ## nodesize = 1 is excluded deliberately.  randomForest splits a node while
  ## it holds more than nodesize observations, so nodesize = 1 grows every
  ## tree to purity, every leaf holds one observation, and the pooled
  ## within-leaf variance has no degrees of freedom left to estimate.  The
  ## shrinkage weight is then not identified from the data rather than merely
  ## small.  Note also that this parameter does not mean what
  ## min_samples_leaf means in scikit-learn: nodesize = 5 permits leaves of
  ## size 1 to 5, whereas min_samples_leaf = 5 guarantees at least five, so
  ## the two implementations grow different partitions at the same nominal
  ## setting and their tables are comparable in pattern rather than entry by
  ## entry.
  rows <- list(); k <- 0
  for (dg in names(DGPS)) {
    for (n0 in leaf_list) {
      for (r in seq_len(reps)) {
        tr <- DGPS[[dg]](n, p); te <- DGPS[[dg]](n_test, p)
        M <- fit_ssrf(tr$X, tr$y, B = max(B_list), nodesize = n0, rule = "js")
        ME <- ssrf_leaf_values(M$rf, tr$X, tr$y, rule = "eb")
        nodes <- attr(predict(M$rf, newdata = te$X, nodes = TRUE), "nodes")
        acc <- list(RF = 0, `SSRF-JS` = 0, SSRF = 0)
        prev <- 0
        for (B in B_list) {
          idx <- (prev + 1):B
          for (b in idx) {
            j  <- match(nodes[, b], M$leafvals[[b]]$leaf_ids)
            rw <- M$leafvals[[b]]$raw[j];    rw[is.na(j)] <- M$leafvals[[b]]$theta
            sh <- M$leafvals[[b]]$shrunk[j]; sh[is.na(j)] <- M$leafvals[[b]]$theta
            eb <- ME[[b]]$shrunk[j];         eb[is.na(j)] <- ME[[b]]$theta
            acc$RF <- acc$RF + rw
            acc$`SSRF-JS` <- acc$`SSRF-JS` + sh
            acc$SSRF <- acc$SSRF + eb
          }
          prev <- B
          for (mth in names(acc)) {
            pr <- acc[[mth]] / B
            k <- k + 1
            rows[[k]] <- data.frame(
              experiment = "substitution", dgp = dg, n = n, p = p, B = B,
              min_leaf = n0, rep = r, method = mth,
              mse = mse_of(pr, te$y), mae = mae_of(pr, te$y),
              mse_true = mse_of(pr, te$f),
              lambda = M$mean_lambda, leaves = M$leaves, n_eff = M$n_eff,
              snr = snr_of(tr$y, tr$f))
          }
        }
      }
      cat(sprintf("  substitution %s nodesize=%d done\n", dg, n0))
    }
  }
  do.call(rbind, rows)
}


## ============================================================================
##  3.  Experiment B: accuracy grid with competing predictors
## ============================================================================

GRID_NP <- list(c(300, 10), c(600, 20), c(1200, 20), c(2000, 24))

fit_all <- function(tr, te, B, n0) {
  preds <- list(); times <- list(); diag <- list()

  t0 <- proc.time()
  M <- fit_ssrf(tr$X, tr$y, B = B, nodesize = n0, rule = "js")
  tf <- (proc.time() - t0)["elapsed"]
  ME <- ssrf_leaf_values(M$rf, tr$X, tr$y, rule = "eb")
  preds$RF        <- ssrf_predict(M$rf, M$leafvals, te$X, "raw")
  preds$`SSRF-JS` <- ssrf_predict(M$rf, M$leafvals, te$X, "shrunk")
  preds$SSRF      <- ssrf_predict(M$rf, ME,          te$X, "shrunk")
  times$RF <- times$`SSRF-JS` <- times$SSRF <- tf
  diag$lambda <- M$mean_lambda; diag$leaves <- M$leaves; diag$n_eff <- M$n_eff

  ## the estimator exactly as it was specified before the leaf-size and
  ## degrees-of-freedom corrections, retained so that the size of that error
  ## can be reported rather than asserted
  ML <- ssrf_leaf_values(M$rf, tr$X, tr$y, rule = "eb", tau_method = "mom",
                         effective_size = FALSE)
  preds$`SSRF-naive` <- ssrf_predict(M$rf, ML, te$X, "shrunk")
  times$`SSRF-naive` <- tf
  diag$lambda_naive <- mean(ssrf_mean_lambda(ML))

  t0 <- proc.time()
  MR <- ssrf_leaf_values(M$rf, tr$X, tr$y, rule = "js", sigma_method = "huber")
  preds$`SSRF-rob` <- ssrf_predict(M$rf, MR, te$X, "shrunk")
  times$`SSRF-rob` <- tf + (proc.time() - t0)["elapsed"]

  t0 <- proc.time()
  H <- fit_ssrf_honest(tr$X, tr$y, B = B, nodesize = max(1, floor(n0 / 2)))
  th <- (proc.time() - t0)["elapsed"]
  preds$`RF-H`      <- honest_predict(H, te$X, "raw")
  preds$`SSRF-JS-H` <- honest_predict(H, te$X, "shrunk")
  times$`RF-H` <- times$`SSRF-JS-H` <- th

  t0 <- proc.time()
  preds$RidgeRF <- fit_ridge_rf(tr$X, tr$y, te$X, B, n0)
  times$RidgeRF <- (proc.time() - t0)["elapsed"]
  t0 <- proc.time()
  preds$LassoRF <- fit_lasso_rf(tr$X, tr$y, te$X, B, n0)
  times$LassoRF <- (proc.time() - t0)["elapsed"]
  t0 <- proc.time()
  preds$GBM <- fit_gbm(tr$X, tr$y, te$X, n_trees = max(B, 300))
  times$GBM <- (proc.time() - t0)["elapsed"]
  t0 <- proc.time()
  tu <- fit_rf_tuned(tr$X, tr$y, te$X, B)
  preds$`RF-tuned` <- tu$pred
  times$`RF-tuned` <- (proc.time() - t0)["elapsed"]
  diag$tuned_nodesize <- tu$nodesize
  t0 <- proc.time()
  MT <- fit_ssrf(tr$X, tr$y, B = B, nodesize = tu$nodesize, mtry = tu$mtry,
                 rule = "js")
  preds$`SSRF-tuned` <- ssrf_predict(MT$rf, MT$leafvals, te$X, "shrunk")
  times$`SSRF-tuned` <- times$`RF-tuned` + (proc.time() - t0)["elapsed"]

  list(preds = preds, times = times, diag = diag)
}

score_all <- function(res, te, meta) {
  m <- length(te$y)
  cal <- sample.int(m, floor(m / 2)); ev <- setdiff(seq_len(m), cal)
  ref <- (res$preds$`SSRF-JS` - te$y)^2
  do.call(rbind, lapply(names(res$preds), function(nm) {
    pr <- res$preds[[nm]]
    ci <- split_conformal(pr[cal], te$y[cal], pr[ev])
    dm <- dm_test((pr - te$y)^2, ref)
    cbind(meta, data.frame(
      method = nm, mse = mse_of(pr, te$y), mae = mae_of(pr, te$y),
      mse_true = mse_of(pr, te$f),
      coverage = mean(te$y[ev] >= ci$lower & te$y[ev] <= ci$upper),
      interval_score = interval_score(ci$lower, ci$upper, te$y[ev]),
      interval_width = mean(ci$upper - ci$lower),
      dm_stat = dm["stat"], dm_p = dm["p"],
      runtime = as.numeric(res$times[[nm]]),
      lambda = res$diag$lambda, lambda_naive = res$diag$lambda_naive,
      leaves = res$diag$leaves, n_eff = res$diag$n_eff,
      tuned_nodesize = res$diag$tuned_nodesize, row.names = NULL))
  }))
}

experiment_grid <- function(reps = REPS, B_small = 10, B_large = 500, n0 = 5) {
  rows <- list(); k <- 0
  for (dg in names(DGPS)) for (np in GRID_NP) {
    n <- np[1]; p <- np[2]; t0 <- proc.time()
    for (r in seq_len(reps)) {
      tr <- DGPS[[dg]](n, p); te <- DGPS[[dg]](max(1000, n), p)
      for (B in c(B_small, B_large)) {
        res <- fit_all(tr, te, B, n0)
        meta <- data.frame(experiment = "grid", dgp = dg, n = n, p = p, B = B,
                           min_leaf = n0, rep = r, snr = snr_of(tr$y, tr$f))
        k <- k + 1; rows[[k]] <- score_all(res, te, meta)
      }
    }
    cat(sprintf("  grid %s n=%d p=%d in %.0fs\n", dg, n, p,
                (proc.time() - t0)["elapsed"]))
  }
  do.call(rbind, rows)
}


## ============================================================================
##  4.  Experiment C: how many extra trees does shrinkage buy?
## ============================================================================

experiment_equivalence <- function(reps = REPS, n = 600, p = 10, n0 = 5,
                                   B_list = c(5, 10, 15, 25, 40, 60, 100, 150,
                                              250, 400, 600)) {
  rows <- list(); k <- 0
  for (dg in names(DGPS)) {
    for (r in seq_len(reps)) {
      tr <- DGPS[[dg]](n, p); te <- DGPS[[dg]](1500, p)
      M <- fit_ssrf(tr$X, tr$y, B = max(B_list), nodesize = n0, rule = "js")
      nodes <- attr(predict(M$rf, newdata = te$X, nodes = TRUE), "nodes")
      accR <- 0; accS <- 0; prev <- 0
      for (B in B_list) {
        for (b in (prev + 1):B) {
          j <- match(nodes[, b], M$leafvals[[b]]$leaf_ids)
          rw <- M$leafvals[[b]]$raw[j];    rw[is.na(j)] <- M$leafvals[[b]]$theta
          sh <- M$leafvals[[b]]$shrunk[j]; sh[is.na(j)] <- M$leafvals[[b]]$theta
          accR <- accR + rw; accS <- accS + sh
        }
        prev <- B
        for (nm in c("RF", "SSRF-JS")) {
          pr <- (if (nm == "RF") accR else accS) / B
          k <- k + 1
          rows[[k]] <- data.frame(experiment = "equivalence", dgp = dg, n = n,
                                  p = p, B = B, min_leaf = n0, rep = r,
                                  method = nm, mse = mse_of(pr, te$y),
                                  mse_true = mse_of(pr, te$f))
        }
      }
    }
    cat(sprintf("  equivalence %s done\n", dg))
  }
  do.call(rbind, rows)
}


## ============================================================================
##  5.  Run, aggregate, and write tables and figures
## ============================================================================

cat("===== Experiment A: substitution surface =====\n")
sub <- experiment_substitution()
write.csv(sub, file.path(OUT_DIR, "sim_substitution.csv"), row.names = FALSE)

cat("===== Experiment B: accuracy grid =====\n")
grd <- experiment_grid()
write.csv(grd, file.path(OUT_DIR, "sim_grid.csv"), row.names = FALSE)

cat("===== Experiment C: tree equivalence =====\n")
eqv <- experiment_equivalence()
write.csv(eqv, file.path(OUT_DIR, "sim_equivalence.csv"), row.names = FALSE)

## ---------------------------------------------------------------------------
##  Aggregation.  Medians are reported with a nonparametric bootstrap standard
##  error; the asymptotic 1.253 s / sqrt(R) formula assumes a normal sampling
##  distribution and misbehaves under the heavy-tailed losses these processes
##  produce.
## ---------------------------------------------------------------------------

agg_substitution <- sub %>%
  group_by(dgp, min_leaf, B, method) %>%
  summarise(mse_true_med = median(mse_true),
            mse_true_se = boot_median_se(mse_true),
            mse_med = median(mse), lambda = mean(lambda),
            n_eff = mean(n_eff), leaves = mean(leaves), .groups = "drop")

gain <- sub %>%
  select(dgp, min_leaf, B, rep, method, mse_true) %>%
  pivot_wider(names_from = method, values_from = mse_true) %>%
  mutate(gain_js = 100 * (`SSRF-JS` / RF - 1),
         gain_eb = 100 * (SSRF / RF - 1),
         win_js = `SSRF-JS` < RF) %>%
  group_by(dgp, min_leaf, B) %>%
  summarise(gain_js = median(gain_js), gain_eb = median(gain_eb),
            win_js = 100 * mean(win_js),
            wilcox_p = tryCatch(
              wilcox.test(RF, `SSRF-JS`, paired = TRUE,
                          alternative = "greater")$p.value,
              error = function(e) NA_real_), .groups = "drop")
write.csv(gain, file.path(OUT_DIR, "sim_substitution_gain.csv"),
          row.names = FALSE)

fmt <- function(x, d = 2) ifelse(is.na(x), "---", sprintf(paste0("%.", d, "f"), x))

##' Assemble one LaTeX table.  Tables with many columns are wrapped in a box
##' scaled to the text width, because at this page size a ten-column table
##' otherwise runs into the margin.
latex_table <- function(body, header, caption, label, align, note = NULL,
                        wide_at = 8) {
  ncol <- sum(strsplit(align, "")[[1]] %in% c("l", "r", "c"))
  inner <- paste0("\\begin{tabular}{", align, "}\n\\toprule\n", header,
                  " \\\\\n\\midrule\n", paste(body, collapse = "\n"),
                  "\n\\bottomrule\n\\end{tabular}")
  if (ncol > wide_at)
    inner <- paste0("\\resizebox{\\textwidth}{!}{%\n", inner, "}")
  paste0("\\begin{table}[htbp]\n\\centering\n\\caption{", caption, "}\n",
         "\\label{", label, "}\n\\small\n\\setlength{\\tabcolsep}{4pt}\n",
         inner, "\n",
         if (is.null(note)) "" else paste0("\\\\[2pt]\n\\parbox{\\textwidth}{\\footnotesize ", note, "}\n"),
         "\\end{table}\n")
}

## Table: the substitution surface
Bs <- sort(unique(gain$B))
body <- unlist(lapply(sort(unique(gain$dgp)), function(dg)
  lapply(sort(unique(gain$min_leaf)), function(nl) {
    g <- gain[gain$dgp == dg & gain$min_leaf == nl, ]
    g <- g[match(Bs, g$B), ]
    paste0(toupper(sub("dgp", "DGP~", dg)), " & ", nl, " & ",
           paste(fmt(g$gain_js, 1), collapse = " & "), " \\\\")
  })))
writeLines(latex_table(
  body,
  paste0("Process & $n_0$ & ", paste(sprintf("$B{=}%d$", Bs), collapse = " & ")),
  paste0("Median percentage change in test mean squared error, measured ",
         "against the true conditional mean, when the leaf means of a fitted ",
         "forest are replaced by their James--Stein shrunken values. ",
         "Negative entries favour shrinkage. Sample size $n=600$, $p=10$, ",
         REPS, " replications. Each row uses one fitted forest truncated to ",
         "each ensemble size, so the comparison across $B$ is paired."),
  "tab:substitution",
  paste0("ll", paste(rep("r", length(Bs)), collapse = ""))),
  file.path(OUT_DIR, "Table_Sim_Substitution.tex"))

## Table: accuracy grid
grid_tab <- grd %>%
  group_by(dgp, n, p, B, method) %>%
  summarise(mse_true = median(mse_true), se = boot_median_se(mse_true),
            cov = median(coverage), isc = median(interval_score),
            rt = mean(runtime), .groups = "drop")
write.csv(grid_tab, file.path(OUT_DIR, "sim_grid_agg.csv"), row.names = FALSE)

methods_show <- c("RF", "SSRF-JS", "SSRF", "SSRF-naive", "RF-tuned",
                  "SSRF-tuned", "RidgeRF", "LassoRF", "GBM")
for (Bv in sort(unique(grid_tab$B))) {
  blk <- grid_tab[grid_tab$B == Bv, ]
  body <- unlist(lapply(sort(unique(blk$dgp)), function(dg)
    lapply(seq_along(GRID_NP), function(i) {
      n <- GRID_NP[[i]][1]; p <- GRID_NP[[i]][2]
      s <- blk[blk$dgp == dg & blk$n == n & blk$p == p, ]
      v <- s$mse_true[match(methods_show, s$method)]
      cells <- fmt(v, 4)
      if (any(is.finite(v))) cells[which.min(v)] <-
        paste0("\\textbf{", cells[which.min(v)], "}")
      paste0(toupper(sub("dgp", "DGP~", dg)), " & $(", n, ",", p, ")$ & ",
             paste(cells, collapse = " & "), " \\\\")
    })))
  writeLines(latex_table(
    body,
    paste0("Process & $(n,p)$ & ",
           paste(gsub("-", "--", methods_show), collapse = " & ")),
    sprintf(paste0("Median test mean squared error against the true ",
                   "conditional mean, ensembles of $B=%d$ trees, %d ",
                   "replications. Bold marks the best method in each row."),
            Bv, REPS),
    sprintf("tab:grid_B%d", Bv),
    paste0("ll", paste(rep("r", length(methods_show)), collapse = ""))),
    file.path(OUT_DIR, sprintf("Table_Sim_Grid_B%d.tex", Bv)))
}

## Tree-equivalence factor: how many unshrunken trees match B shrunken ones
eq <- eqv %>% group_by(dgp, B, method) %>%
  summarise(mse_true = mean(mse_true), .groups = "drop") %>%
  pivot_wider(names_from = method, values_from = mse_true)
equiv_factor <- function(d) {
  f <- approxfun(log(d$B), d$RF, rule = 2)
  vapply(seq_len(nrow(d)), function(i) {
    target <- d$`SSRF-JS`[i]
    lo <- log(min(d$B)); hi <- log(max(d$B) * 20)
    if (f(hi) > target) return(NA_real_)
    r <- tryCatch(uniroot(function(z) f(z) - target, c(lo, hi))$root,
                  error = function(e) NA_real_)
    exp(r) / d$B[i]
  }, numeric(1))
}
eq <- eq %>% group_by(dgp) %>% mutate(factor = equiv_factor(pick(everything()))) %>%
  ungroup()
write.csv(eq, file.path(OUT_DIR, "sim_equivalence_agg.csv"), row.names = FALSE)

## Tree-equivalence table, for parity with the Python generator
eq_body <- vapply(sort(unique(eq$dgp)), function(dg) {
  s <- eq[eq$dgp == dg, ]
  s <- s[match(sort(unique(eq$B)), s$B), ]
  cells <- ifelse(is.na(s$factor), "$\\infty$", fmt(s$factor, 1))
  paste0(toupper(sub("dgp", "DGP~", dg)), " & ",
         paste(cells, collapse = " & "), " \\\\")
}, character(1))
writeLines(latex_table(
  eq_body,
  paste0("Process & ",
         paste(sprintf("$B{=}%d$", sort(unique(eq$B))), collapse = " & ")),
  paste0("Tree-equivalence factor: the number of unshrunken trees whose risk ",
         "matches that of $B$ shrunken ones, divided by $B$. An entry below ",
         "$1$ means the shrunken forest is matched by fewer unshrunken trees, ",
         "so the shrinkage has done harm; an entry of $\\infty$ means no ",
         "ensemble size in the range examined reaches the shrunken risk."),
  "tab:equivalence",
  paste0("l", paste(rep("r", length(unique(eq$B))), collapse = ""))),
  file.path(OUT_DIR, "Table_Sim_Equivalence.tex"))

## Macros, so that the numbers quoted in the text cannot drift from the tables
nc <- function(name, value)
  sprintf("\\providecommand{\\%s}{}\\renewcommand{\\%s}{%s}",
          name, name, value)
pick <- function(nl, B, d = NULL) {
  s <- gain[gain$min_leaf == nl & gain$B == B, ]
  if (!is.null(d)) s <- s[s$dgp == d, ]
  median(s$gain_js)
}
mac <- c(
  nc("SimReps", REPS),
  nc("GainSmallLeafBOne", sprintf("%.1f", -pick(5, 1))),
  nc("GainSmallLeafBTen", sprintf("%.1f", -pick(5, 10))),
  nc("GainSmallLeafBFiveHundred", sprintf("%.1f", -pick(5, 500))),
  nc("GainLargeLeafBOne", sprintf("%.1f", -pick(20, 1))),
  nc("GainLargeLeafBTen", sprintf("%.1f", -pick(20, 10))),
  nc("GainLargeLeafBFiveHundred", sprintf("%.1f", pick(20, 500))),
  nc("HeavyTailPenalty", sprintf("%.1f", pick(20, 500, "dgp3"))),
  nc("MedianEquivFactor", sprintf("%.1f", median(eq$factor[eq$B <= 25],
                                                 na.rm = TRUE))),
  nc("MeanLambdaSmallLeaf",
     sprintf("%.3f", mean(sub$lambda[sub$min_leaf == 5]))),
  nc("MeanLambdaLargeLeaf",
     sprintf("%.3f", mean(sub$lambda[sub$min_leaf == 20]))),
  nc("MedianSNR", sprintf("%.3f", median(sub$snr))))
writeLines(mac, file.path(OUT_DIR, "sim_macros.tex"))

## ---------------------------------------------------------------------------
##  Figures
## ---------------------------------------------------------------------------

theme_pub <- theme_bw(base_size = 11) +
  theme(panel.grid.minor = element_blank(),
        strip.background = element_rect(fill = "grey92", colour = NA),
        legend.position = "bottom")

p1 <- ggplot(gain, aes(x = B, y = gain_js, colour = factor(min_leaf),
                       shape = factor(min_leaf))) +
  geom_hline(yintercept = 0, colour = "grey40", linewidth = 0.4) +
  geom_line(linewidth = 0.7) + geom_point(size = 2.2) +
  scale_x_log10(breaks = Bs) +
  facet_wrap(~ dgp, labeller = labeller(dgp = function(z)
    toupper(sub("dgp", "DGP ", z)))) +
  scale_colour_brewer(palette = "Set1") +
  labs(x = "Number of trees B (log scale)",
       y = "Change in test MSE from shrinkage (%)",
       colour = "minimum leaf size", shape = "minimum leaf size") +
  theme_pub
ggsave(file.path(OUT_DIR, "Figure_Sim_Substitution.pdf"), p1,
       width = 9, height = 3.6)

p2 <- ggplot(eqv %>% group_by(dgp, B, method) %>%
               summarise(mse = mean(mse_true), .groups = "drop"),
             aes(x = B, y = mse, colour = method, shape = method)) +
  geom_line(linewidth = 0.7) + geom_point(size = 2.2) +
  scale_x_log10() + facet_wrap(~ dgp, scales = "free_y",
    labeller = labeller(dgp = function(z) toupper(sub("dgp", "DGP ", z)))) +
  scale_colour_brewer(palette = "Dark2") +
  labs(x = "Number of trees B (log scale)",
       y = "Test MSE against the true conditional mean",
       colour = NULL, shape = NULL) + theme_pub
ggsave(file.path(OUT_DIR, "Figure_Sim_Equivalence.pdf"), p2,
       width = 9, height = 3.6)

lam <- sub %>% distinct(dgp, min_leaf, rep, lambda, n_eff)
p3 <- ggplot(lam, aes(x = lambda, fill = factor(min_leaf))) +
  geom_histogram(bins = 30, colour = "white", position = "identity",
                 alpha = 0.75) +
  facet_wrap(~ dgp, labeller = labeller(dgp = function(z)
    toupper(sub("dgp", "DGP ", z)))) +
  scale_fill_brewer(palette = "Set2") +
  labs(x = expression("Leaf-size-weighted mean shrinkage weight " *
                        bar(hat(lambda))),
       y = "Frequency", fill = "minimum leaf size") + theme_pub
ggsave(file.path(OUT_DIR, "Figure_Sim_Lambda.pdf"), p3, width = 9, height = 3.4)

cat("\n=================== SIMULATION COMPLETE =====================\n")
cat("Outputs written to:", normalizePath(OUT_DIR), "\n")
print(list.files(OUT_DIR))
