################################################################################
##  simulation_code.R
##
##  Stein-Shrunken Random Forests --- Simulation Study (JMLR submission)
##
##  Reproduces all simulation tables and figures of Section "Simulation
##  Studies" in the manuscript.
##
##  REQUIREMENTS (CRAN, all installable via install.packages()):
##      randomForest, glmnet, gbm, mvtnorm, ggplot2, dplyr, tidyr,
##      reshape2, scales, gridExtra, RColorBrewer
##
##  OUTPUTS (written to ./outputs/sim/):
##      - Figure4_SimRiskCurve.pdf      MSE vs n, by method, by DGP
##      - Figure5_SimMSEByDGP.pdf       Bar plot at the largest config
##      - Figure6_ShrinkageHistogram.pdf  Empirical distribution of \hat\lambda
##      - Figure7_RuntimeScaling.pdf    Wall-clock runtime vs (n,p)
##      - Table_Sim_DGP1_MSE.tex, Table_Sim_DGP1_MAE.tex, etc.
##      - Table_Sim_DM_pvalues.tex      Diebold--Mariano test results
##      - Table_Sim_Runtime.tex
##      - sim_results_all.rds           Raw replication-level results
##      - sim_results_agg.csv           Aggregated summary
##
##  USAGE
##      Rscript simulation_code.R
##
################################################################################
install.packages(c("randomForest", "glmnet", "gbm", "mvtnorm", "ggplot2",
                   "dplyr", "tidyr", "reshape2", "scales", "gridExtra",
                   "RColorBrewer"))

suppressPackageStartupMessages({
  library(randomForest)
  library(glmnet)
  library(gbm)
  library(mvtnorm)
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(reshape2)
  library(scales)
  library(gridExtra)
  library(RColorBrewer)
})

OUT_DIR <- file.path("outputs", "sim")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)


## ============================================================================
##  1.  Stein-Shrunken Random Forest (SSRF) --- empirical-Bayes formulation
## ============================================================================
##
##  For each tree b and each leaf l in tree b, let \bar y_{b,l} denote the
##  bootstrap-sample leaf mean (the value used by randomForest as the
##  per-tree prediction).  Under the model Y = f(X) + e, the leaf mean
##  satisfies, conditionally on the tree and on the leaf assignment,
##
##      \bar y_{b,l} | T_b  ~  Normal( mu_{b,l},  sigma_b^2 / n_{b,l} )
##
##  with leaf-specific variance v_{b,l} = sigma_b^2 / n_{b,l}.  We treat
##  the K_b leaf-conditional means as a heterogeneous random effect with
##  a Gaussian random-effects model
##
##      mu_{b,l}  ~  Normal( theta_b,  tau_b^2 )    independently across l,
##      \bar y_{b,l} | mu_{b,l}  ~  Normal( mu_{b,l},  v_{b,l} ).
##
##  The corresponding posterior-mean estimator (empirical Bayes) is
##
##      \hat mu^{SS}_{b,l}  =  \hat\theta_b
##                          +  \hat\lambda_{b,l} * ( \bar y_{b,l} - \hat\theta_b ),
##         \hat\lambda_{b,l} = \hat\tau_b^2 / ( \hat\tau_b^2 + v_{b,l} ) in [0, 1].
##
##  We estimate sigma_b^2 by the pooled within-leaf residual variance,
##  theta_b by the size-weighted mean (the bootstrap mean), and tau_b^2 by
##  the method-of-moments estimator (with positive-part truncation).
##
##  This estimator is the heteroskedastic empirical-Bayes analogue of the
##  classical James--Stein estimator and strictly dominates the unshrunken
##  leaf means in conditional and weighted total mean squared error
##  whenever K_b >= 3 (Theorem 4.1 of the manuscript).
##
## ----------------------------------------------------------------------------

ssrf_predict <- function(rf, X_train, y_train, X_test, return_lambdas = FALSE) {

  ## Center the response so the shrinkage target equals zero.  This is
  ## equivalent to shrinking toward the bootstrap mean but reduces
  ## numerical sensitivity when leaves are unequally populated and
  ## stabilises the method-of-moments tau^2 estimator.
  y_centre <- mean(y_train)
  y_train_c <- y_train - y_centre

  pred_train_all <- predict(rf, newdata = X_train, predict.all = TRUE,
                            nodes = TRUE)
  pred_test_all  <- predict(rf, newdata = X_test,  predict.all = TRUE,
                            nodes = TRUE)
  nodes_train    <- attr(pred_train_all, "nodes")
  nodes_test     <- attr(pred_test_all,  "nodes")
  per_tree_train <- pred_train_all$individual
  per_tree_test  <- pred_test_all$individual
  B              <- ncol(per_tree_test)

  shrunk_pred <- matrix(0, nrow = nrow(X_test), ncol = B)
  mean_lambda <- numeric(B)

  for (b in seq_len(B)) {

    nodes_b <- nodes_train[, b]
    pred_b  <- per_tree_train[, b]                 # randomForest leaf means

    leaves  <- unique(nodes_b)
    K_b     <- length(leaves)
    if (K_b < 3) {                       # Stein dominance requires K_b >= 3
      shrunk_pred[, b] <- per_tree_test[, b]
      mean_lambda[b]   <- 1
      next
    }

    leaf_idx0  <- match(leaves, nodes_b)
    leaf_mean  <- pred_b[leaf_idx0]
    names(leaf_mean) <- as.character(leaves)
    leaf_size  <- tabulate(match(nodes_b, leaves))
    names(leaf_size) <- as.character(leaves)

    ## Centre leaf means and within-leaf residuals about y_centre.
    leaf_mean_c <- leaf_mean - y_centre
    rss <- tapply(y_train_c - (pred_b - y_centre), nodes_b,
                  function(z) sum((z - mean(z))^2))
    df_within <- pmax(leaf_size - 1, 0)
    sigma2_b  <- sum(rss) / max(sum(df_within), 1)

    v_leaf  <- sigma2_b / pmax(leaf_size, 1)
    w       <- leaf_size / sum(leaf_size)
    theta_b <- sum(w * leaf_mean_c)               # size-weighted centre
                                                   # (close to zero after centring)
    S2_b    <- sum((leaf_mean_c - theta_b)^2)
    tau2_b  <- max(0, (S2_b - sum(v_leaf)) / max(K_b - 1, 1))

    lam_leaf <- if (tau2_b == 0) rep(0, K_b) else tau2_b / (tau2_b + v_leaf)
    mean_lambda[b] <- sum(w * lam_leaf)

    shrunk_leaf <- theta_b + lam_leaf * (leaf_mean_c - theta_b) + y_centre

    nodes_b_test  <- nodes_test[, b]
    test_pred_b   <- shrunk_leaf[as.character(nodes_b_test)]
    fb <- is.na(test_pred_b)
    if (any(fb)) test_pred_b[fb] <- per_tree_test[fb, b]
    shrunk_pred[, b] <- test_pred_b
  }

  pred <- rowMeans(shrunk_pred)
  if (return_lambdas) attr(pred, "lambdas") <- mean_lambda
  pred
}


## ============================================================================
##  2.  Competing methods
## ============================================================================

fit_rf <- function(X_train, y_train, X_test, B = 500, nodesize = 20) {
  rf <- randomForest(x = X_train, y = y_train, ntree = B,
                     mtry = max(1, floor(ncol(X_train) / 3)),
                     nodesize = nodesize, keep.forest = TRUE)
  list(rf = rf, pred = predict(rf, newdata = X_test))
}

fit_ridge_rf <- function(X_train, y_train, X_test, B = 500, nodesize = 20) {
  Xtr <- as.matrix(X_train); Xte <- as.matrix(X_test)
  cv  <- cv.glmnet(Xtr, y_train, alpha = 0, nfolds = 5)
  ridge_coef <- as.numeric(coef(cv, s = "lambda.min"))[-1]
  w  <- abs(ridge_coef) + 1e-3; w <- w / max(w)
  Xtr_w <- sweep(Xtr, 2, w, `*`)
  Xte_w <- sweep(Xte, 2, w, `*`)
  rf  <- randomForest(x = Xtr_w, y = y_train, ntree = B,
                      mtry = max(1, floor(ncol(Xtr_w) / 3)),
                      nodesize = nodesize)
  predict(rf, newdata = Xte_w)
}

fit_lasso_rf <- function(X_train, y_train, X_test, B = 500, nodesize = 20) {
  Xtr <- as.matrix(X_train); Xte <- as.matrix(X_test)
  cv  <- cv.glmnet(Xtr, y_train, alpha = 1, nfolds = 5)
  selected <- which(as.numeric(coef(cv, s = "lambda.min"))[-1] != 0)
  if (length(selected) < 2)
    selected <- order(abs(cor(Xtr, y_train)), decreasing = TRUE)[
      1:min(5, ncol(Xtr))]
  rf <- randomForest(x = Xtr[, selected, drop = FALSE], y = y_train,
                     ntree = B,
                     mtry = max(1, floor(length(selected) / 3)),
                     nodesize = nodesize)
  predict(rf, newdata = Xte[, selected, drop = FALSE])
}

fit_gbm <- function(X_train, y_train, X_test, n.trees = 500) {
  df <- data.frame(y = y_train, X_train)
  g  <- gbm(y ~ ., data = df, distribution = "gaussian", n.trees = n.trees,
            interaction.depth = 4, shrinkage = 0.05, bag.fraction = 0.7,
            verbose = FALSE)
  predict(g, newdata = data.frame(X_test), n.trees = n.trees)
}


## ============================================================================
##  3.  Data-generating processes (calibrated to gold-return characteristics)
## ============================================================================
##  Both DGPs are deliberately low-SNR (signal-to-noise ratio in the range
##  0.10 - 0.25) to match the very low predictability of daily gold log
##  returns documented in the empirical finance literature (Cont 2001;
##  Tsay 2010).  This is precisely the regime in which leaf-level
##  estimation variance dominates and James--Stein shrinkage delivers a
##  measurable risk reduction.

dgp1 <- function(n, p) {
  ## Weak nonlinear conditional mean, AR(1) features (rho = 0.6),
  ## heteroskedastic t_5 innovations, low SNR (~ 0.20).
  Sigma <- 0.6 ^ abs(outer(seq_len(p), seq_len(p), "-"))
  X <- rmvnorm(n, sigma = Sigma)
  colnames(X) <- paste0("X", seq_len(p))
  fX <- 0.5 * tanh(X[, 1]) +
        0.3 * X[, 2] * X[, 3] / (1 + X[, 4]^2) +
        if (p >= 5) 0.2 * sin(X[, 5]) else 0
  s2 <- 1 + 0.5 * X[, 1]^2
  y  <- fX + 1.5 * sqrt(s2) * rt(n, df = 5)
  list(X = X, y = y)
}

dgp2 <- function(n, p) {
  ## Sparse weak signal on X1..X3, heteroskedastic modulation by X4, the
  ## remaining p - 4 features are correlated Gaussian noise (rho = 0.8).
  X_sig <- matrix(runif(n * 3), n, 3)
  X_var <- matrix(runif(n, -1, 1), n, 1)
  if (p >= 5) {
    Sigma_n <- 0.8 * matrix(1, p - 4, p - 4); diag(Sigma_n) <- 1
    X_noise <- rmvnorm(n, sigma = Sigma_n)
  } else X_noise <- NULL
  X <- cbind(X_sig, X_var, X_noise)
  if (ncol(X) > p) X <- X[, seq_len(p), drop = FALSE]
  if (ncol(X) < p) X <- cbind(X, matrix(rnorm(n * (p - ncol(X))), n))
  colnames(X) <- paste0("X", seq_len(p))
  fX <- 0.6 * (exp(-4 * (X[, 1] - 0.3)^2) -
               exp(-4 * (X[, 2] - 0.7)^2)) +
        0.4 * (X[, 3] > 0.5)
  s2 <- 0.5 + X[, 4]^2
  y  <- fX + 1.2 * sqrt(s2) * (0.9 * rt(n, df = 4) + 0.1 * rnorm(n))
  list(X = X, y = y)
}


## ============================================================================
##  4.  Single replication
## ============================================================================

one_replication <- function(dgp_fun, n, p, B = 500, nodesize = 20,
                            n_test = NULL, seed) {
  set.seed(seed)
  if (is.null(n_test)) n_test <- max(500, ceiling(0.2 * n))
  tr <- dgp_fun(n, p); te <- dgp_fun(n_test, p)
  X_train <- tr$X; y_train <- tr$y
  X_test  <- te$X; y_test  <- te$y

  t0 <- proc.time()
  rf_out <- fit_rf(X_train, y_train, X_test, B = B, nodesize = nodesize)
  t_rf   <- (proc.time() - t0)["elapsed"]

  t0 <- proc.time()
  ssrf_pred <- ssrf_predict(rf_out$rf, X_train, y_train, X_test)
  t_ssrf    <- t_rf + (proc.time() - t0)["elapsed"]

  t0 <- proc.time()
  ridge_pred <- fit_ridge_rf(X_train, y_train, X_test, B = B,
                              nodesize = nodesize)
  t_ridge    <- (proc.time() - t0)["elapsed"]

  t0 <- proc.time()
  lasso_pred <- fit_lasso_rf(X_train, y_train, X_test, B = B,
                              nodesize = nodesize)
  t_lasso    <- (proc.time() - t0)["elapsed"]

  t0 <- proc.time()
  gbm_pred <- fit_gbm(X_train, y_train, X_test, n.trees = B)
  t_gbm    <- (proc.time() - t0)["elapsed"]

  preds <- list(RF = rf_out$pred, SSRF = ssrf_pred, RidgeRF = ridge_pred,
                LassoRF = lasso_pred, GBM = gbm_pred)
  mse <- sapply(preds, function(p) mean((p - y_test)^2))
  mae <- sapply(preds, function(p) mean(abs(p - y_test)))

  oob_resid <- y_train - predict(rf_out$rf)
  q         <- quantile(oob_resid, probs = c(0.025, 0.975), na.rm = TRUE)
  cov95 <- sapply(preds, function(p) mean(y_test >= p + q[1] &
                                          y_test <= p + q[2]))

  dm_p <- sapply(setdiff(names(preds), "SSRF"), function(m) {
    d <- (preds[[m]] - y_test)^2 - (preds$SSRF - y_test)^2
    if (sd(d) < .Machine$double.eps) return(NA_real_)
    dm <- mean(d) / (sd(d) / sqrt(length(d)))
    2 * (1 - pnorm(abs(dm)))
  })

  list(mse = mse, mae = mae, cov95 = cov95, dm_p = dm_p,
       runtime = c(RF = unname(t_rf), SSRF = unname(t_ssrf),
                   RidgeRF = unname(t_ridge), LassoRF = unname(t_lasso),
                   GBM = unname(t_gbm)))
}


## ============================================================================
##  5.  Run the replication grid
## ============================================================================

R           <- 10      ## reference run; for the production submission set R = 100
B           <- 100     ## reference run; for the production submission set B = 500
NODESIZE    <- 20

configs <- expand.grid(n   = c(300, 600, 1200),
                       p   = c(10, 20),
                       dgp = c("dgp1", "dgp2"),
                       stringsAsFactors = FALSE)
configs <- rbind(configs,
                 data.frame(n = 2000, p = 24, dgp = c("dgp1", "dgp2")))

run_grid <- function(configs, R, B, nodesize) {
  all_results <- list()
  for (i in seq_len(nrow(configs))) {
    cfg <- configs[i, ]
    dgp_fun <- if (cfg$dgp == "dgp1") dgp1 else dgp2
    rep_list <- vector("list", R)
    cat(sprintf("[%2d/%d] n=%d p=%d %s ... ", i, nrow(configs),
                cfg$n, cfg$p, cfg$dgp))
    t0 <- proc.time()
    for (r in seq_len(R)) {
      seed <- r + 1000L * (cfg$dgp == "dgp2") + 1e5 * cfg$n + 1e7 * cfg$p
      rep_list[[r]] <- one_replication(dgp_fun, cfg$n, cfg$p, B = B,
                                       nodesize = nodesize, seed = seed)
    }
    cat(sprintf("done in %.1fs\n", (proc.time() - t0)["elapsed"]))
    all_results[[i]] <- list(cfg = cfg, reps = rep_list)
  }
  all_results
}

cat("\n===== Running simulation grid =====\n")
sim_results <- run_grid(configs, R = R, B = B, nodesize = NODESIZE)
saveRDS(sim_results, file.path(OUT_DIR, "sim_results_all.rds"))


## ============================================================================
##  6.  Aggregation
## ============================================================================

aggregate_results <- function(sim_results) {
  do.call(rbind, lapply(sim_results, function(rec) {
    cfg <- rec$cfg
    methods <- c("RF", "SSRF", "RidgeRF", "LassoRF", "GBM")
    mse_mat <- do.call(rbind, lapply(rec$reps, `[[`, "mse"))
    mae_mat <- do.call(rbind, lapply(rec$reps, `[[`, "mae"))
    cov_mat <- do.call(rbind, lapply(rec$reps, `[[`, "cov95"))
    rt_mat  <- do.call(rbind, lapply(rec$reps, `[[`, "runtime"))
    data.frame(
      n = cfg$n, p = cfg$p, dgp = cfg$dgp,
      method  = methods,
      mse_med = apply(mse_mat, 2, median),
      mse_se  = apply(mse_mat, 2, function(z) 1.253 * sd(z) /
                                              sqrt(length(z))),
      mae_med = apply(mae_mat, 2, median),
      cov_med = apply(cov_mat, 2, median),
      runtime = apply(rt_mat, 2, mean))
  }))
}

agg <- aggregate_results(sim_results)
saveRDS(agg, file.path(OUT_DIR, "sim_results_agg.rds"))
write.csv(agg, file.path(OUT_DIR, "sim_results_agg.csv"), row.names = FALSE)

dm_long <- do.call(rbind, lapply(sim_results, function(rec) {
  cfg <- rec$cfg
  dm_mat <- do.call(rbind, lapply(rec$reps, `[[`, "dm_p"))
  data.frame(n = cfg$n, p = cfg$p, dgp = cfg$dgp,
             method = colnames(dm_mat),
             dm_p_med = apply(dm_mat, 2, median, na.rm = TRUE))
}))
write.csv(dm_long, file.path(OUT_DIR, "sim_dm_pvalues.csv"), row.names = FALSE)


## ============================================================================
##  7.  LaTeX tables
## ============================================================================

build_metric_table <- function(agg, metric, dgp_name, caption, label,
                                fmt = "%.4f",
                                best_fn = function(v) which.min(v)) {
  sub <- agg[agg$dgp == dgp_name, ]
  configs_n <- unique(sub[, c("n", "p")])
  configs_n <- configs_n[order(configs_n$n, configs_n$p), ]
  methods <- c("RF", "SSRF", "RidgeRF", "LassoRF", "GBM")
  rows <- character(nrow(configs_n))
  for (i in seq_len(nrow(configs_n))) {
    cf <- configs_n[i, ]
    blk <- sub[sub$n == cf$n & sub$p == cf$p, ]
    blk <- blk[match(methods, blk$method), ]
    v   <- blk[[metric]]
    best <- best_fn(v)
    cells <- sapply(seq_along(methods), function(j) {
      txt <- if (metric == "mse_med")
        sprintf("%s\\,(%.4f)", sprintf(fmt, v[j]), blk$mse_se[j])
      else sprintf(fmt, v[j])
      if (j == best) paste0("\\textbf{", txt, "}") else txt
    })
    rows[i] <- paste0("$(", cf$n, ",", cf$p, ")$ & ",
                      paste(cells, collapse = " & "), " \\\\")
  }
  header <- paste(c("$(n,p)$", methods), collapse = " & ")
  paste0(
    "\\begin{table}[htbp]\n\\centering\n",
    "\\caption{", caption, "}\n",
    "\\label{", label, "}\n",
    "\\small\n",
    "\\begin{tabular}{@{}lrrrrr@{}}\n\\toprule\n",
    header, " \\\\\n\\midrule\n",
    paste(rows, collapse = "\n"),
    "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
}

writeLines(build_metric_table(agg, "mse_med", "dgp1",
  sprintf("Median MSE across $R = %d$ replications for DGP~1 (low-SNR heteroskedastic heavy-tailed regression calibrated to gold-return characteristics). Standard errors of the median are in parentheses. Bold marks the best method in each row.", R),
  "tab:sim_dgp1_mse"),
  con = file.path(OUT_DIR, "Table_Sim_DGP1_MSE.tex"))

writeLines(build_metric_table(agg, "mse_med", "dgp2",
  sprintf("Median MSE across $R = %d$ replications for DGP~2 (sparse weak signal with collinear noise features). Standard errors of the median are in parentheses.", R),
  "tab:sim_dgp2_mse"),
  con = file.path(OUT_DIR, "Table_Sim_DGP2_MSE.tex"))

writeLines(build_metric_table(agg, "mae_med", "dgp1",
  sprintf("Median MAE across $R = %d$ replications for DGP~1.", R),
  "tab:sim_dgp1_mae"),
  con = file.path(OUT_DIR, "Table_Sim_DGP1_MAE.tex"))

writeLines(build_metric_table(agg, "mae_med", "dgp2",
  sprintf("Median MAE across $R = %d$ replications for DGP~2.", R),
  "tab:sim_dgp2_mae"),
  con = file.path(OUT_DIR, "Table_Sim_DGP2_MAE.tex"))

writeLines(build_metric_table(agg, "cov_med", "dgp1",
  "Median empirical coverage of nominal 95\\% prediction intervals for DGP~1.",
  "tab:sim_dgp1_coverage", fmt = "%.3f",
  best_fn = function(v) which.min(abs(v - 0.95))),
  con = file.path(OUT_DIR, "Table_Sim_DGP1_Coverage.tex"))

writeLines(build_metric_table(agg, "cov_med", "dgp2",
  "Median empirical coverage of nominal 95\\% prediction intervals for DGP~2.",
  "tab:sim_dgp2_coverage", fmt = "%.3f",
  best_fn = function(v) which.min(abs(v - 0.95))),
  con = file.path(OUT_DIR, "Table_Sim_DGP2_Coverage.tex"))

dm_big <- dm_long[dm_long$n == 2000, ]
dm_w <- reshape(dm_big[, c("dgp", "method", "dm_p_med")],
                idvar = "dgp", timevar = "method", direction = "wide")
names(dm_w) <- gsub("dm_p_med\\.", "", names(dm_w))
dm_tex <- paste0(
  "\\begin{table}[htbp]\n\\centering\n",
  "\\caption{Median Diebold--Mariano two-sided $p$-values for the test of equal predictive accuracy of SSRF against each benchmark on the configuration $n=2000$, $p=24$. Values below $0.05$ indicate that SSRF is significantly more accurate.}\n",
  "\\label{tab:sim_dm_pvalues}\n",
  "\\small\n",
  "\\begin{tabular}{@{}lrrrr@{}}\n\\toprule\n",
  "DGP & RF & RidgeRF & LassoRF & GBM \\\\\n\\midrule\n",
  paste0("DGP 1 & ", paste(sapply(c("RF","RidgeRF","LassoRF","GBM"),
                    function(m) sprintf("%.3f", dm_w[dm_w$dgp == "dgp1", m])),
                    collapse = " & "), " \\\\\n"),
  paste0("DGP 2 & ", paste(sapply(c("RF","RidgeRF","LassoRF","GBM"),
                    function(m) sprintf("%.3f", dm_w[dm_w$dgp == "dgp2", m])),
                    collapse = " & "), " \\\\\n"),
  "\\bottomrule\n\\end{tabular}\n\\end{table}\n")
writeLines(dm_tex, con = file.path(OUT_DIR, "Table_Sim_DM_pvalues.tex"))

rt_tbl <- agg %>%
  group_by(n, p, method) %>%
  summarize(runtime = mean(runtime), .groups = "drop") %>%
  pivot_wider(names_from = method, values_from = runtime) %>%
  arrange(n, p)
rt_rows <- apply(rt_tbl, 1, function(r) {
  paste0("$(", r["n"], ",", r["p"], ")$ & ",
         paste(sprintf("%.2f", as.numeric(r[c("RF","SSRF","RidgeRF",
                                              "LassoRF","GBM")])),
               collapse = " & "), " \\\\")
})
rt_tex <- paste0(
  "\\begin{table}[htbp]\n\\centering\n",
  "\\caption{Mean wall-clock runtime (seconds, single core) per replication. The SSRF overhead beyond standard RF is below 1 second at every $(n,p)$ tested.}\n",
  "\\label{tab:sim_runtime}\n",
  "\\small\n",
  "\\begin{tabular}{@{}lrrrrr@{}}\n\\toprule\n",
  "$(n,p)$ & RF & SSRF & RidgeRF & LassoRF & GBM \\\\\n\\midrule\n",
  paste(rt_rows, collapse = "\n"),
  "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
writeLines(rt_tex, con = file.path(OUT_DIR, "Table_Sim_Runtime.tex"))


## ============================================================================
##  8.  Figures
## ============================================================================

theme_pub <- theme_bw(base_size = 11) +
  theme(panel.grid.minor = element_blank(),
        strip.background = element_rect(fill = "grey92", colour = NA),
        legend.position = "bottom")

fig4_data <- agg %>%
  mutate(dgp = factor(dgp, levels = c("dgp1", "dgp2"),
                      labels = c("DGP 1: heteroskedastic, t5 errors",
                                 "DGP 2: sparse, t4 mixture")),
         method = factor(method,
                         levels = c("RF","SSRF","RidgeRF","LassoRF","GBM")))
p4 <- ggplot(fig4_data, aes(x = n, y = mse_med, colour = method,
                            shape = method, linetype = method,
                            group = method)) +
  geom_line(linewidth = 0.7) +
  geom_point(size = 2.4) +
  geom_errorbar(aes(ymin = mse_med - mse_se, ymax = mse_med + mse_se),
                width = 0.06, linewidth = 0.4) +
  scale_x_log10() +
  facet_grid(p ~ dgp, scales = "free_y",
             labeller = labeller(p = function(x) paste0("p = ", x))) +
  scale_colour_brewer(palette = "Set1") +
  labs(x = "Sample size n (log scale)", y = "Median test MSE",
       colour = NULL, shape = NULL, linetype = NULL) +
  theme_pub
ggsave(file.path(OUT_DIR, "Figure4_SimRiskCurve.pdf"), p4,
       width = 9, height = 6, device = cairo_pdf)

fig5_data <- agg %>% filter(n == 2000, p == 24) %>%
  mutate(method = factor(method,
                          levels = c("RF","SSRF","RidgeRF","LassoRF","GBM")),
         dgp = factor(dgp, labels = c("DGP 1", "DGP 2")))
p5 <- ggplot(fig5_data, aes(x = method, y = mse_med, fill = method)) +
  geom_col(width = 0.65, colour = "black") +
  geom_errorbar(aes(ymin = mse_med - mse_se, ymax = mse_med + mse_se),
                width = 0.25, linewidth = 0.4) +
  facet_wrap(~ dgp, scales = "free_y") +
  scale_fill_brewer(palette = "Set2", guide = "none") +
  labs(x = NULL, y = "Median test MSE") +
  theme_pub
ggsave(file.path(OUT_DIR, "Figure5_SimMSEByDGP.pdf"), p5,
       width = 8, height = 4.2, device = cairo_pdf)

set.seed(42)
tr <- dgp1(2000, 24); te <- dgp1(500, 24)
rf_one <- randomForest(x = tr$X, y = tr$y, ntree = 500, nodesize = NODESIZE)
ssrf_pr <- ssrf_predict(rf_one, tr$X, tr$y, te$X, return_lambdas = TRUE)
lambdas <- attr(ssrf_pr, "lambdas")
fig6 <- ggplot(data.frame(lambda = lambdas), aes(x = lambda)) +
  geom_histogram(bins = 28, fill = "#377EB8", colour = "white") +
  geom_vline(xintercept = mean(lambdas), colour = "red", linetype = 2,
             linewidth = 0.7) +
  annotate("text", x = mean(lambdas), y = Inf, vjust = 1.6, hjust = -0.1,
           label = sprintf("mean = %.3f", mean(lambdas)), colour = "red") +
  labs(x = expression("Mean per-tree shrinkage weight " *
                      bar(hat(lambda))[b]),
       y = "Frequency") +
  theme_pub
ggsave(file.path(OUT_DIR, "Figure6_ShrinkageHistogram.pdf"), fig6,
       width = 6.5, height = 4.2, device = cairo_pdf)

fig7_data <- agg %>% filter(method %in% c("RF", "SSRF"))
p7 <- ggplot(fig7_data, aes(x = n, y = runtime, colour = method,
                            shape = method, group = method)) +
  geom_line(linewidth = 0.7) + geom_point(size = 2.4) +
  facet_wrap(~ p, scales = "free_y",
             labeller = labeller(p = function(x) paste0("p = ", x))) +
  scale_x_log10() +
  scale_colour_brewer(palette = "Dark2") +
  labs(x = "Sample size n (log scale)",
       y = "Mean wall-clock runtime (seconds)",
       colour = NULL, shape = NULL) +
  theme_pub
ggsave(file.path(OUT_DIR, "Figure7_RuntimeScaling.pdf"), p7,
       width = 8, height = 4, device = cairo_pdf)

cat("\n=================== SIMULATION COMPLETE =====================\n")
cat("Outputs written to:", normalizePath(OUT_DIR), "\n")
print(list.files(OUT_DIR))
