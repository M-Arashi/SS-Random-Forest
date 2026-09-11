################################################################################
##  ssrf_core.R
##
##  Stein-shrunken leaf estimation for random forests: core routines.
##
##  Sourced by simulation_code.R and by realdata_code.R.  Nothing in this file
##  runs an analysis; it defines the estimator, the competing predictors and
##  the evaluation measures used by both.
##
##  ---------------------------------------------------------------------------
##  The estimator
##  ---------------------------------------------------------------------------
##  A fitted Breiman forest is left exactly as it is, and only the value stored
##  at each leaf is changed.  Within tree b, write
##
##      K_b        number of leaves,
##      c_{i}      number of times training observation i enters leaf l of the
##                 tree's bootstrap sample,
##      C_l        = sum_i c_i, the raw leaf count,
##      ybar_{b,l} = sum_i c_i Y_i / C_l, the leaf mean,
##      v_{b,l}    sampling variance of that mean,
##      theta_b    precision-weighted anchor,
##      Q_b        = sum_l (ybar_{b,l} - theta_b)^2 / v_{b,l}.
##
##  Four points of accounting matter and each of them changes the fitted
##  shrinkage weight materially.
##
##  1.  A bootstrap leaf mean is a weighted average of distinct observations,
##      so its variance is sigma^2 sum_i c_i^2 / C_l^2, not sigma^2 / C_l.
##      In-bag multiplicities follow a Poisson(1) law conditioned to be
##      positive, for which sum c_i^2 / C is close to two, so the raw count
##      overstates the information in a leaf by roughly a factor of two.  The
##      effective size n_eff = C^2 / sum_i c_i^2 is used throughout.
##
##  2.  With multiplicities, E[sum_i c_i (Y_i - ybar)^2] = sigma^2 (C - sum c_i^2 / C),
##      so the pooled within-leaf variance divides by sum_l (C_l - sum_i c_i^2 / C_l).
##      Dividing by sum_l (C_l - 1) understates sigma^2 by about the same
##      factor, and the two errors compound.
##
##  3.  Leaf-mean variances inside one tree span orders of magnitude, and the
##      unweighted moment estimator of the between-leaf variance, although
##      unbiased, is very noisy there.  The default is the DerSimonian-Laird
##      estimator, which is the inverse-variance-weighted moment estimator for
##      this random-effects model.
##
##  4.  Under honest growing the leaf values and both variance components are
##      computed on observations that took no part in choosing the splits.
##
##  Two shrinkage rules are provided.  The empirical-Bayes rule uses a
##  per-leaf weight tau^2 / (tau^2 + v_l); the James-Stein rule uses the single
##  positive-part weight (1 - (K-3)/Q)_+ and shrinks toward the same anchor.
##  In a tree with equal leaf variances the first weight is exactly
##  (1 - (K-1)/Q)_+, so the two rules differ only in a degrees-of-freedom
##  constant, and the Stein constant is the one that carries a finite-sample
##  dominance guarantee.
##
##  REQUIREMENTS: randomForest, glmnet, gbm, matrixStats
################################################################################

suppressPackageStartupMessages({
  library(randomForest)
  library(glmnet)
  library(gbm)
  library(matrixStats)
})

EPS <- 1e-12

## ============================================================================
##  1.  Variance components
## ============================================================================

##' Huber's constant beta_k = E[psi_k(Z)^2] for a standard normal Z.
huber_beta <- function(k = 1.345) {
  2 * (pnorm(k) - 0.5) - 2 * k * dnorm(k) + 2 * k^2 * (1 - pnorm(k))
}

##' Pooled within-leaf scale from residuals, with a multiplicity-aware df.
##'
##' @param resid  within-leaf residuals, one entry per row of the estimation
##'               sample (rows repeated according to their multiplicity)
##' @param df     sum_l (C_l - sum_i c_i^2 / C_l)
##' @param method "pooled", "mad" or "huber"
pooled_scale <- function(resid, df, method = "pooled", k = 1.345) {
  if (df <= 0) return(NA_real_)
  if (method == "pooled") return(sum(resid^2) / df)

  corr <- length(resid) / max(df, 1)
  s <- 1.4826 * median(abs(resid - median(resid)))
  if (!is.finite(s) || s <= 0) return(NA_real_)
  if (method == "mad") return(corr * s^2)
  if (method != "huber") stop("unknown scale method: ", method)

  beta <- huber_beta(k)
  for (it in 1:60) {
    u <- pmax(pmin(resid / s, k), -k)
    s_new <- s * sqrt(mean(u^2) / beta)
    if (!is.finite(s_new) || s_new <= 0) break
    if (abs(s_new - s) <= 1e-8 * s) { s <- s_new; break }
    s <- s_new
  }
  corr * s^2
}

##' Cochran's heterogeneity statistic and the precision-weighted anchor.
cochran_q <- function(leaf_mean, v_leaf) {
  u <- 1 / pmax(v_leaf, EPS)
  theta <- sum(u * leaf_mean) / sum(u)
  list(Q = sum(u * (leaf_mean - theta)^2), theta = theta, u = u)
}

##' DerSimonian-Laird moment estimator of the between-leaf variance.
##'
##' Expanding Q under the random-effects model gives
##'     E[Q] = (K - 1) + tau^2 ( sum_l u_l - sum_l u_l^2 / sum_l u_l ),
##' and the estimator is the positive part of the corresponding solution.
tau2_dl <- function(leaf_mean, v_leaf) {
  K  <- length(leaf_mean)
  cq <- cochran_q(leaf_mean, v_leaf)
  den <- sum(cq$u) - sum(cq$u^2) / sum(cq$u)
  tau2 <- if (den <= EPS) 0 else max((cq$Q - (K - 1)) / den, 0)
  list(tau2 = tau2, theta = cq$theta, Q = cq$Q)
}

##' Unweighted moment estimator, corrected for unequal anchor weights.
##'
##' With theta_hat = sum_l w_l ybar_l and S^2 = sum_l (ybar_l - theta_hat)^2,
##'     E[S^2] = sum_l c_l (tau^2 + v_l),  c_l = 1 - 2 w_l + K w_l^2,
##' so the unbiased estimator divides by sum_l c_l.  Dividing by K - 1 and
##' subtracting sum_l v_l, which is correct only for equal leaf sizes,
##' overstates tau^2 whenever the partition is unbalanced.
tau2_mom <- function(leaf_mean, v_leaf, w) {
  K <- length(leaf_mean)
  theta <- sum(w * leaf_mean)
  dev <- leaf_mean - theta
  cc <- 1 - 2 * w + K * w^2
  list(tau2 = max((sum(dev^2) - sum(cc * v_leaf)) / max(sum(cc), EPS), 0),
       theta = theta)
}

##' Median-based estimator, for trees whose leaf means contain outliers.
tau2_robust <- function(leaf_mean, v_leaf, w) {
  theta <- sum(w * leaf_mean)
  dev <- leaf_mean - theta
  mad <- 1.4826 * median(abs(dev - median(dev)))
  list(tau2 = max(mad^2 - median(v_leaf), 0), theta = theta)
}


## ============================================================================
##  2.  Leaf summaries from a fitted forest
## ============================================================================

##' Ancestor of every node of a tree at a chosen depth.
##'
##' randomForest stores a tree as a data frame whose rows are nodes, and the
##' node identifiers returned by predict(..., nodes = TRUE) index those rows,
##' so the tree can be walked directly.  Nodes at or above the chosen depth are
##' their own ancestors, which makes a leaf shallower than the depth its own
##' group.
node_ancestors <- function(rf, b, depth) {
  tr <- randomForest::getTree(rf, b)
  nl <- tr[, "left daughter"]; nr <- tr[, "right daughter"]
  m <- nrow(tr)
  anc <- integer(m); dep <- integer(m)
  anc[1] <- 1L; dep[1] <- 0L
  stack <- 1L
  while (length(stack)) {
    nd <- stack[length(stack)]; stack <- stack[-length(stack)]
    for (ch in c(nl[nd], nr[nd])) {
      if (ch == 0) next
      dep[ch] <- dep[nd] + 1L
      anc[ch] <- if (dep[ch] <= depth) ch else anc[nd]
      stack <- c(stack, ch)
    }
  }
  anc
}

##' Split the leaves of one tree into anchoring groups.
##'
##' Groups holding fewer than four leaves cannot support the Stein rule, so
##' they are pooled into one residual group rather than left unshrunk.
leaf_groups <- function(rf, b, leaf_ids, anchor = "tree", anchor_depth = 2) {
  K <- length(leaf_ids)
  if (identical(anchor, "tree") || K < 8) return(list(seq_len(K)))
  anc <- node_ancestors(rf, b, anchor_depth)
  key <- anc[leaf_ids]
  gs <- split(seq_len(K), key)
  big <- gs[vapply(gs, length, integer(1)) >= 4]
  small <- unlist(gs[vapply(gs, length, integer(1)) < 4], use.names = FALSE)
  out <- unname(big)
  if (length(small)) out <- c(out, list(small))
  if (!length(out)) out <- list(seq_len(K))
  out
}

##' Shrink one group of leaves under the chosen rule.
shrink_group <- function(lm_g, v_g, w_g, rule, tau_method) {
  Kg <- length(lm_g)
  cq <- cochran_q(lm_g, v_g)
  est <- switch(tau_method,
                dl     = tau2_dl(lm_g, v_g),
                mom    = tau2_mom(lm_g, v_g, w_g),
                robust = tau2_robust(lm_g, v_g, w_g),
                stop("unknown tau method: ", tau_method))
  if (Kg < 3) return(list(shrunk = lm_g, lambda = rep(1, Kg),
                          theta = est$theta, tau2 = est$tau2, Q = cq$Q))
  if (identical(rule, "eb")) {
    lam <- if (est$tau2 <= 0) rep(0, Kg) else est$tau2 / (est$tau2 + v_g)
    return(list(shrunk = est$theta + lam * (lm_g - est$theta), lambda = lam,
                theta = est$theta, tau2 = est$tau2, Q = cq$Q))
  }
  if (Kg < 4 || cq$Q <= EPS)
    return(list(shrunk = lm_g, lambda = rep(1, Kg), theta = cq$theta,
                tau2 = est$tau2, Q = cq$Q))
  l1 <- max(1 - (Kg - 3) / cq$Q, 0)
  list(shrunk = cq$theta + l1 * (lm_g - cq$theta), lambda = rep(l1, Kg),
       theta = cq$theta, tau2 = est$tau2, Q = cq$Q)
}

##' Per-tree leaf statistics computed on the sample that defines the leaves.
##'
##' @param nodes_tr  n x B matrix of terminal-node identifiers for the whole
##'                  training set, from predict(rf, X, nodes = TRUE)
##' @param inbag     n x B matrix of in-bag multiplicities
##' @param y         training responses
##' @param b         tree index
leaf_stats <- function(nodes_tr, inbag, y, b) {
  cnt <- inbag[, b]
  keep <- cnt > 0
  nid <- nodes_tr[keep, b]
  cc  <- cnt[keep]
  yy  <- y[keep]

  f <- factor(nid)
  C   <- as.numeric(tapply(cc, f, sum))
  C2  <- as.numeric(tapply(cc^2, f, sum))
  Sy  <- as.numeric(tapply(cc * yy, f, sum))
  Sy2 <- as.numeric(tapply(cc * yy^2, f, sum))
  leaf_ids <- as.integer(levels(f))

  leaf_mean <- Sy / pmax(C, EPS)
  ## within-leaf weighted sum of squares, computed without forming residuals
  ss <- Sy2 - Sy^2 / pmax(C, EPS)
  list(leaf_ids = leaf_ids, C = C, sum_c2 = C2, leaf_mean = leaf_mean,
       n_eff = C^2 / pmax(C2, EPS), ss = ss,
       df = sum(C - C2 / pmax(C, EPS)),
       node = nid, cc = cc, yy = yy)
}

##' Shrunken leaf values for every tree of a fitted forest.
##'
##' @param rf         a randomForest object fitted with keep.inbag = TRUE
##' @param X,y        the training data the forest was fitted to
##' @param rule       "eb" or "js"
##' @param sigma_method "pooled", "mad" or "huber"
##' @param tau_method   "dl", "mom" or "robust"
##' @param effective_size  FALSE reproduces the naive accounting in which a
##'        bootstrap leaf is credited with its raw count; retained only so the
##'        size of that error can be measured
##' @return a list with one entry per tree, each holding the leaf identifiers,
##'         the raw and shrunken leaf values, and the fitted weights
ssrf_leaf_values <- function(rf, X, y, rule = c("eb", "js"),
                             sigma_method = "pooled", tau_method = "dl",
                             effective_size = TRUE, anchor = "tree",
                             anchor_depth = 2) {
  rule <- match.arg(rule)
  if (is.null(rf$inbag))
    stop("the forest must be fitted with keep.inbag = TRUE")
  nodes_tr <- attr(predict(rf, newdata = X, nodes = TRUE), "nodes")
  inbag <- rf$inbag
  B <- ncol(inbag)

  out <- vector("list", B)
  for (b in seq_len(B)) {
    st <- leaf_stats(nodes_tr, inbag, y, b)
    K <- length(st$leaf_ids)
    n_eff <- if (effective_size) st$n_eff else st$C
    df    <- if (effective_size) st$df else sum(pmax(st$C - 1, 0))

    resid <- st$yy - st$leaf_mean[match(st$node, st$leaf_ids)]
    resid <- rep(resid, st$cc)          # weight by multiplicity
    sigma2 <- pooled_scale(resid, df, sigma_method)
    if (!is.finite(sigma2) || sigma2 <= 0) sigma2 <- var(y)
    sigma2 <- max(sigma2, EPS)

    v_leaf <- sigma2 / pmax(n_eff, 1)
    w <- n_eff / sum(n_eff)

    est <- switch(tau_method,
                  dl     = tau2_dl(st$leaf_mean, v_leaf),
                  mom    = tau2_mom(st$leaf_mean, v_leaf, w),
                  robust = tau2_robust(st$leaf_mean, v_leaf, w),
                  stop("unknown tau method: ", tau_method))
    cq <- cochran_q(st$leaf_mean, v_leaf)

    ## the leaves are shrunk within anchoring groups; with anchor = "tree"
    ## there is a single group and this reduces to the tree-level rule
    groups <- leaf_groups(rf, b, st$leaf_ids, anchor, anchor_depth)
    shrunk <- st$leaf_mean; lam <- rep(1, K)
    th <- numeric(0); wt <- numeric(0)
    for (g in groups) {
      sg <- shrink_group(st$leaf_mean[g], v_leaf[g], w[g] / sum(w[g]),
                         rule, tau_method)
      shrunk[g] <- sg$shrunk; lam[g] <- sg$lambda
      th <- c(th, sg$theta); wt <- c(wt, sum(n_eff[g]))
    }
    est$theta <- sum(wt * th) / sum(wt)

    out[[b]] <- list(leaf_ids = st$leaf_ids, raw = st$leaf_mean,
                     shrunk = shrunk, lambda = lam, n_eff = n_eff,
                     sigma2 = sigma2, tau2 = est$tau2, Q = cq$Q,
                     theta = est$theta, K = K)
  }
  out
}

##' Forest prediction under a chosen leaf rule.
##'
##' @param values "raw" or "shrunk"
ssrf_predict <- function(rf, leafvals, X_new, values = c("shrunk", "raw")) {
  values <- match.arg(values)
  nodes <- attr(predict(rf, newdata = X_new, nodes = TRUE), "nodes")
  B <- length(leafvals)
  acc <- numeric(nrow(nodes))
  for (b in seq_len(B)) {
    lv <- leafvals[[b]]
    j <- match(nodes[, b], lv$leaf_ids)
    v <- lv[[values]][j]
    miss <- is.na(j)
    if (any(miss)) v[miss] <- lv$theta
    acc <- acc + v
  }
  acc / B
}

##' Leaf-size-weighted mean shrinkage weight in each tree.
ssrf_mean_lambda <- function(leafvals)
  vapply(leafvals, function(lv) sum(lv$n_eff / sum(lv$n_eff) * lv$lambda),
         numeric(1))

##' Fit a Breiman forest and return it together with both sets of leaf values.
##'
##' @param B         number of trees
##' @param nodesize  minimum leaf size
fit_ssrf <- function(X, y, B = 500, nodesize = 5, mtry = NULL,
                     rule = "js", sigma_method = "pooled", tau_method = "dl",
                     effective_size = TRUE) {
  if (is.null(mtry)) mtry <- max(1, floor(ncol(X) / 3))
  rf <- randomForest(x = X, y = y, ntree = B, mtry = mtry,
                     nodesize = nodesize, keep.forest = TRUE,
                     keep.inbag = TRUE)
  lv <- ssrf_leaf_values(rf, X, y, rule = rule, sigma_method = sigma_method,
                         tau_method = tau_method,
                         effective_size = effective_size)
  list(rf = rf, leafvals = lv,
       mean_lambda = mean(ssrf_mean_lambda(lv)),
       leaves = mean(vapply(lv, function(z) z$K, numeric(1))),
       n_eff = mean(unlist(lapply(lv, function(z) z$n_eff))))
}

##' Honest forest: the splits of each tree are chosen on one half of a
##' subsample and the leaf values and variance components on the other half.
##' Both halves are drawn without replacement, so every multiplicity is one.
fit_ssrf_honest <- function(X, y, B = 500, nodesize = 3, mtry = NULL,
                            sample_fraction = 0.5, rule = "js",
                            sigma_method = "pooled", tau_method = "dl") {
  X <- as.matrix(X)
  n <- nrow(X)
  if (is.null(mtry)) mtry <- max(1, floor(ncol(X) / 3))
  n_sub <- max(4 * nodesize, floor(sample_fraction * n))
  trees <- vector("list", B)
  for (b in seq_len(B)) {
    idx <- sample.int(n, n_sub, replace = FALSE)
    half <- floor(length(idx) / 2)
    i1 <- idx[seq_len(half)]
    i2 <- idx[(half + 1):length(idx)]
    tb <- randomForest(x = X[i1, , drop = FALSE], y = y[i1], ntree = 1,
                       mtry = mtry, nodesize = nodesize, replace = FALSE,
                       sampsize = length(i1), keep.forest = TRUE)
    nd <- attr(predict(tb, newdata = X[i2, , drop = FALSE], nodes = TRUE),
               "nodes")[, 1]
    f <- factor(nd)
    C <- as.numeric(table(f))
    lm_ <- as.numeric(tapply(y[i2], f, mean))
    ss <- as.numeric(tapply(y[i2], f, function(z) sum((z - mean(z))^2)))
    ids <- as.integer(levels(f))
    K <- length(ids)
    df <- sum(pmax(C - 1, 0))
    sigma2 <- if (df > 0) sum(ss) / df else var(y[i2])
    if (!is.finite(sigma2) || sigma2 <= 0) sigma2 <- var(y)
    v <- sigma2 / pmax(C, 1)
    est <- tau2_dl(lm_, v)
    cq <- cochran_q(lm_, v)
    if (K < 4) {
      shr <- lm_; lam <- rep(1, K)
    } else if (rule == "eb") {
      lam <- if (est$tau2 <= 0) rep(0, K) else est$tau2 / (est$tau2 + v)
      shr <- est$theta + lam * (lm_ - est$theta)
    } else {
      l1 <- max(1 - (K - 3) / max(cq$Q, EPS), 0)
      lam <- rep(l1, K); shr <- cq$theta + l1 * (lm_ - cq$theta)
    }
    trees[[b]] <- list(tree = tb, leaf_ids = ids, raw = lm_, shrunk = shr,
                       lambda = lam, n_eff = C, theta = est$theta, K = K)
  }
  trees
}

##' Prediction from an honest forest built by fit_ssrf_honest.
honest_predict <- function(trees, X_new, values = c("shrunk", "raw")) {
  values <- match.arg(values)
  X_new <- as.matrix(X_new)
  acc <- numeric(nrow(X_new))
  for (tb in trees) {
    nd <- attr(predict(tb$tree, newdata = X_new, nodes = TRUE), "nodes")[, 1]
    j <- match(nd, tb$leaf_ids)
    v <- tb[[values]][j]
    v[is.na(j)] <- tb$theta
    acc <- acc + v
  }
  acc / length(trees)
}


## ============================================================================
##  5.  Out-of-bag machinery, anchor selection and forest-level calibration
## ============================================================================

##' Matrix of the value each tree reports at each row.
forest_per_tree <- function(rf, leafvals, X_new, values = c("shrunk", "raw")) {
  values <- match.arg(values)
  nodes <- attr(predict(rf, newdata = X_new, nodes = TRUE), "nodes")
  out <- matrix(0, nrow(nodes), ncol(nodes))
  for (b in seq_len(ncol(nodes))) {
    lv <- leafvals[[b]]
    j <- match(nodes[, b], lv$leaf_ids)
    v <- lv[[values]][j]
    v[is.na(j)] <- lv$theta
    out[, b] <- v
  }
  out
}

##' Mean squared out-of-bag error under a given leaf rule.
##' Each training row is predicted by the trees whose sample excluded it, so
##' the score is honest for choices made after the forest was grown, such as
##' the anchoring depth.
ssrf_oob_error <- function(rf, leafvals, X, y, values = "shrunk") {
  T <- forest_per_tree(rf, leafvals, X, values)
  oob <- rf$inbag == 0
  cnt <- rowSums(oob)
  use <- cnt > 0
  if (!any(use)) return(NA_real_)
  pred <- rowSums(T * oob)[use] / cnt[use]
  mean((pred - y[use])^2)
}

##' Choose the anchoring depth by out-of-bag error, after growing.
##'
##' Shrinking every leaf toward one tree-level anchor is a poor choice when the
##' partition separates distinct regimes, and shrinking toward a subtree anchor
##' is a poor choice when the subtrees hold too few leaves for the between-leaf
##' variance to be estimable.  Which of the two binds is a property of the
##' data, so it is selected.  Changing the depth changes only the leaf values,
##' so no tree is regrown and each candidate costs one pass over the leaves.
select_anchor <- function(rf, X, y, rule = "js",
                          candidates = list(list("tree", 0),
                                            list("ancestor", 1),
                                            list("ancestor", 2),
                                            list("ancestor", 3)),
                          sigma_method = "pooled", tau_method = "dl") {
  best <- NULL; best_err <- Inf; best_lv <- NULL
  for (cd in candidates) {
    lv <- ssrf_leaf_values(rf, X, y, rule = rule, sigma_method = sigma_method,
                           tau_method = tau_method, anchor = cd[[1]],
                           anchor_depth = cd[[2]])
    err <- ssrf_oob_error(rf, lv, X, y)
    if (is.finite(err) && err < best_err) {
      best_err <- err; best <- cd; best_lv <- lv
    }
  }
  if (is.null(best)) {
    best <- list("tree", 0)
    best_lv <- ssrf_leaf_values(rf, X, y, rule = rule)
  }
  list(anchor = best[[1]], anchor_depth = best[[2]], leafvals = best_lv,
       oob_error = best_err)
}

##' Bias-corrected infinitesimal-jackknife variance of the forest prediction.
##'
##' With N_{b,i} the number of times observation i enters the sample of tree b
##' and t_b(x) the value tree b reports at x,
##'     V_IJ(x) = sum_i [ B^-1 sum_b (N_{b,i} - Nbar_i)(t_b(x) - tbar(x)) ]^2 ,
##' whose Monte-Carlo bias from finitely many trees is n s^2(x) / B with s^2(x)
##' the across-tree variance.  Subtracting that term gives the corrected
##' estimator of Wager, Hastie and Efron, truncated below because the
##' correction can overshoot.
ij_variance <- function(rf, leafvals, X_new) {
  T <- forest_per_tree(rf, leafvals, X_new, "raw")
  B <- ncol(T)
  Tc <- T - rowMeans(T)
  N <- rf$inbag                                   # n x B
  Nc <- N - rowMeans(N)
  cov_ <- Tc %*% t(Nc) / B                        # m x n
  s2 <- rowSums(Tc^2) / B
  n <- nrow(N)
  list(V = pmax(rowSums(cov_^2) - n * s2 / B, s2 / B * 1e-3), s2 = s2, T = T)
}

##' Cross-validated contraction factor and anchor for the forest prediction.
##'
##' The slope of the regression of the response on a prediction that did not
##' see it estimates Cov(m, f) / Var(fhat), which is the factor that minimises
##' forest risk, because the noise in the response is uncorrelated with such a
##' prediction.  No variance model enters and the estimator works at any
##' ensemble size; it costs n_folds forest fits.
calibrate_by_cv <- function(X, y, B, nodesize = 5, mtry = NULL, n_folds = 5) {
  X <- as.matrix(X); n <- nrow(X)
  fold <- sample(rep_len(seq_len(n_folds), n))
  pred <- numeric(n)
  for (k in seq_len(n_folds)) {
    tr <- fold != k; te <- !tr
    if (sum(te) == 0) next
    if (sum(tr) < 10) { pred[te] <- mean(y[tr]); next }
    M <- fit_ssrf(X[tr, , drop = FALSE], y[tr], B = B, nodesize = nodesize,
                  mtry = mtry, rule = "js")
    pred[te] <- ssrf_predict(M$rf, M$leafvals, X[te, , drop = FALSE], "raw")
  }
  v <- var(pred)
  list(lambda = if (v <= EPS) 0 else as.numeric(cov(pred, y) / v),
       theta = mean(y))
}

##' Out-of-bag contraction factor, using the jackknife for the variances.
##'
##' An out-of-bag average uses only the trees that excluded the point, so its
##' variance is not that of the deployed forest; writing s^2(x) for the
##' across-tree variance and B_oob(x) for the number of excluding trees,
##'     V_oob(x) = V_B(x) - s^2(x)/B + s^2(x)/B_oob(x),
##' and that identity transports the slope to the ensemble in use.  The
##' estimator inherits the jackknife's requirement that B be large relative to
##' n, so the cross-validated route is preferred at small ensembles.
calibrate_oob <- function(rf, leafvals, X, y) {
  ijv <- ij_variance(rf, leafvals, X)
  T <- ijv$T; B <- ncol(T)
  oob <- rf$inbag == 0
  n_oob <- rowSums(oob)
  use <- n_oob >= max(3, 0.05 * B)
  if (sum(use) < 10) return(list(lambda = 1, theta = mean(y)))
  f_oob <- rowSums(T * oob)[use] / n_oob[use]
  vo <- var(f_oob)
  if (vo <= EPS) return(list(lambda = 0, theta = mean(y)))
  cov_mf <- as.numeric(cov(f_oob, y[use]))
  V_oob <- ijv$V[use] - ijv$s2[use] / B + ijv$s2[use] / n_oob[use]
  var_m <- max(vo - mean(V_oob), 0)
  list(lambda = min(max(cov_mf / max(var_m + mean(ijv$V), EPS), 0), 2),
       theta = mean(y))
}

##' Prediction from a calibrated forest-level contraction.
calibrated_predict <- function(rf, leafvals, X_new, cal) {
  pr <- ssrf_predict(rf, leafvals, X_new, "raw")
  cal$theta + cal$lambda * (pr - cal$theta)
}


## ============================================================================
##  3.  Competing predictors
## ============================================================================

##' Ridge-rotated forest.
##'
##' Rescaling each feature by its ridge coefficient leaves a CART forest
##' completely unchanged, because the impurity reduction of a split depends
##' only on the induced partition and a positive per-feature rescaling
##' preserves every candidate partition.  A benchmark defined that way is the
##' standard forest under another name.  The forest is therefore grown on a
##' genuine rotation: an orthonormal basis is completed around the
##' cross-validated ridge direction by a Householder reflection, so that
##' splits on the leading coordinate are oblique in the original ones.
householder_basis <- function(v) {
  p <- length(v); nv <- sqrt(sum(v^2))
  if (nv < EPS) return(diag(p))
  e1 <- c(1, rep(0, p - 1))
  u <- v / nv - e1
  nu <- sqrt(sum(u^2))
  if (nu < EPS) return(diag(p))
  u <- u / nu
  diag(p) - 2 * outer(u, u)
}

fit_ridge_rf <- function(X_tr, y_tr, X_te, B = 500, nodesize = 5) {
  Xtr <- as.matrix(X_tr); Xte <- as.matrix(X_te)
  ## with a single predictor a rotation is the identity and the penalised fit
  ## is undefined, so the benchmark is the forest itself
  if (ncol(Xtr) < 2) {
    rf <- randomForest(x = Xtr, y = y_tr, ntree = B, mtry = 1,
                       nodesize = nodesize)
    return(as.numeric(predict(rf, newdata = Xte)))
  }
  sdv <- apply(Xtr, 2, sd); sdv[sdv < EPS] <- 1
  Ztr <- sweep(Xtr, 2, sdv, "/"); Zte <- sweep(Xte, 2, sdv, "/")
  cv <- cv.glmnet(Ztr, y_tr, alpha = 0, nfolds = 5)
  b <- as.numeric(coef(cv, s = "lambda.min"))[-1]
  H <- householder_basis(b)
  rf <- randomForest(x = Ztr %*% H, y = y_tr, ntree = B,
                     mtry = max(1, floor(ncol(Ztr) / 3)), nodesize = nodesize)
  as.numeric(predict(rf, newdata = Zte %*% H))
}

fit_lasso_rf <- function(X_tr, y_tr, X_te, B = 500, nodesize = 5) {
  Xtr <- as.matrix(X_tr); Xte <- as.matrix(X_te)
  ## with a single predictor there is nothing to select
  if (ncol(Xtr) < 2) {
    rf <- randomForest(x = Xtr, y = y_tr, ntree = B, mtry = 1,
                       nodesize = nodesize)
    return(as.numeric(predict(rf, newdata = Xte)))
  }
  cv <- cv.glmnet(Xtr, y_tr, alpha = 1, nfolds = 5)
  sel <- which(as.numeric(coef(cv, s = "lambda.min"))[-1] != 0)
  if (length(sel) < 2)
    sel <- order(abs(cor(Xtr, y_tr)), decreasing = TRUE)[1:min(5, ncol(Xtr))]
  rf <- randomForest(x = Xtr[, sel, drop = FALSE], y = y_tr, ntree = B,
                     mtry = max(1, floor(length(sel) / 3)), nodesize = nodesize)
  as.numeric(predict(rf, newdata = Xte[, sel, drop = FALSE]))
}

##' Gradient-boosted trees with the number of rounds chosen by cross-validation.
##' Running a fixed large number of rounds in a low signal-to-noise problem
##' overfits, and reporting that would understate what boosting can do.
fit_gbm <- function(X_tr, y_tr, X_te, n_trees = 500) {
  ## column names are replaced by neutral ones so that the training and test
  ## frames cannot disagree after R sanitises names that came from a
  ## model matrix
  X_tr <- as.matrix(X_tr); X_te <- as.matrix(X_te)
  cn <- paste0("V", seq_len(ncol(X_tr)))
  colnames(X_tr) <- cn; colnames(X_te) <- cn
  ## rows are permuted because gbm takes its validation block as the first
  ## train.fraction of the rows in the order supplied
  ord <- sample.int(nrow(X_tr))
  d <- data.frame(y = y_tr[ord], as.data.frame(X_tr[ord, , drop = FALSE]))
  n <- nrow(d)
  ## small samples cannot support a held-out block and the default node size,
  ## so both are scaled down and early stopping is turned off rather than
  ## letting the fit fail
  small <- n < 80
  tf   <- if (small) 1.0 else 0.75
  bagf <- if (small) 0.9 else 0.7
  minobs <- max(2L, min(10L, floor(n * tf * bagf / 3)))
  invisible(capture.output(suppressWarnings(
    g <- gbm(y ~ ., data = d, distribution = "gaussian", n.trees = n_trees,
             interaction.depth = min(4, max(1, ncol(X_tr))), shrinkage = 0.05,
             bag.fraction = bagf, train.fraction = tf, cv.folds = 0,
             n.minobsinnode = minobs, verbose = FALSE, n.cores = 1))))
  best <- if (small) n_trees else
    suppressWarnings(gbm.perf(g, method = "test", plot.it = FALSE))
  if (!is.finite(best) || best < 1) best <- n_trees
  as.numeric(predict(g, newdata = as.data.frame(X_te), n.trees = best))
}

##' Forest with the leaf size and split width chosen by cross-validation.
##' Enlarging the leaf is the standard way of reducing leaf-level estimation
##' variance, so tuning it is the fair comparison for a method whose stated
##' purpose is to reduce that same variance.
fit_rf_tuned <- function(X_tr, y_tr, X_te, B = 500, n_folds = 3, n_cv = 40) {
  Xtr <- as.matrix(X_tr); n <- nrow(Xtr); p <- ncol(Xtr)
  grid <- expand.grid(nodesize = c(2, 5, 10, 20, 40, 80),
                      mtry = unique(c(max(1, floor(p / 3)), max(1, floor(p / 2)))))
  fold <- sample(rep_len(seq_len(n_folds), n))
  err <- numeric(nrow(grid))
  for (g in seq_len(nrow(grid))) {
    e <- 0
    for (k in seq_len(n_folds)) {
      tr <- fold != k; va <- !tr
      rf <- randomForest(x = Xtr[tr, , drop = FALSE], y = y_tr[tr],
                         ntree = n_cv, mtry = grid$mtry[g],
                         nodesize = grid$nodesize[g])
      e <- e + mean((predict(rf, Xtr[va, , drop = FALSE]) - y_tr[va])^2)
    }
    err[g] <- e
  }
  best <- grid[which.min(err), ]
  rf <- randomForest(x = Xtr, y = y_tr, ntree = B, mtry = best$mtry,
                     nodesize = best$nodesize)
  list(pred = as.numeric(predict(rf, newdata = as.matrix(X_te))),
       nodesize = best$nodesize, mtry = best$mtry)
}

##' Local linear forest: forest kernel weights combined with a local ridge fit.
##'
##' A regression forest defines the data-adaptive kernel
##'     alpha_i(x) = B^{-1} sum_b 1{X_i in L_b(x)} / |L_b(x)| ,
##' and the local linear forest replaces the weighted average of the responses
##' by the intercept of the weighted ridge fit of Y on (X - x).  It is the
##' closest competitor in spirit, because it too changes what a leaf reports
##' without moving any split.
fit_llf <- function(X_tr, y_tr, X_te, B = 200, nodesize = 10,
                    lambdas = c(0.01, 0.1, 1, 10)) {
  Xtr <- as.matrix(X_tr); Xte <- as.matrix(X_te)
  n <- nrow(Xtr); p <- ncol(Xtr)
  if (n < 4 * (p + 2)) {
    ## too few observations for a local linear fit to be identified
    rf <- randomForest(x = Xtr, y = y_tr, ntree = B,
                       mtry = max(1, floor(p / 3)), nodesize = nodesize)
    return(as.numeric(predict(rf, newdata = Xte)))
  }
  hold <- sample.int(n, max(10, floor(0.2 * n)))
  fitset <- setdiff(seq_len(n), hold)
  rf <- randomForest(x = Xtr[fitset, , drop = FALSE], y = y_tr[fitset],
                     ntree = B, mtry = max(1, floor(p / 3)),
                     nodesize = nodesize, keep.forest = TRUE)
  nodes_tr <- attr(predict(rf, Xtr[fitset, , drop = FALSE], nodes = TRUE), "nodes")
  sdv <- apply(Xtr[fitset, , drop = FALSE], 2, sd); sdv[sdv < EPS] <- 1

  weights_for <- function(Xq) {
    nq <- attr(predict(rf, Xq, nodes = TRUE), "nodes")
    W <- matrix(0, nrow(Xq), length(fitset))
    for (b in seq_len(ncol(nq))) {
      tab <- table(nodes_tr[, b])
      ids <- as.integer(names(tab)); cnt <- as.numeric(tab)
      j <- match(nq[, b], ids)
      for (i in which(!is.na(j))) {
        sel <- nodes_tr[, b] == ids[j[i]]
        W[i, sel] <- W[i, sel] + 1 / cnt[j[i]]
      }
    }
    W / ncol(nq)
  }

  local_fit <- function(Xq, lam) {
    W <- weights_for(Xq)
    Zs <- sweep(Xtr[fitset, , drop = FALSE], 2, sdv, "/")
    pen <- lam * diag(p + 1); pen[1, 1] <- 0
    vapply(seq_len(nrow(Xq)), function(i) {
      w <- W[i, ]; nz <- which(w > 0)
      if (length(nz) < p + 2)
        return(if (sum(w) > 0) sum(w * y_tr[fitset]) / sum(w)
               else mean(y_tr[fitset]))
      D <- cbind(1, sweep(Zs[nz, , drop = FALSE], 2, Xq[i, ] / sdv, "-"))
      A <- crossprod(D, D * w[nz]) + pen
      bvec <- crossprod(D, w[nz] * y_tr[fitset][nz])
      out <- tryCatch(solve(A, bvec)[1], error = function(e) NA_real_)
      if (is.na(out)) sum(w[nz] * y_tr[fitset][nz]) / sum(w[nz]) else out
    }, numeric(1))
  }

  errs <- vapply(lambdas, function(l)
    mean((local_fit(Xtr[hold, , drop = FALSE], l) - y_tr[hold])^2), numeric(1))
  local_fit(Xte, lambdas[which.min(errs)])
}


## ============================================================================
##  4.  Evaluation
## ============================================================================

mse_of <- function(p, y) mean((p - y)^2)
mae_of <- function(p, y) mean(abs(p - y))

##' Nonparametric bootstrap standard error of a median.
##'
##' The asymptotic formula 1.253 s / sqrt(R) assumes a normal sampling
##' distribution and misbehaves badly when the replication-level losses are
##' heavy-tailed, which is the regime these simulations sit in.
boot_median_se <- function(x, n_boot = 2000) {
  x <- x[is.finite(x)]
  if (length(x) < 2) return(NA_real_)
  m <- replicate(n_boot, median(sample(x, length(x), replace = TRUE)))
  sd(m)
}

##' Diebold-Mariano test with the Harvey-Leybourne-Newbold correction.
##'
##' The loss differential is d = loss_a - loss_b, so a positive mean favours
##' method b.  The plain statistic is oversized in small samples, so the
##' correction sqrt([T + 1 - 2h + h(h-1)/T] / T) is applied and the reference
##' distribution is Student's t on T - 1 degrees of freedom.
dm_test <- function(loss_a, loss_b, h = 1) {
  d <- loss_a - loss_b
  d <- d[is.finite(d)]
  T <- length(d)
  if (T < 8) return(c(stat = NA_real_, p = NA_real_))
  dbar <- mean(d); dc <- d - dbar
  lrv <- sum(dc^2) / T
  if (h > 1) for (k in 1:(h - 1))
    lrv <- lrv + 2 * (1 - k / h) * sum(dc[(k + 1):T] * dc[1:(T - k)]) / T
  if (lrv <= 0) return(c(stat = NA_real_, p = NA_real_))
  st <- dbar / sqrt(lrv / T) * sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
  c(stat = st, p = 2 * pt(-abs(st), df = T - 1))
}

##' Split conformal prediction intervals with absolute-residual scores.
##' Out-of-bag residual quantiles borrowed from one method and applied to all
##' of them are not calibrated for any of them; these are, per method.
split_conformal <- function(pred_cal, y_cal, pred_test, alpha = 0.05) {
  r <- abs(y_cal - pred_cal)
  n <- length(r)
  k <- ceiling((n + 1) * (1 - alpha))
  q <- if (k > n) max(r) else sort(r)[k]
  list(lower = pred_test - q, upper = pred_test + q)
}

##' Mean interval score of Gneiting and Raftery, negatively oriented.
interval_score <- function(lower, upper, y, alpha = 0.05) {
  mean((upper - lower) + (2 / alpha) * (pmax(lower - y, 0) + pmax(y - upper, 0)))
}

##' Sign-based strategy profit and loss, net of round-trip trading costs.
##' A position flip is charged on the day of the flip; charging it once and
##' then dividing by the length of the sample makes the strategy effectively
##' costless and overstates the net result.
##' @param cost  round-trip cost of a position flip, in the units of y.  The
##'   targets used here are daily log returns in percent, so two basis points
##'   is 0.02, and passing 2 would charge two percent a flip.
net_pnl <- function(pred, y, cost = 0.02) {
  pos <- sign(pred); pos[pos == 0] <- 1
  gross <- pos * y
  turn <- abs(c(0, diff(pos))) / 2
  net <- gross - cost * turn
  s <- sd(net)
  list(cum = cumsum(net), sharpe = if (s > 0) mean(net) / s * sqrt(252) else NA,
       mean = mean(net), turnover = sum(turn))
}

##' Holm step-down adjustment, returned in the input order.
holm_adjust <- function(p) p.adjust(p, method = "holm")
