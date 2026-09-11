################################################################################
##  rebuild_gold_tables.R
##
##  Rebuilds the gold-return tables from the fitted predictions saved by
##  realdata_code.R, without refitting anything.  Use it after a change to a
##  scoring rule, such as the units of the trading cost, so that a correction
##  to how results are reported does not require the whole rolling-origin
##  evaluation to be run again.
##
##  USAGE
##      Rscript rebuild_gold_tables.R
################################################################################

.script_dir <- function() {
  a <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(a)) return(dirname(sub("^--file=", "", a[1])))
  for (cand in c("R", ".")) if (file.exists(file.path(cand, "ssrf_core.R")))
    return(cand)
  "."
}
source(file.path(.script_dir(), "ssrf_core.R"))

OUT_DIR <- file.path("outputs", "real")
obj <- readRDS(file.path(OUT_DIR, "gold_results.rds"))
DATA_SOURCE <- obj$source
SOURCE_NOTE <- if (identical(DATA_SOURCE, "calibrated substitute"))
  paste0("These figures were produced from a series calibrated to the ",
         "published moments of daily gold log returns, because the ",
         "market-data download was unavailable in the environment that ran ",
         "the script. They are not estimates from the market data.") else ""

fmt <- function(x, d = 2)
  ifelse(is.na(x), "---", sprintf(paste0("%.", d, "f"), x))

gold_table <- function(res, caption, label, note = "") {
  meth <- names(res$preds); y <- res$y
  mse <- vapply(res$preds, function(p) mse_of(p, y), numeric(1))
  mae <- vapply(res$preds, function(p) mae_of(p, y), numeric(1))
  half <- seq_len(floor(length(y) / 2))
  isc <- vapply(res$preds, function(p) {
    ci <- split_conformal(p[half], y[half], p[-half])
    interval_score(ci$lower, ci$upper, y[-half])
  }, numeric(1))
  ## the targets are daily log returns in percent, so a two basis point
  ## round-trip cost is 0.02 in those units
  pnl <- lapply(res$preds, function(p) net_pnl(p, y, cost = 0.02))
  others <- setdiff(meth, "SSRF-JS")
  dmp <- vapply(others, function(m)
    dm_test((res$preds[[m]] - y)^2, (res$preds$`SSRF-JS` - y)^2)["p"],
    numeric(1))
  dmp <- stats::p.adjust(dmp, method = "holm")
  rows <- vapply(meth, function(m)
    paste0(gsub("-", "--", m), " & ", fmt(mse[[m]], 4), " & ",
           fmt(mae[[m]], 4), " & ", fmt(isc[[m]], 3), " & ",
           fmt(pnl[[m]]$sharpe, 3), " & ",
           if (m == "SSRF-JS") "(ref.)" else fmt(dmp[[m]], 3), " \\\\"),
    character(1))
  paste0("\\begin{table}[htbp]\n\\centering\n\\caption{", caption, "}\n",
         "\\label{", label, "}\n\\small\n\\setlength{\\tabcolsep}{5pt}\n",
         "\\begin{tabular}{lrrrrr}\n\\toprule\n",
         "Method & MSE & MAE & Interval score & Sharpe & Holm $p$ \\\\\n",
         "\\midrule\n", paste(rows, collapse = "\n"),
         "\n\\bottomrule\n\\end{tabular}\n",
         if (nzchar(note))
           paste0("\\\\[2pt]\n\\parbox{\\textwidth}{\\footnotesize ", note, "}\n")
         else "", "\\end{table}\n")
}

for (B in names(obj$results)) {
  writeLines(gold_table(obj$results[[B]]$gold,
    sprintf(paste0("Rolling-origin out-of-sample performance for the ",
                   "next-day gold log return, in percent, with ensembles of ",
                   "$B=%s$ trees. The Sharpe ratio is that of a sign-based ",
                   "strategy charged two basis points on the day a position ",
                   "flips. The last column reports Holm-adjusted ",
                   "Diebold--Mariano $p$-values against SSRF--JS."), B),
    sprintf("tab:gold_B%s", B), SOURCE_NOTE),
    file.path(OUT_DIR, sprintf("Table_Real_Gold_B%s.tex", B)))
  writeLines(gold_table(obj$results[[B]]$bic,
    sprintf(paste0("Rolling-origin out-of-sample performance for the ",
                   "Gold/USD--EUR bicurrency spread, in percent, with ",
                   "ensembles of $B=%s$ trees."), B),
    sprintf("tab:bic_B%s", B), SOURCE_NOTE),
    file.path(OUT_DIR, sprintf("Table_Real_Bicurrency_B%s.tex", B)))
}
cat("Rebuilt gold tables from", DATA_SOURCE, "\n")
