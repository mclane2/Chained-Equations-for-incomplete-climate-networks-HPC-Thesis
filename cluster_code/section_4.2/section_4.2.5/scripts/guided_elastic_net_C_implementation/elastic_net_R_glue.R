# Filename: elastic_net_R_glue.R
#
# Description: Loads the compiled C shared library (elastic_net.so)
# and exposes R functions that wrap the underlying C implementation via .Call.
#
# Author: M. Lane
# Version: 5.0
# Date: 2026-06-19

dyn.load("scripts/static_elastic_net_C_implementation/static_ENCE_impute.so")

# One synchronous imputation cycle, parallel across stations in C.
# Returns list(df, lambda, alpha).
static_ence_impute_cycle <- function(df_old, missing_idx, folds, lambda_io, alpha_io, 
                              lambdas = c(0.2, 0.15, 0.10, 0.05),
                              alphas  = c(0.1, 0.35, 0.65, 0.9),
                              nfolds  = 10L, thresh = 1e-7, maxit = 100000L,
                              nthreads = 0L) {

  df_mat <- as.matrix(df_old)
  miss   <- as.matrix(missing_idx)
  fld    <- as.matrix(folds)
  storage.mode(df_mat) <- "double"
  storage.mode(miss)   <- "integer"
  storage.mode(fld)    <- "integer"

  lam <- as.double(lambda_io); lam[is.na(lam)] <- -1.0
  alp <- as.double(alpha_io);  alp[is.na(alp)] <- -1.0

  .Call("static_ence_impute_cycle_R", df_mat, miss, fld,
        as.double(lambdas), as.double(alphas),
        as.integer(nfolds), as.double(thresh), as.integer(maxit),
        as.integer(nthreads), lam, alp, PACKAGE = "static_ENCE_impute")
}