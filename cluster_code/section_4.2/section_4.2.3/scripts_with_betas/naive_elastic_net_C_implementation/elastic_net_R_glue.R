# Filename: elastic_net_R_glue.R
#
# Description: Loads the compiled C shared library (elastic_net.so)
# and exposes R functions that wrap the underlying C implementation via .Call.
#
# Author: M. Lane
# Version: 2.0
# Date: 2026-05-29

dyn.load("scripts_with_betas/naive_elastic_net_C_implementation/naive_ENCE_impute.so")

# Build the n x p CV fold matrix, reproducing the per-column seed/sample logic.
# Observed rows carry their fold id
make_folds <- function(missing_idx, nfolds = 10L, seed = 323) {
  folds <- matrix(0L, nrow(missing_idx), ncol(missing_idx))
  for (col in 1:ncol(missing_idx)) {

    obs   <- !missing_idx[, col]
    n_obs <- sum(obs)

    if (n_obs > 0) {
      set.seed(seed)
      folds[obs, col] <- sample(rep(1:nfolds, length.out = n_obs))
    }
  }
  folds
}

# One synchronous imputation cycle, parallel across stations in C.
# Returns list(df, lambda, alpha).
naive_ence_impute_cycle <- function(df_old, missing_idx, folds, lambda_io, alpha_io, 
                              lambdas = c(0.05, 0.1, 0.15, 0.2),
                              alphas  = c(0.1, 0.35, 0.65, 0.9),
                              nfolds  = 10L, thresh = 1e-7, maxit = 100000L,
                              nthreads = 0L) {
  df_mat <- as.matrix(df_old);      storage.mode(df_mat) <- "double"
  miss   <- as.matrix(missing_idx); storage.mode(miss)   <- "integer"
  fld    <- as.matrix(folds);       storage.mode(fld)    <- "integer"

  lam <- as.double(lambda_io); lam[is.na(lam)] <- -1.0
  alp <- as.double(alpha_io);  alp[is.na(alp)] <- -1.0

  .Call("naive_ence_impute_cycle_R", df_mat, miss, fld,
        as.double(lambdas), as.double(alphas),
        as.integer(nfolds), as.double(thresh), as.integer(maxit),
        as.integer(nthreads), lam, alp, PACKAGE = "naive_ENCE_impute")
}