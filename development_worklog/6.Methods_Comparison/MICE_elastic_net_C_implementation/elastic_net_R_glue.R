# Filename: elastic_net_R_glue.R
#
# Description: Loads the compiled C shared library (elastic_net.so)
# and exposes R functions that wrap the underlying C implementation via .Call.
#
# Author: M. Lane
# Version: 5.0
# Date: 2026-06-19

dyn.load("MICE_elastic_net_C_implementation/ENCE_impute.so")

durr_impute_cycle <- function(df_old, missing_idx, folds, lambda_io, alpha_io, boot, noise,
                            lambdas = c(0.2,0.15,0.10,0.05), alphas = c(0.1,0.35,0.65,0.9),
                            nfolds = 10L, thresh = 1e-7, maxit = 100000L, nthreads = 0L) {
  df_mat <- as.matrix(df_old);      storage.mode(df_mat) <- "double"
  miss   <- as.matrix(missing_idx); storage.mode(miss)   <- "integer"
  fld    <- as.matrix(folds);       storage.mode(fld)    <- "integer"
  bt     <- as.matrix(boot);        storage.mode(bt)     <- "integer"
  nz     <- as.matrix(noise);       storage.mode(nz)     <- "double"

  lam <- as.double(lambda_io); lam[is.na(lam)] <- -1.0
  alp <- as.double(alpha_io);  alp[is.na(alp)] <- -1.0

  .Call("ence_impute_cycle_R", df_mat, miss, fld,
        as.double(lambdas), as.double(alphas),
        as.integer(nfolds), as.double(thresh), as.integer(maxit),
        as.integer(nthreads), lam, alp, nz, bt)
}