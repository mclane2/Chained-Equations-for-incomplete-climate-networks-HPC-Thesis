
/**
 * Filename: elastic_net_R_wrapper.c
 * 
 * Description: This is the R SEXP wrapper so that R can call ence_impute_cycle implemented in C.
 * Unpacks R objects into plain C types, runs the fit, and packs results back into an R list.
 * 
 * Author: M. Lane
 * Version: 6.0
 * Date: 2026-06-25
 */
#include <R.h>
#include <Rinternals.h>
#include "ENCE_impute.h"


SEXP mice_impute_cycle_R(SEXP df_old_sexp, SEXP missing_sexp, SEXP folds_sexp, SEXP lambdas_sexp, SEXP alphas_sexp,
                         SEXP nfolds_sexp, SEXP thresh_sexp, SEXP maxit_sexp, SEXP nthreads_sexp, SEXP lambda_io_sexp, SEXP alpha_io_sexp,
                         SEXP noise_sexp, SEXP boot_sexp)
{
    int n = nrows(df_old_sexp);
    int p = ncols(df_old_sexp);

    /* Output matrix df_new */
    SEXP df_new_sexp = PROTECT(allocMatrix(REALSXP, n, p));

    /* Output betas */
    SEXP betas_sexp = PROTECT(allocMatrix(REALSXP, p, p));

    /* Duplicate the in/out hyperparameter vectors */
    SEXP lambda_out = PROTECT(duplicate(lambda_io_sexp));
    SEXP alpha_out  = PROTECT(duplicate(alpha_io_sexp));

    int failed = ence_impute_cycle(REAL(df_old_sexp), REAL(df_new_sexp), INTEGER(missing_sexp), INTEGER(folds_sexp), n, p,
                      REAL(lambdas_sexp), LENGTH(lambdas_sexp), REAL(alphas_sexp), LENGTH(alphas_sexp),
                      asInteger(nfolds_sexp), asReal(thresh_sexp), asInteger(maxit_sexp), asInteger(nthreads_sexp),
                      REAL(lambda_out), REAL(alpha_out),
                      REAL(noise_sexp), REAL(betas_sexp), INTEGER(boot_sexp));
    if (failed) Rf_error("Malloc Error in ence_impute_cycle");

    /* Pack into a named list: df, lambda, alpha, betas */
    SEXP result = PROTECT(allocVector(VECSXP, 4));
    SEXP names  = PROTECT(allocVector(STRSXP, 4));

    /* Fill list */
    SET_VECTOR_ELT(result, 0, df_new_sexp);
    SET_VECTOR_ELT(result, 1, lambda_out);
    SET_VECTOR_ELT(result, 2, alpha_out);
    SET_VECTOR_ELT(result, 3, betas_sexp);

    /* Attach labels */
    SET_STRING_ELT(names, 0, mkChar("df"));
    SET_STRING_ELT(names, 1, mkChar("lambda"));
    SET_STRING_ELT(names, 2, mkChar("alpha"));
    SET_STRING_ELT(names, 3, mkChar("betas"));
    /* Attach label to the list */
    setAttrib(result, R_NamesSymbol, names);

    // Unprotect the allocated vectors
    UNPROTECT(6);
    return result;
}