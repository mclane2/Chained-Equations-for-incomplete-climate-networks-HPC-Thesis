
/**
 * Filename: elastic_net_R_wrapper.c
 * 
 * Description: This is the R SEXP wrapper so that R can call the elastic net function implemented in C.
 * Unpacks R objects into plain C types, runs the fit, and packs results back into an R list.
 * 
 * Author: M. Lane
 * Version: 1.0
 * Date: 2026-05-15
 */
#include <R.h>
#include <Rinternals.h>
#include "elastic_net.h"

SEXP elastic_net_fit_R(SEXP Z_sexp, SEXP z_sexp, SEXP lambda_sexp, SEXP alpha_sexp, SEXP thresh_sexp, SEXP maxit_sexp)
{
    int n = nrows(Z_sexp);
    int p = ncols(Z_sexp);

    /* beta is returned to R, so allocate as an R vector */
    SEXP beta_sexp = PROTECT(allocVector(REALSXP, p));
    double intercept;

    elastic_net_fit(REAL(Z_sexp), REAL(z_sexp), n, p, asReal(lambda_sexp), asReal(alpha_sexp), asReal(thresh_sexp), asInteger(maxit_sexp), REAL(beta_sexp), &intercept);

    SEXP result = PROTECT(allocVector(VECSXP, 2));
    SEXP names  = PROTECT(allocVector(STRSXP, 2));
    SET_VECTOR_ELT(result, 0, ScalarReal(intercept));
    SET_VECTOR_ELT(result, 1, beta_sexp);
    SET_STRING_ELT(names, 0, mkChar("intercept"));
    SET_STRING_ELT(names, 1, mkChar("beta"));
    setAttrib(result, R_NamesSymbol, names);

    UNPROTECT(3);
    return result;
}