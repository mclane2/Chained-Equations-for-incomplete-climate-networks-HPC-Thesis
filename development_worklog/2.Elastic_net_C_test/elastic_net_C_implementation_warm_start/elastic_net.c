/**
 * Filename: elastic_net.c
 * 
 * Description: Implements elastic net regression fit similar to glmnet 
 * (Driver for functions in elastic_net_functions.c)
 * 
 * Version 2 - Added warm start optimization
 * 
 * Author: M. Lane
 * Version: 2.0
 * Date: 2026-06-07
 */

#include <stdlib.h>
#include <string.h>
#include "elastic_net_functions.h"
#include "elastic_net.h"

/* Elastic-net regularisation for a fixed alpha.
 *
 * Fits the whole lambda sequence in one pass, warm-starting each solve from
 * the previous lambda.
 */
int elastic_net_path(const double *Z_in, const double *z_in, int n, int p,
                     const double *lambdas, int n_lambda, double alpha,
                     double thresh, int maxit, double *beta_out, double *intercept_out)
{
    /* Allocating memory for buffers and copies */
    double *Z    = malloc((size_t)n * p * sizeof(double));
    double *z    = malloc((size_t)n * sizeof(double));
    double *r    = malloc((size_t)n * sizeof(double));
    double *Zm   = malloc((size_t)p * sizeof(double));
    double *Zs   = malloc((size_t)p * sizeof(double));
    double *beta = calloc((size_t)p, sizeof(double)); 

    /* Allocation error check */
    if (!Z || !z || !r || !Zm || !Zs || !beta) {
        free(Z); free(z); free(r); free(Zm); free(Zs); free(beta);
        return -1;
    }

    /* Making copies of data */
    memcpy(Z, Z_in, (size_t)n * p * sizeof(double));
    memcpy(z, z_in, (size_t)n * sizeof(double));

    /* Standardise  transformation once for the whole path */
    double zm, zs;
    standardise(Z, z, n, p, Zm, Zs, &zm, &zs);

    /* Residual starts as z for the first sweep (r = z) */
    memcpy(r, z, (size_t)n * sizeof(double));

    /* Loop over lambdas */
    int status = 0;
    for (int l = 0; l < n_lambda; ++l) {

        /* beta and r carry over from the previous lambda  (warm start)*/
        status |= coord_descent(Z, r, beta, n, p, lambdas[l] / zs, alpha, thresh, maxit);

        /* Copy out, then back-transform the copy so working beta stays standardised */
        double *bcol = beta_out + (size_t)l * p;
        memcpy(bcol, beta, (size_t)p * sizeof(double));
        intercept_out[l] = back_transform(bcol, Zm, Zs, zm, zs, p);
    }

    free(Z); free(z); free(r); free(Zm); free(Zs); free(beta);
    return status;
}