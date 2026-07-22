/**
 * Filename: elastic_net.c
 * 
 * Description: Implements elastic net regression fit similar to glmnet 
 * (Driver for functions in elastic_net_functions.c)
 * 
 * Version 2 - Added warm start optimization
 * Version 3 - Added edits for active set optimization
 * Version 4 - Added covariance updates
 * 
 * Author: M. Lane
 * Version: 4.0
 * Date: 2026-07-22
 */

#include <stdlib.h>
#include <string.h>
#include "elastic_net_functions.h"
#include "elastic_net.h"

/* Elastic-net regularisation for a fixed alpha, using covariance updates.
 *
 *
 */
int elastic_net_path_cov(const double *Z_in, const double *z_in, int n, int p, const double *lambdas, int n_lambda, double alpha,
                          double thresh, int maxit, double *beta_out, double *intercept_out, double *C, unsigned char *has_row)
{
    /* Allocating memory for buffers and copies */
    double *Z    = malloc((size_t)n * p * sizeof(double));
    double *z    = malloc((size_t)n * sizeof(double));
    double *Zm   = malloc((size_t)p * sizeof(double));
    double *Zs   = malloc((size_t)p * sizeof(double));
    double *beta = calloc((size_t)p, sizeof(double));
    int *is_active  = malloc((size_t)p * sizeof(int));
    int *active_idx = malloc((size_t)p * sizeof(int));
    double *g = malloc((size_t)p * sizeof(double));

    /* Allocation error check */
    if (!Z || !z || !Zm || !Zs || !beta || !is_active || !active_idx || !g) {
        free(Z); free(z); free(Zm); free(Zs); free(beta); free(is_active); free(active_idx); free(g);
        return -1;
    }

    /* Making copies of data */
    memcpy(Z, Z_in, (size_t)n * p * sizeof(double));
    memcpy(z, z_in, (size_t)n * sizeof(double));

    /* Standardise  transformation once for the whole path */
    double zm, zs;
    standardise(Z, z, n, p, Zm, Zs, &zm, &zs);

    /* g = c when all beta = 0 */
    for (int k = 0; k < p; ++k) {

        const double *col_k = Z + k * n;  // pointer to the k-th column of Z

        /*Initialise g = c */
        double s = 0.0;
        #pragma omp simd reduction(+:s)
        for (int i = 0; i < n; ++i){
            s += col_k[i] * z[i];
        }
        g[k] = s;
    }

    /* Loop over lambdas */
    int status = 0;
    for (int l = 0; l < n_lambda; ++l) {

        /* beta and g carry over from the previous lambda (warm start).*/
        status |= coord_descent_cov(Z, beta, n, p, lambdas[l] / zs, alpha, thresh, maxit, is_active, active_idx, g, C, has_row);

        /* Copy out, then back-transform the copy so beta stays standardised */
        double *bcol = beta_out + (size_t)l * p;
        memcpy(bcol, beta, (size_t)p * sizeof(double));
        intercept_out[l] = back_transform(bcol, Zm, Zs, zm, zs, p);
    }

    free(Z); free(z); free(Zm); free(Zs); free(beta); free(is_active); free(active_idx); free(g);
    return status;
}