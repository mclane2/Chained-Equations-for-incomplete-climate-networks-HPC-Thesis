/**
 * Filename: elastic_net.c
 * 
 * Description: Implements elastic net regression fit similar to glmnet 
 * (Driver for functions in elastic_net_functions.c)
 * 
 * Author: M. Lane
 * Version: 1.0
 * Date: 2026-05-15
 */

#include <stdlib.h>
#include <string.h>
#include "elastic_net_functions.h"
#include "elastic_net.h"

int elastic_net_fit(const double *Z_in, const double *z_in, int n, int p, double lambda, double alpha, double thresh, int maxit, double *beta_out, double *intercept_out)
{
    /* Allocating memory for buffers and copies (standardise modifies in place) */
    double *Z = malloc((size_t)n * p * sizeof(double));
    double *z = malloc(n * sizeof(double));
    double *r = malloc(n * sizeof(double));
    double *Zm = malloc(p * sizeof(double));
    double *Zs = malloc(p * sizeof(double));

    /* Allocation error check */
    if (!Z || !z || !r || !Zm || !Zs) {
        free(Z); free(z); free(r); free(Zm); free(Zs);
        return -1;
    }

    /* Making copies of data */
    memcpy(Z, Z_in, (size_t)n * p * sizeof(double));
    memcpy(z, z_in, (size_t)n * sizeof(double));
    memset(beta_out, 0, (size_t)p * sizeof(double));

    /* Standardisation transformation */
    double zm, zs;
    standardise(Z, z, n, p, Zm, Zs, &zm, &zs);

    /* Residual starts as z for the first sweep (r = z) */
    memcpy(r, z, (size_t)n * sizeof(double));

    /* coord_descent() returns 0 if converged, 1 if maximum number of iterations was hit */
    int status = coord_descent(Z, r, beta_out, n, p, lambda / zs, alpha, thresh, maxit);

    /* back_trasnform() transforms standardised coefficients back to original-scale coefficients and returns the intercept */
    *intercept_out = back_transform(beta_out, Zm, Zs, zm, zs, p);

    /* Clean up memory*/
    free(Z); free(z); free(r); free(Zm); free(Zs);

    return status;
}