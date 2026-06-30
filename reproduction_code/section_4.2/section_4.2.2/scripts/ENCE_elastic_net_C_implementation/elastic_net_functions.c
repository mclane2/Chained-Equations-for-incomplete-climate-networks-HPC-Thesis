/**
 * Filename: elastic_net_functions.c
 * 
 * Description: This file contains the functions called in elastic_net.C
 * update_beta()    - Updates beta_k for coordinate update for elastic net, called only by coord_descent()
 * coord_descent()  - Main function for cyclical coordinate descent algorithm for elastic-net regression
 * standardise()    - Performs standardisation transformation on matrix Z and target vector z
 * back_transform() - Transforms standardised-scale coefficients to original-scale coefficients and also computes and returns the intercept.
 * 
 * Author: M. Lane
 * Version: 3.0 (v.2 Added active set cycling) + (v.3 Added SIMD vectorization pragmas)
 * Date: 2026-06-12
 */
#include <math.h>
#include <string.h>

#include "elastic_net_functions.h"


/* Beta_k coordinate update for elastic net.
 *
 * Given the partial residual gradient g_k and the current beta_k, computes the new beta_k and returns the change
 *
 * l1 = lambda * alpha 
 * l2 = lambda * (1 - alpha)
 * 
 */
static inline double update_beta(double *beta_k, double g_k, double l1, double l2)
{
    double beta_old = *beta_k;
    double u = g_k + beta_old;        // partial-residual gradient 

    /* Soft-threshold update */
    double v = fabs(u) - l1; 
    if (v > 0.0) {       
        *beta_k = copysign(v, u) / (1.0 + l2);
    } else {                          // l_1 penalty wins, set B_k to 0
        *beta_k = 0.0;
    }

    /* Return change in Beta */
    return *beta_k - beta_old;
}



/* Cyclical coordinate descent for elastic-net regression
 *
 * Minimises the elastic net objective function for standardized data
 *
 * Inputs:
 *   Z         - n x p column-major predictor matrix
 *   r         - residual vector  (r = z for first iteration)
 *   beta      - length-p model coefficient vector (for each other station)
 *   n, p      - dimensions
 *   lambda    - regularisation strength
 *   alpha     - elastic-net mixing parameter
 *   threshold - convergence threshold
 *   maxit     - cap on outer loop iterations (glmnet default = 100,000)
 *   is_active  - length-p, active-set membership flags
 *   active_idx - length-p, list of active coordinate indices
 *
 * On return, r holds the final residual and beta holds the fitted coefficients
 * Returns 0 if converged, 1 if maxit was hit
 */
int coord_descent(const double *Z, double *r, double *beta, int n, int p, double lambda, double alpha,
                  double threshold, int maxit, int *is_active, int *active_idx)
{
    /* Precompute l1 and l2 */
    const double l1 = lambda * alpha;
    const double l2 = lambda * (1.0 - alpha);

    /* Active set starts empty; the first full sweep fills it */
    int n_active = 0;
    for (int k = 0; k < p; ++k){
        is_active[k] = 0;
    }

    int passes = 0;     // Total sweeps so far

    /* Outer loop, full sweeps until it changes nothing */
    double dlx = threshold;
    while (dlx >= threshold && passes < maxit) {

        dlx = 0.0;      // Convergence tracker for full sweep

        /* For each predictor k (the other stations) */
        for (int k = 0; k < p; ++k) {
            const double *col = Z + k * n;  // Pointer to the k-th column of Z

            /* Compute gradient at coordinate k */
            double g_k = 0.0;
            #pragma omp simd reduction(+:g_k)
            for (int i = 0; i < n; ++i){
                g_k += col[i] * r[i];
            }

            /* Soft-threshold update, returns delta = new_beta_k - old_beta_k */
            double delta = update_beta(&beta[k], g_k, l1, l2);

            /* If L1 penalty does not set this Beta to 0 */
            if (delta != 0.0) {
                /* Update residual */
                #pragma omp simd
                for (int i = 0; i < n; ++i){
                    r[i] -= delta * col[i];
                }
                /* Accumulate the maximum delta for this sweep */
                if (delta * delta > dlx){
                    dlx = delta * delta;
                }
            }

            /* If this Beta is nonzero and not yet listed, add it to the active set */
            if (beta[k] != 0.0 && !is_active[k]) {
                is_active[k] = 1;
                active_idx[n_active++] = k;
            }
        }

        ++passes;   // Count this full sweep

        
        double dlx_a = threshold;
        
        /* Inner loop, sweep only the active set until it converges */
        while (dlx_a >= threshold && passes < maxit) {

            dlx_a = 0.0;

            /* For each active predictor */
            for (int a = 0; a < n_active; ++a) {

                const int k = active_idx[a];    // Index of the a-th active station
                const double *col = Z + k * n;  // Pointer to the k-th column of Z

                /* Compute gradient at coordinate k */
                double g_k = 0.0;
                #pragma omp simd reduction(+:g_k)
                for (int i = 0; i < n; ++i){
                    g_k += col[i] * r[i];
                }

                /* Soft-threshold update, returns delta = new_beta_k - old_beta_k */
                double delta = update_beta(&beta[k], g_k, l1, l2);

                /* If L1 penalty does not set this Beta to 0 */
                if (delta != 0.0) {
                    /* Update residual */
                    #pragma omp simd
                    for (int i = 0; i < n; ++i){
                        r[i] -= delta * col[i];
                    }
                    /* Accumulate the maximum delta for this sweep */
                    if (delta * delta > dlx_a){
                        dlx_a = delta * delta;
                    }
                }
            }

            ++passes;   // Count this active sweep
        }
    }

    /* Converged if we left because dlx dropped below threshold, not because of maxit */
    return (passes >= maxit) ? 1 : 0;
}



/*
 * Standardise matrix Z and target vector z for elastic net model
 *
 * Outputs (allocated outside function):
 *   Zm    - original mean of column j of Z
 *   Zs    - original std dev of column j of Z
 *   *zm   - original mean of z
 *   *zs   - original std dev of z
 *
 * After call:
 *   Non-constant columns of Z have sum 0 and squared L2 norm 1
 *   Constant columns of Z are zeroed out
 *   z has sum 0 and squared L2 norm 1
 * 
 */
void standardise(double *Z, double *z, int n, int p, double *Zm, double *Zs, double *zm, double *zs)
{
    const double inv_sqrt_n = 1.0 / sqrt((double)n);

    /* Looping through each column */
    for (int j = 0; j < p; ++j) {

        double *col = Z + j * n; // pointer to the j-th column

        /* Check for constant columns: early exit on first differing entry */
        int is_constant = 1;
        double first = col[0];
        for (int i = 1; i < n; ++i) {
            if (col[i] != first) {
                is_constant = 0;
                break; 
            }
        }
        if (is_constant) {
            Zm[j] = first;
            Zs[j] = 1.0;                          // for back-transform 
            memset(col, 0, n * sizeof(double));   // zero the column
            continue;
        }


        /* Column mean */
        double sum = 0.0;
        for (int i = 0; i < n; ++i){
            sum += col[i];
        }
        Zm[j] = sum / n;

        /* Subtract mean, accumulate sum of squared deviations */
        double ssq = 0.0;
        for (int i = 0; i < n; ++i) {
            col[i] -= Zm[j];
            ssq += col[i] * col[i];
        }

        /* Population std dev */
        Zs[j] = sqrt(ssq / n);

        /* Scale so column has squared L2 norm 1 */
        const double scale = Zs[j] / inv_sqrt_n;
        for (int i = 0; i < n; ++i) col[i] /= scale;
    }

    /* Target z mean */
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += z[i];
    *zm = sum / n;

    /* Subtract mean, accumulate sum of squared deviations */
    double ssq = 0.0;
    for (int i = 0; i < n; ++i) {
        z[i] -= *zm;
        ssq += z[i] * z[i];
    }

    /* Population std dev */
    *zs = sqrt(ssq / n);

    /* Scale so z vector has squared L2 norm 1 */
    const double z_scale = (*zs) / inv_sqrt_n;
    for (int i = 0; i < n; ++i) z[i] /= z_scale;
}



/*
 * Converts standardised coefficients back to original-scale coefficients and returns the intercept.
 *
 * Inputs:
 *   beta - fitted coeffficients on the standardised scale
 *   Zm   - column means of original Z   (from standardise)
 *   Zs   - column std devs of original Z (from standardise)
 *   zm   - mean of original z
 *   zs   - std dev of original z
 *   p    - number of stations
 *
 * After Call:
 *   beta transformed to original scale coefficients
 *   Returns regression intercept
 */
double back_transform(double *beta, const double *Zm, const double *Zs, double zm, double zs, int p)
{
    double a = zm;

    for (int j = 0; j < p; ++j) {
        beta[j] *= zs / Zs[j];      // back-transform to original scale
        a -= Zm[j] * beta[j];       // computing intercept
    }
    return a;
}