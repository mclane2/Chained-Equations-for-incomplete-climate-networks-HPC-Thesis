/**
 * Filename: ENCE_impute.c
 * 
 * Description: Implements the ENCE_impute_parallel and column_impute R functions in C to allow for OpenMP parallelism
 * 
 * 
 * Author: M. Lane
 * Version: 3.0 (v.3 Editted for MICE DURR)
 * Date: 2026-06-19 
 */

#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <math.h>

#include "elastic_net.h"
#include "ENCE_impute.h"


/* Mean squared error of the elastic-net prediction against y.
 *
 *   X         - nrow x p predictor matrix
 *   beta      - coefficients vector
 *   intercept - model intercept
 *   y         - length-nrow observed targets
 *
 * Returns MSE
 */
static double predict_mse(const double *X, int nrow, int p, const double *beta, double intercept, const double *y)
{
    double sse = 0.0;
    for (int i = 0; i < nrow; ++i) {
        double yhat = intercept;
        for (int j = 0; j < p; ++j) {
            yhat += X[(size_t)j * nrow + i] * beta[j];
        }
        double e = y[i] - yhat;
        sse += e * e;
    }
    return sse / nrow;
}


/* Impute every missing value of target station (Re-implementation of column_impute() in R code)
 *
 *   df_old        - n x p data matrix
 *   target_out    - length-n output column for this station
 *   col           - index of the target station
 *   missing_col   - length-n int, nonzero = originally missing in this station
 *   fold_col      - length-n int, CV fold id for observed rows
 *   lambdas/alphas, n_lambda/n_alpha, nfolds - CV grid and fold count
 *   thresh, maxit - coordinate-descent controls
 *   lambda_io/alpha_io - chosen (lambda, alpha); if *lambda_io < 0, run CV
 * 
 *  CHANGE THESE STUPID COMMENTS
 *   noise_col     - length-n; first n_missing entries are N(0,1) draws (one per originally-missing
 *                   cell), added scaled by residual SD to the regression predictions so each
 *                   imputation carries DURR between-imputation variability. Trailing entries 0.
 *   beta_out      - output buffer for this column's fitted elastic-net coefficients; written for
 *                   diagnostic dumping / comparison against R's betas.
 *   boot_col      - length-n int; first n_obs entries are the 1-based observed-row indices
 *                   resampled with replacement (DURR bootstrap). Solver fits on these rows rather
 *                   than the raw observed rows. Trailing entries 0.
 *
 */                       
static int column_impute(const double *df_old, double *target_out, int n, int p, int col, const int *missing_col, const int *fold_col,
                          const double *lambdas, int n_lambda, const double *alphas,  int n_alpha, int nfolds, double thresh, int maxit,
                          double *lambda_io, double *alpha_io, const double *noise_col, double *beta_out, const int *boot_col)
{
    const int p_cov = p - 1;   // number of other stations

    /* Copy the target column into target_out from df_old */
    memcpy(target_out, df_old + (size_t)col * n, (size_t)n * sizeof(double));

    /* Count observed/missing rows for this station */
    int n_obs = 0;
    int n_miss = 0;
    for (int i = 0; i < n; ++i) {
        if (missing_col[i]) ++n_miss; 
        else ++n_obs;
    }

    /* If no missing entries */
    if (n_miss == 0) {
    for (int k = 0; k < p; ++k) beta_out[k] = 0.0;
    return 0;
}

    int *miss_rows = malloc((size_t)n_miss * sizeof(int));            // missing indices of this station
    double *x_obs  = malloc((size_t)n_obs  * p_cov * sizeof(double)); // observed entries of other stations
    double *x_miss = malloc((size_t)n_miss * p_cov * sizeof(double)); // missing entries of other stations
    double *y_obs  = malloc((size_t)n_obs * sizeof(double));          // observed values of this station
    double *beta   = malloc((size_t)p_cov * sizeof(double));          // holds fitted coefficients

    /* Malloc error check*/
    if (!miss_rows || !x_obs || !x_miss || !y_obs || !beta) {
        free(miss_rows); free(x_obs); free(x_miss); free(y_obs); free(beta);
        return 1;
    }

    /* Split rows into observed / missing */
    int im = 0;
    for (int i = 0; i < n; ++i) {
        if (missing_col[i]) miss_rows[im++] = i;
    }


    /* Build covariate matrices */
    for (int jc = 0; jc < p_cov; ++jc) {
        int src = (jc < col) ? jc : jc + 1; // for columns before target station jc, for columns after target station jc+1
        const double *src_col = df_old + (size_t)src * n;
        double *xo = x_obs  + (size_t)jc * n_obs;
        double *xm = x_miss + (size_t)jc * n_miss;
    
        for (int io = 0; io < n_obs;  ++io) xo[io] = src_col[boot_col[io] - 1];
        for (int im = 0; im < n_miss; ++im) xm[im] = src_col[miss_rows[im]];
    }

    /* Build target column */
    const double *tcol = df_old + (size_t)col * n;
    for (int io = 0; io < n_obs; ++io) y_obs[io] = tcol[boot_col[io] - 1];


    double lambda = *lambda_io;
    double alpha  = *alpha_io;

    /* Cross-validation only if the hyperparameters were not already chosen */
    if (lambda < 0.0) {

        /* Accumulated mean CV error, column-major (a + l*n_alpha) to match R's which.min order */
        double *cvm   = calloc((size_t)n_alpha * n_lambda, sizeof(double));
        int *fold_obs = malloc((size_t)n_obs * sizeof(int));
        int *train_id = malloc((size_t)n_obs * sizeof(int));
        int *val_id   = malloc((size_t)n_obs * sizeof(int));
        double *xt    = malloc((size_t)n_obs * p_cov * sizeof(double));
        double *xv    = malloc((size_t)n_obs * p_cov * sizeof(double));
        double *yt    = malloc((size_t)n_obs * sizeof(double));
        double *yv    = malloc((size_t)n_obs * sizeof(double));
        double *beta_path = malloc((size_t)p_cov * n_lambda * sizeof(double));
        double *int_path  = malloc((size_t)n_lambda * sizeof(double));

        /* Malloc error check */
        if (!cvm || !fold_obs || !train_id || !val_id || !xt || !xv || !yt || !yv || !beta_path || !int_path ) {
            free(cvm); free(fold_obs); free(train_id); free(val_id);
            free(xt); free(xv); free(yt); free(yv);
            free(miss_rows); free(x_obs); free(x_miss); free(y_obs); free(beta);
            free(beta_path); free(int_path);
            return 1;
        }

        /* Fold id for each observed row */
        for (int io = 0; io < n_obs; ++io) fold_obs[io] = fold_col[io];

        /* Loop through folds */
        for (int f = 1; f <= nfolds; ++f) {

            /* Partition into train and test */
            int n_train = 0;
            int n_val = 0;
            for (int io = 0; io < n_obs; ++io) {     // loop over observed entries

                if (fold_obs[io] == f) val_id[n_val++] = io;  // validation set
                else train_id[n_train++] = io;                // training set
            }

            if (n_val == 0 || n_train == 0) continue;   /* Skip to next fold incase theres nothing to test on*/

            /* Gather train / validation designs */
            for (int jc = 0; jc < p_cov; ++jc) {
                const double *src = x_obs + (size_t)jc * n_obs;
                double *dt = xt + (size_t)jc * n_train;
                double *dv = xv + (size_t)jc * n_val;
                for (int t = 0; t < n_train; ++t) dt[t] = src[train_id[t]];
                for (int v = 0; v < n_val; ++v) dv[v] = src[val_id[v]];
            }
            for (int t = 0; t < n_train; ++t) yt[t] = y_obs[train_id[t]];
            for (int v = 0; v < n_val;   ++v) yv[v] = y_obs[val_id[v]];

            for (int a = 0; a < n_alpha; ++a) {
                elastic_net_path(xt, yt, n_train, p_cov, lambdas, n_lambda, alphas[a], thresh, maxit, beta_path, int_path);

                for (int l = 0; l < n_lambda; ++l) {
                    const double *b = beta_path + (size_t)l * p_cov;
                    cvm[a + (size_t)l * n_alpha] += predict_mse(xv, n_val, p_cov, b, int_path[l], yv) / nfolds;
                }
            }
        }

        /* Pick the first (alpha, lambda) with lowest mean CV error */
        int best_a = 0;
        int best_l = 0;
        double best = cvm[0];
        for (int l = 0; l < n_lambda; ++l) {
            for (int a = 0; a < n_alpha; ++a) {
                double v = cvm[a + (size_t)l * n_alpha];
                if (v < best) { best = v; best_a = a; best_l = l; }
            }
        }
        alpha  = alphas[best_a];
        lambda = lambdas[best_l];

        free(cvm); free(fold_obs); free(train_id); free(val_id);
        free(xt); free(xv); free(yt); free(yv); free(beta_path); free(int_path);
    }

    /* Final fit on all observed data */
    double intercept;
    elastic_net_path(x_obs, y_obs, n_obs, p_cov, &lambda, 1, alpha, thresh, maxit, beta, &intercept);

    /* We need the regression error for sampling */
    double s2hat = predict_mse(x_obs, n_obs, p_cov, beta, intercept, y_obs);

    /* Predict at the missing rows */
    for (int im = 0; im < n_miss; ++im) { 
        double yhat = intercept;
        for (int j = 0; j < p_cov; ++j) {
            yhat += x_miss[(size_t)j * n_miss + im] * beta[j];
        }
        /* Sample imputed values using DURR */
        target_out[miss_rows[im]] = yhat  + sqrt(s2hat) * noise_col[im];
    }

    /* Record the hyperparameters used */
    *lambda_io = lambda;
    *alpha_io  = alpha;

    /* Record the final beta parameters */
    beta_out[0] = intercept;
    for (int jc = 0; jc < p_cov; ++jc) beta_out[1 + jc] = beta[jc];

    free(miss_rows); free(x_obs); free(x_miss); free(y_obs); free(beta);

    return 0;
}


/* One synchronous imputation cycle over all stations.
 *
 * Each iteration reads only df_old and writes a disjoint column of df_new
 */
int ence_impute_cycle(const double *df_old, double *df_new, const int *missing_idx, const int *folds, int n, int p, const double *lambdas, int n_lambda,
                       const double *alphas, int n_alpha, int nfolds, double thresh, int maxit, int nthreads, double *lambda_io, double *alpha_io,
                       const double *noise, double *betas_out, const int *boot)
{
    int failed = 0;

    if (nthreads <= 0) nthreads = omp_get_max_threads();   /* Use all cores if ntheads is not passed */


    #pragma omp parallel for schedule(dynamic) num_threads(nthreads)
    for (int col = 0; col < p; ++col) {

        if (column_impute(df_old, df_new + (size_t)col * n, n, p, col, missing_idx + (size_t)col * n, folds + (size_t)col * n,
                      lambdas, n_lambda, alphas, n_alpha, nfolds, thresh, maxit, &lambda_io[col], &alpha_io[col], noise + (size_t)col*n, betas_out + (size_t)col*p, boot + (size_t)col*n) !=0){
            
            /* Report if there's a Malloc error or if function failed */
            #pragma omp atomic write
            failed = 1;
        }
    }

    return failed;
}