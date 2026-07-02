#ifndef ENCE_IMPUTE_H
#define ENCE_IMPUTE_H
 
int ence_impute_cycle(const double *df_old, double *df_new, const int *missing_idx, const int *folds, int n, int p, const double *lambdas, int n_lambda,
                       const double *alphas,  int n_alpha, int nfolds, double thresh, int maxit, int nthreads, double *lambda_io, double *alpha_io);
 
#endif