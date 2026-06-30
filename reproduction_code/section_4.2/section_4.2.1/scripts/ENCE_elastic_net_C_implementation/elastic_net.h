#ifndef ELASTIC_NET_H
#define ELASTIC_NET_H

int elastic_net_path(const double *Z_in, const double *z_in, int n, int p,
                     const double *lambdas, int n_lambda, double alpha,
                     double thresh, int maxit, double *beta_out, double *intercept_out);
#endif