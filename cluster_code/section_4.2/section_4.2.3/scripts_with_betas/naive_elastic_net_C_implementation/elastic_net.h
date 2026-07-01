#ifndef ELASTIC_NET_H
#define ELASTIC_NET_H

int elastic_net_fit(const double *Z_in, const double *z_in, int n, int p, double lambda, double alpha, double thresh, int maxit, double *beta_out, double *intercept_out);

#endif