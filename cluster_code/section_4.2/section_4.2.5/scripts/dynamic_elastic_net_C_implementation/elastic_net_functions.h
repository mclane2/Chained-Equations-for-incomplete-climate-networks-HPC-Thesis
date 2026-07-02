#ifndef ELASTIC_NET_FUNCTIONS_H
#define ELASTIC_NET_FUNCTIONS_H

void standardise(double *Z, double *z, int n, int p, double *Zm, double *Zs, double *zm, double *zs);

int coord_descent(const double *Z, double *r, double *beta, int n, int p, double lambda, double alpha, double thresh, int maxit, int *is_active, int *active_idx);

double back_transform(double *beta, const double *Zm, const double *Zs, double zm, double zs, int p);

#endif