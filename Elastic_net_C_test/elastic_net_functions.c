#include <math.h>
#include <string.h>

#include <math.h>

/*
 * Standardise matrix Z and target vector z for elastic net model
 *
 *   Z is the n-by-p matrix in column-major layout (Z[i + j*n] = entry at row i, column j)
 *   From the caller's perspective, this is Z_{-j} when imputing station j
 * 
 *   z is the length-n target vector (the station being imputed, Z_j).
 *
 * Outputs (allocated outside function):
 *   Zm[j] - original mean of column j of Z
 *   Zs[j] - original std dev of column j of Z
 *   *zm   - original mean of z
 *   *zs   - original std dev of z
 *
 * After call:
 *   Non-constant columns of Z have sum 0 and squared L2 norm 1
 *   Constant columns of Z are zeroed out
 *   z has sum 0 and squared L2 norm 1
 *
 * Variance uses 1/n divisor (matching glmnet).
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