/**
 * ridge_gsl.h - Ridge Regression Implementation Header using GSL
 * Optimized C interface for ridge regression with significance testing
 * using GSL and optionally OpenMP/OpenBLAS multi-threading support.
 */
#ifndef RIDGE_GSL_H
#define RIDGE_GSL_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Set the number of threads for OpenBLAS operations if available.
 * (Note: OpenMP threads might be controlled via OMP_NUM_THREADS env var).
 *
 * @param num_threads Number of threads to use (0 = auto/default, >0 sets threads)
 * @return Previous number of threads set (or 1 if unavailable).
 */
int ridge_gsl_set_blas_threads(int num_threads);

/**
 * Get the current number of threads used by OpenBLAS if available.
 * (Note: This might not reflect OpenMP threads used elsewhere).
 *
 * @return Current number of BLAS threads (or 1 if unavailable).
 */
int ridge_gsl_get_blas_threads();

/**
 * Performs ridge regression with permutation testing or t-test using GSL.
 * Optimized for direct memory access via pointers from Python/Cython.
 * Uses T column permutation instead of Y row permutation for the permutation test.
 *
 * @param X_vec Pointer to the input data matrix (n_genes x n_features),
 *              expected to be in row-major order and C-contiguous.
 * @param Y_vec Pointer to the output data matrix (n_genes x n_samples),
 *              expected to be in row-major order and C-contiguous.
 * @param n_pt Pointer to the number of genes/observations (n).
 * @param p_pt Pointer to the number of features (p).
 * @param m_pt Pointer to the number of samples (m).
 * @param lambda_pt Pointer to the ridge regularization parameter (lambda >= 0).
 * @param nrand_pt Pointer to the number of permutations. If the value pointed to
 *                 is <= 0, a t-test is performed instead.
 * @param beta_vec Pointer to the output buffer for beta coefficients (p x m),
 *                 expected row-major. Will be filled by this function.
 * @param se_vec Pointer to the output buffer for standard errors (p x m),
 *               expected row-major. Will be filled by this function.
 * @param zscore_vec Pointer to the output buffer for z-scores (permutation test)
 *                   or t-statistics (t-test) (p x m), expected row-major.
 *                   Will be filled by this function.
 * @param pvalue_vec Pointer to the output buffer for p-values (p x m),
 *                   expected row-major. Will be filled by this function.
 * @param df_out Pointer to a double where the calculated degrees of freedom (df)
 *               for the t-test will be stored. Set to NAN if permutation test is run
 *               or if an error occurs during t-test df calculation.
 * @return 0 on success, non-zero error code on failure (see RIDGE_ERROR_* defines).
 *         On failure, output arrays (beta, se, zscore, pvalue, df_out) are
 *         typically filled with NAN.
 */
int ridge_gsl_reg(
    double *X_vec,         // Input: n x p
    double *Y_vec,         // Input: n x m
    int *n_pt,             // Input: dimension n
    int *p_pt,             // Input: dimension p
    int *m_pt,             // Input: dimension m
    double *lambda_pt,     // Input: regularization lambda
    double *nrand_pt,      // Input: number of permutations (or 0 for t-test)
    double *beta_vec,      // Output: p x m
    double *se_vec,        // Output: p x m
    double *zscore_vec,    // Output: p x m
    double *pvalue_vec,    // Output: p x m
    double *df_out         // Output: degrees of freedom (for t-test)
);

#ifdef __cplusplus
} // extern "C"
#endif

#endif /* RIDGE_GSL_H */