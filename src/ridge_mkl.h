/**
 * ridge_mkl.h - Intel MKL optimized ridge regression
 * Supports both dense x dense and dense x sparse operations
 */
#ifndef RIDGE_MKL_H
#define RIDGE_MKL_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Sets the number of threads MKL should use for subsequent operations.
 * Wrapper around mkl_set_num_threads.
 *
 * @param num_threads Number of threads (>= 0). 0 often means use MKL default/environment var.
 * @return Previous maximum thread setting reported by MKL, or -1 if MKL not compiled in.
 */
int ridge_mkl_set_threads(int num_threads);

/**
 * Gets the maximum number of threads MKL is currently configured to use.
 * Wrapper around mkl_get_max_threads.
 *
 * @return Current maximum MKL thread setting, or 1 if MKL not compiled in or MKL reports <=0.
 */
int ridge_mkl_get_threads(void);

/**
 * Performs ridge regression with permutation testing or t-test using Intel MKL.
 * Optimized for dense X and Y matrices.
 * 
 * @param X Dense input matrix (n_genes x n_features), row-major (C-style).
 * @param Y Dense output matrix (n_genes x n_samples), row-major (C-style).
 * @param n_genes Number of rows in X and Y (observations).
 * @param n_features Number of columns in X (features).
 * @param n_samples Number of columns in Y (samples/tasks).
 * @param lambda_val Ridge regularization parameter (lambda >= 0).
 * @param n_rand Number of permutations. If <= 0, a t-test is performed instead.
 * @param beta Output buffer for beta coefficients (n_features x n_samples), row-major.
 * @param se Output buffer for standard errors (n_features x n_samples), row-major.
 * @param zscore Output buffer for z-scores/t-statistics (n_features x n_samples), row-major.
 * @param pvalue Output buffer for p-values (n_features x n_samples), row-major.
 * @param df_out Pointer to store degrees of freedom for t-test. Set to NAN for permutation test.
 * @return 0 on success, non-zero error code on failure.
 */
int ridge_mkl_dense(
    const double *X,
    const double *Y,
    int n_genes,
    int n_features,
    int n_samples,
    double lambda_val,
    int n_rand,
    double *beta,
    double *se,
    double *zscore,
    double *pvalue,
    double *df_out
);

/**
 * Performs ridge regression with permutation testing using Intel MKL.
 * Optimized for sparse Y matrices in CSR format with dense X.
 * Uses 0-based indexing for sparse matrix representation.
 *
 * @param X Dense input matrix (n_genes x n_features), row-major (C-style).
 * @param n_genes Number of rows in X and Y (observations).
 * @param n_features Number of columns in X (features).
 * @param Y_vals 'values' array from CSR sparse matrix Y (length nnz).
 * @param Y_col 'col_ind' array from CSR sparse matrix Y (length nnz), 0-based.
 * @param Y_row 'row_ptr' array from CSR sparse matrix Y (length n_genes+1), 0-based.
 * @param n_samples Number of columns in Y (samples/tasks).
 * @param nnz Number of non-zero elements in Y.
 * @param lambda_val Ridge regularization parameter (lambda >= 0).
 * @param n_rand Number of permutations (must be > 0).
 * @param beta Output buffer for beta coefficients (n_features x n_samples), row-major.
 * @param se Output buffer for standard errors (n_features x n_samples), row-major.
 * @param zscore Output buffer for z-scores (n_features x n_samples), row-major.
 * @param pvalue Output buffer for p-values (n_features x n_samples), row-major.
 * @return 0 on success, non-zero error code on failure.
 */
int ridge_mkl_sparse(
    const double *X,
    int n_genes,
    int n_features,
    const double *Y_vals,
    const int    *Y_col,
    const int    *Y_row,
    int n_samples,
    int nnz,
    double lambda_val,
    int n_rand,
    double *beta,
    double *se,
    double *zscore,
    double *pvalue
);

#ifdef __cplusplus
}
#endif

#endif /* RIDGE_MKL_H */