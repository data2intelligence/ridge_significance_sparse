/*
 * ridge_mkl.c - Optimized C implementation for ridge regression using Intel MKL
 * Supports both dense x dense and dense x sparse operations
 * Provides permutation testing and t-test capabilities
 */

#include <stdlib.h>         // For malloc, calloc, free, exit, size_t
#include <string.h>         // For memcpy, strcmp
#include <stdio.h>          // For fprintf, stderr
#include <math.h>           // For fabs, sqrt, INFINITY, NAN
#include <time.h>           // For time() seeding
#include <unistd.h>         // For getpid()

// OpenMP header for parallelization directives
#ifdef _OPENMP
#include <omp.h>            // Include OpenMP header for pragmas
#endif

// --- MKL Headers ---
// Include base MKL header first - defines types like MKL_INT
#include "mkl.h"
// Explicitly include MKL VSL headers for random number generation
#include "mkl_vsl.h"
#include "mkl_vsl_functions.h" // Contains prototypes like viRngUniform
// Include MKL Sparse BLAS header
#include "mkl_spblas.h"
// LAPACKE header is usually included via mkl.h, but explicit include can help clarity
#include "mkl_lapacke.h"
// CBLAS header is usually included via mkl.h
#include "mkl_cblas.h"

// Include own header last for function prototypes
#include "ridge_mkl.h"

// --- Constants ---
#define EPS 1e-12 // Epsilon for floating-point comparisons
#define MIN_PARALLEL_SIZE 5000 // Threshold for enabling OpenMP parallel for loops

// --- Helper Macros for Error Checking ---

// Check malloc/calloc results, set error code and goto cleanup on failure
// Use different labels for different functions
#define CHECK_ALLOC_DENSE(ptr) do { \
    if ((ptr) == NULL) { \
        fprintf(stderr, "ERROR (dense): Memory allocation failed at %s:%d\n", __FILE__, __LINE__); \
        exit_status = 1; /* MKL memory allocation error */ \
        goto cleanup_dense; \
    } \
} while (0)

#define CHECK_ALLOC_SPARSE(ptr) do { \
    if ((ptr) == NULL) { \
        fprintf(stderr, "ERROR (sparse): Memory allocation failed at %s:%d\n", __FILE__, __LINE__); \
        exit_status = 1; /* MKL memory allocation error */ \
        goto cleanup_sparse; \
    } \
} while (0)

#define CHECK_ALLOC_TTEST(ptr) do { \
    if ((ptr) == NULL) { \
        fprintf(stderr, "ERROR (ttest): Memory allocation failed at %s:%d\n", __FILE__, __LINE__); \
        exit_status = 1; /* MKL memory allocation error */ \
        goto ttest_cleanup; \
    } \
} while (0)

// Check MKL Sparse BLAS status, set error code and goto cleanup on failure
#define CHECK_SPARSE_STATUS(status, msg) do { \
    if ((status) != SPARSE_STATUS_SUCCESS) { \
        fprintf(stderr, "ERROR (sparse): %s failed with MKL Sparse status = %d at %s:%d\n", \
                (msg), (int)(status), __FILE__, __LINE__); \
        exit_status = 5; /* MKL Sparse matrix operation error */ \
        goto cleanup_sparse; \
    } \
} while (0)

// Check LAPACKE integer return code, set error code and goto appropriate cleanup
#define CHECK_LAPACK_STATUS_DENSE(info, func_name) do { \
    if ((info) != 0) { \
        fprintf(stderr, "ERROR (dense): %s failed with LAPACK info = %d at %s:%d\n", \
                (func_name), (int)(info), __FILE__, __LINE__); \
        if (strcmp((func_name), "LAPACKE_dpotrf") == 0) \
            exit_status = 3; /* Factorization error */ \
        else if (strcmp((func_name), "LAPACKE_dpotrs") == 0) \
            exit_status = 4; /* Solve error */ \
        else \
            exit_status = 2; /* Generic MKL matrix operation error */\
        goto cleanup_dense; \
    } \
} while (0)

#define CHECK_LAPACK_STATUS_SPARSE(info, func_name) do { \
    if ((info) != 0) { \
        fprintf(stderr, "ERROR (sparse): %s failed with LAPACK info = %d at %s:%d\n", \
                (func_name), (int)(info), __FILE__, __LINE__); \
        if (strcmp((func_name), "LAPACKE_dpotrf") == 0) \
            exit_status = 3; /* Factorization error */ \
        else if (strcmp((func_name), "LAPACKE_dpotrs") == 0) \
            exit_status = 4; /* Solve error */ \
        else \
            exit_status = 2; /* Generic MKL matrix operation error */\
        goto cleanup_sparse; \
    } \
} while (0)

#define CHECK_LAPACK_STATUS_TTEST(info, func_name) do { \
    if ((info) != 0) { \
        fprintf(stderr, "ERROR (ttest): %s failed with LAPACK info = %d at %s:%d\n", \
                (func_name), (int)(info), __FILE__, __LINE__); \
        exit_status = (strcmp((func_name), "LAPACKE_dpotrf") == 0) ? 3 : \
                     (strcmp((func_name), "LAPACKE_dpotrs") == 0) ? 4 : 2; \
        goto ttest_cleanup; \
    } \
} while (0)

// Check MKL VSL status, set error code and goto appropriate cleanup
#define CHECK_VSL_STATUS_DENSE(status, msg) do { \
    if ((status) != VSL_STATUS_OK) { \
        fprintf(stderr, "ERROR (dense): %s failed with VSL status = %d at %s:%d\n", \
                (msg), (int)(status), __FILE__, __LINE__); \
        exit_status = 6; /* MKL random number generation error */ \
        goto cleanup_dense; \
    } \
} while (0)

#define CHECK_VSL_STATUS_SPARSE(status, msg) do { \
    if ((status) != VSL_STATUS_OK) { \
        fprintf(stderr, "ERROR (sparse): %s failed with VSL status = %d at %s:%d\n", \
                (msg), (int)(status), __FILE__, __LINE__); \
        exit_status = 6; /* MKL random number generation error */ \
        goto cleanup_sparse; \
    } \
} while (0)

// --- Thread Control Functions ---

/**
 * Sets the number of threads MKL should use for subsequent operations.
 * Wrapper around mkl_set_num_threads.
 */
int ridge_mkl_set_threads(int num_threads) {
    int prev_max_threads = -1; // Default to error/unavailable
    #ifdef HAVE_MKL // Macro defined during compilation via setup.py
        prev_max_threads = mkl_get_max_threads();
        if (num_threads >= 0) {
            mkl_set_num_threads(num_threads);
        } else {
            // Negative input ignored, return current value
        }
    #else
         // This code block won't be compiled if HAVE_MKL is not defined
         fprintf(stderr, "Warning: MKL support not compiled in, cannot set MKL threads.\n");
    #endif
    return prev_max_threads;
}

/**
 * Gets the maximum number of threads MKL is currently configured to use.
 * Wrapper around mkl_get_max_threads.
 */
int ridge_mkl_get_threads(void) {  // Added void parameter for C standard compliance
    int current_max_threads = 1; // Default to 1 if MKL is not available
    #ifdef HAVE_MKL
        current_max_threads = mkl_get_max_threads();
        // Ensure a positive value is returned
        if (current_max_threads <= 0) current_max_threads = 1;
    #else
        // fprintf(stderr, "Warning: MKL support not compiled in, cannot get MKL threads.\n");
    #endif
    return current_max_threads;
}

// --- Forward declarations for helper functions ---
static void permute_T_columns(double *T_permuted, const double *T, const int *indices,
                              int p, int n);

static int calculate_ttest_direct(const double *X, const double *Y, const double *T,
                                  const double *beta, double *se, double *zscore,
                                  double *pvalue, int n, int p, int m,
                                  double lambda, double *df_out);

static double cdf_tdist(double t, double df);

/**
 * Performs ridge regression with permutation testing or t-test using Intel MKL.
 * Optimized for dense X and Y matrices.
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
) {
    // --- Input Parameter Validation ---
    if (n_genes <= 0 || n_features <= 0 || n_samples <= 0) {
         fprintf(stderr, "ERROR (dense): Input dimensions (n_genes=%d, n_features=%d, n_samples=%d) must be positive.\n",
                 n_genes, n_features, n_samples);
         return 8; // Invalid input parameter
    }
    if (lambda_val < 0.0) {
         fprintf(stderr, "ERROR (dense): lambda_val (%f) must be non-negative.\n", lambda_val);
         return 8; // Invalid input parameter
    }
    if (X == NULL || Y == NULL || beta == NULL || se == NULL || zscore == NULL || pvalue == NULL || df_out == NULL) {
         fprintf(stderr, "ERROR (dense): Received NULL pointer for required input/output array.\n");
        return 8; // Invalid input parameter
    }

    // --- Variable Declarations ---
    double *XtX = NULL;              // [p x p], symmetric matrix X^T*X + lambda*I
    double *T = NULL;                // [p x n], matrix T = inv(X^T*X + lambda*I) * X^T
    double *X_transpose = NULL;      // [p x n], temporary for X^T
    size_t total_elements = (size_t)n_features * n_samples; // Total elements in output arrays
    int exit_status = 0;             // Final status to return (0 = success)
    *df_out = NAN;                   // Initialize df output

    // Permutation related arrays
    double *T_permuted = NULL;       // [p x n], permuted T matrix
    double *beta_perm = NULL;        // [p x m], permuted beta
    double *sum_b = NULL;            // [p x m], sum of permuted betas
    double *sum_b2 = NULL;           // [p x m], sum of squares of permuted betas
    double *count_ge = NULL;         // [p x m], count |beta_perm| >= |beta_obs|
    double *abs_beta = NULL;         // [p x m], absolute value of observed beta
    int *indices = NULL;             // [n], array for permutation indices
    VSLStreamStatePtr stream = NULL; // VSL Random number stream state pointer

    // LAPACK variables
    int info;                        // LAPACK return code

    // --- Memory Allocation ---
    XtX = (double*)mkl_malloc((size_t)n_features * n_features * sizeof(double), 64); CHECK_ALLOC_DENSE(XtX);
    T = (double*)mkl_malloc((size_t)n_features * n_genes * sizeof(double), 64);       CHECK_ALLOC_DENSE(T);
    X_transpose = (double*)mkl_malloc((size_t)n_features * n_genes * sizeof(double), 64); CHECK_ALLOC_DENSE(X_transpose);

    // --- Create X^T and then compute XtX ---

    // Transpose X (n x p) to row-major X^T (p x n)
    #pragma omp parallel for schedule(static) collapse(2)
    for (int i = 0; i < n_genes; i++) {
        for (int j = 0; j < n_features; j++) {
            X_transpose[j * n_genes + i] = X[i * n_features + j];
        }
    }

    // Compute XtX = X^T * X (symmetric matrix)
    cblas_dsyrk(CblasRowMajor, CblasUpper, CblasNoTrans, // Use X^T (p x n)
                n_features, n_genes, // N=p, K=n
                1.0, X_transpose, n_genes, // A=X^T (p x n), lda=n
                0.0, XtX, n_features); // C=XtX (p x p), ldc=p

    // Fill in lower triangle and add lambda to diagonal
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_features; i++) {
        for (int j = 0; j < i; j++) {
            XtX[i * n_features + j] = XtX[j * n_features + i];
        }
        XtX[i * n_features + i] += lambda_val;
    }

    // --- Solve the system: (X^T*X + lambda*I) * T = X^T ---
    // Copy X^T into T as the right-hand side for dpotrs
    memcpy(T, X_transpose, (size_t)n_features * n_genes * sizeof(double));

    // 1) Perform Cholesky factorization of XtX+lambda*I
    info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'U', n_features, XtX, n_features);
    CHECK_LAPACK_STATUS_DENSE(info, "LAPACKE_dpotrf");

    // 2) Solve the system for T using the factorization
    // A = Factorized (XtX+lambda*I) (p x p), B = Input X^T / Output T (p x n)
    // n = p, nrhs = n, lda = p, ldb = n
    info = LAPACKE_dpotrs(LAPACK_ROW_MAJOR, 'U', n_features, n_genes,
                          XtX, n_features, // Factorized A, lda=p
                          T, n_genes);     // Input B / Output X (T), ldb=n
    CHECK_LAPACK_STATUS_DENSE(info, "LAPACKE_dpotrs (solve for T)");

    // --- Compute beta = T * Y ---
    // T is (p x n), Y is (n x m), beta is (p x m)
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n_features, n_samples, n_genes, // M=p, N=m, K=n
                1.0, T, n_genes,           // A=T (p x n), lda=n
                Y, n_samples,          // B=Y (n x m), ldb=m
                0.0, beta, n_samples);        // C=beta (p x m), ldc=m

    // --- Perform t-test or permutation test ---
    if (n_rand <= 0) {
        // --- T-test ---
        fprintf(stderr, "DEBUG: Entering T-test calculation...\n"); fflush(stderr); // Mark entry
        double calculated_df = NAN;
        // Pass T directly to t-test function
        int ttest_status = calculate_ttest_direct(X, Y, T, beta, se, zscore, pvalue,
                                                 n_genes, n_features, n_samples,
                                                 lambda_val, &calculated_df);
        if (ttest_status != 0) {
            fprintf(stderr, "ERROR (dense): T-test calculation failed with status %d.\n", ttest_status);
            // Don't jump to cleanup here, let ridge_mkl_dense handle final status
            exit_status = 7; // T-test error
            *df_out = NAN;
        } else {
            *df_out = calculated_df;
        }
        fprintf(stderr, "DEBUG: Exiting T-test calculation (status %d).\n", ttest_status); fflush(stderr); // Mark exit
    } else {
        // --- Permutation Test ---
        // fprintf(stderr, "DEBUG: Starting permutation test...\n"); fflush(stderr); // Mark entry

        // Additional allocations for permutation test
        T_permuted = (double*)mkl_malloc((size_t)n_features * n_genes * sizeof(double), 64); CHECK_ALLOC_DENSE(T_permuted);
        beta_perm = (double*)mkl_malloc((size_t)n_features * n_samples * sizeof(double), 64); CHECK_ALLOC_DENSE(beta_perm);
        indices = (int*)mkl_malloc((size_t)n_genes * sizeof(int), 64); CHECK_ALLOC_DENSE(indices);
        abs_beta = (double*)mkl_malloc(total_elements * sizeof(double), 64); CHECK_ALLOC_DENSE(abs_beta);

        // Use mkl_calloc for aligned zero-initialized arrays
        sum_b = (double*)mkl_calloc(total_elements, sizeof(double), 64); CHECK_ALLOC_DENSE(sum_b);
        sum_b2 = (double*)mkl_calloc(total_elements, sizeof(double), 64); CHECK_ALLOC_DENSE(sum_b2);
        count_ge = (double*)mkl_calloc(total_elements, sizeof(double), 64); CHECK_ALLOC_DENSE(count_ge);

        // Calculate absolute values of observed beta
        #pragma omp parallel for schedule(static)
        for (size_t idx = 0; idx < total_elements; idx++) {
            abs_beta[idx] = fabs(beta[idx]);
        }

        // Initialize indices array for permutation
        for (int i = 0; i < n_genes; i++) {
            indices[i] = i;
        }

        // --- Setup VSL for permutation ---
        int vsl_seed = 42; // Fixed seed for reproducibility
        int vsl_status = vslNewStream(&stream, VSL_BRNG_MT19937, vsl_seed);
        CHECK_VSL_STATUS_DENSE(vsl_status, "vslNewStream");

        // --- Permutation Loop ---
        #pragma omp parallel
        {
            // Thread-local random indices
            int random_index;
            int tid = omp_get_thread_num();
            int local_exit_status = 0; // Thread-local error status
            
            // Thread-local arrays for permutation
            int *local_indices = NULL;
            double *local_T_permuted = NULL;
            double *local_beta_perm = NULL;

            // Allocate thread-local arrays
            local_indices = (int*)malloc(n_genes * sizeof(int));
            local_T_permuted = (double*)malloc((size_t)n_features * n_genes * sizeof(double));
            local_beta_perm = (double*)malloc((size_t)n_features * n_samples * sizeof(double));
            
            if (!local_indices || !local_T_permuted || !local_beta_perm) {
                fprintf(stderr, "Thread %d: Failed to allocate thread-local memory\n", tid);
                local_exit_status = 1;
                // Continue with main loop, will be skipped due to error check
            }

            // Only one thread initializes the permutation array for the first permutation
            #pragma omp single
            {
                for (int j = 0; j < n_genes; j++) indices[j] = j;
            }

            #pragma omp for schedule(dynamic, 1)
            for (int r = 0; r < n_rand; r++) {
                // Exit this thread's loop if another thread reported an error or local error
                if (exit_status != 0 || local_exit_status != 0) continue;

                // Each thread reinitializes indices for its assigned permutation
                // Create a thread-local copy of indices to shuffle
                memcpy(local_indices, indices, n_genes * sizeof(int));

                // Fisher-Yates shuffle using VSL random integers
                #pragma omp critical(vsl_rng)
                {
                     if (exit_status == 0) { // Only proceed if no global error yet
                        for (int j = n_genes - 1; j > 0; j--) {
                            vsl_status = viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, 1, &random_index, 0, j + 1);
                            if (vsl_status != VSL_STATUS_OK) {
                                fprintf(stderr, "Thread %d: Error in viRngUniform at r=%d, j=%d\n", tid, r, j);
                                exit_status = 6; // Set global error status
                                local_exit_status = 6; // Set local status
                                break; // Exit inner loop
                            }
                            // Swap local_indices[j] with local_indices[random_index]
                            int tmp = local_indices[j];
                            local_indices[j] = local_indices[random_index];
                            local_indices[random_index] = tmp;
                        }
                    }
                } // end critical section

                // Check for error during shuffle or from another thread
                if (exit_status != 0 || local_exit_status != 0) continue;

                // Apply permutation to columns of T to get T_permuted
                permute_T_columns(local_T_permuted, T, local_indices, n_features, n_genes);

                // Compute permuted beta = T_permuted * Y
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            n_features, n_samples, n_genes,
                            1.0, local_T_permuted, n_genes, // A(p x n), lda=n
                            Y, n_samples,          // B(n x m), ldb=m
                            0.0, local_beta_perm, n_samples); // C(p x m), ldc=m

                // Accumulate statistics
                for (int i = 0; i < n_features; i++) {
                    for (int j = 0; j < n_samples; j++) {
                        size_t idx = (size_t)i * n_samples + j;
                        double v_perm = local_beta_perm[idx];

                        #pragma omp atomic update
                        sum_b[idx] += v_perm;

                        #pragma omp atomic update
                        sum_b2[idx] += v_perm * v_perm;

                        if (fabs(v_perm) >= abs_beta[idx] - EPS) {
                            #pragma omp atomic update
                            count_ge[idx] += 1.0;
                        }
                    }
                }
            } // End of parallel for over permutations
            
            // Free thread-local resources
            free(local_indices);
            free(local_T_permuted);
            free(local_beta_perm);
            
        } // End of parallel region

        // Exit early if errors occurred during parallel permutation processing
        if (exit_status != 0) {
            goto cleanup_dense;
        }

        // --- Finalize statistics: SE, Z-score, P-value ---
        double n_rand_plus_1 = (double)(n_rand + 1);
        double n_rand_d = (double)n_rand;

        #pragma omp parallel for schedule(static)
        for (size_t idx = 0; idx < total_elements; idx++) {
            double mean_perm = sum_b[idx] / n_rand_d;
            double mean_sq_perm = sum_b2[idx] / n_rand_d;
            double var_perm = mean_sq_perm - (mean_perm * mean_perm);

            // Improved numerical stability for variance
            if (var_perm < EPS) {
                if (var_perm < -EPS) {
                    #pragma omp critical(variance_warning)
                    {
                        fprintf(stderr, "Warning (dense): Negative variance calculated (%e) at index %zu. Clamping to 0.\n",
                                var_perm, idx);
                    }
                }
                var_perm = 0.0; // Clamp near-zero variance including small negative values
            }

            se[idx] = sqrt(var_perm);

            // Improved z-score calculation
            if (se[idx] > EPS) {
                zscore[idx] = (beta[idx] - mean_perm) / se[idx];
            } else {
                 if (fabs(beta[idx] - mean_perm) < EPS) {
                     zscore[idx] = 0.0;
                 } else {
                     zscore[idx] = (beta[idx] > mean_perm) ? INFINITY : -INFINITY;
                 }
            }

            // P-value calculation
            double p_raw = (count_ge[idx] + 1.0) / n_rand_plus_1;
            pvalue[idx] = fmin(1.0, fmax(0.0, p_raw)); // Clamp p-value to [0, 1]
        }

        // Not applicable for permutation test
        *df_out = NAN;
        // fprintf(stderr, "DEBUG: Finished permutation test successfully.\n"); fflush(stderr); // Mark successful exit
    }

cleanup_dense: // Unique label for dense function cleanup
    // fprintf(stderr, "DEBUG: Entering dense cleanup block (exit_status=%d).\n", exit_status); fflush(stderr); // Mark entry to cleanup
    // Clean up VSL resources
    if (stream != NULL) {
        vslDeleteStream(&stream);
    }

    // Free all aligned memory using mkl_free
    if (XtX != NULL) { mkl_free(XtX); }
    if (T != NULL) { mkl_free(T); }
    if (X_transpose != NULL) { mkl_free(X_transpose); }
    if (T_permuted != NULL) { mkl_free(T_permuted); }
    if (beta_perm != NULL) { mkl_free(beta_perm); }
    if (indices != NULL) { mkl_free(indices); }
    if (abs_beta != NULL) { mkl_free(abs_beta); }
    if (sum_b != NULL) { mkl_free(sum_b); }
    if (sum_b2 != NULL) { mkl_free(sum_b2); }
    if (count_ge != NULL) { mkl_free(count_ge); }

    // If error occurred, fill outputs with NAN
    if (exit_status != 0) {
        // fprintf(stderr, "DEBUG: Filling dense outputs with NAN due to error.\n"); fflush(stderr);
        if (n_features > 0 && n_samples > 0) {
            size_t size = (size_t)n_features * n_samples;
            // Only NaN the subsequent outputs if error happened after beta calculation
            if (exit_status >= 3 && exit_status != 7) { // Matrix ops failed (not t-test specific error)
                #pragma omp parallel for schedule(static)
                for (size_t i = 0; i < size; i++) {
                    se[i] = NAN;
                    zscore[i] = NAN;
                    pvalue[i] = NAN;
                }
            }
            // If error happened before beta calculation, NaN everything
            if (exit_status <= 2) {
                #pragma omp parallel for schedule(static)
                for (size_t i = 0; i < size; i++) {
                    beta[i] = NAN;
                    se[i] = NAN;
                    zscore[i] = NAN;
                    pvalue[i] = NAN;
                }
            }
            // If t-test specifically failed, beta is likely OK, but stats are not
             if (exit_status == 7) {
                 #pragma omp parallel for schedule(static)
                 for (size_t i = 0; i < size; i++) {
                    se[i] = NAN;
                    zscore[i] = NAN;
                    pvalue[i] = NAN;
                 }
             }
        }
        *df_out = NAN;
    }
    // fprintf(stderr, "DEBUG: Exiting ridge_mkl_dense (final status %d).\n", exit_status); fflush(stderr); // Mark final exit
    return exit_status;
}

/**
 * Performs ridge regression with permutation testing using Intel MKL.
 * Optimized for sparse Y matrices in CSR format with dense X.
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
) {
    // --- Input Parameter Validation ---
    if (n_genes <= 0 || n_features <= 0 || n_samples <= 0) {
         fprintf(stderr, "ERROR (sparse): Input dimensions (n_genes=%d, n_features=%d, n_samples=%d) must be positive.\n",
                 n_genes, n_features, n_samples);
         return 8; // Invalid input parameter
    }
    if (lambda_val < 0.0) {
         fprintf(stderr, "ERROR (sparse): lambda_val (%f) must be non-negative.\n", lambda_val);
         return 8; // Invalid input parameter
    }
    if (n_rand <= 0) {
        fprintf(stderr, "ERROR (sparse): n_rand (%d) must be > 0 for ridge_mkl_sparse (permutation test).\n", n_rand);
        return 8; // Invalid input parameter
    }
    if (X == NULL || Y_vals == NULL || Y_col == NULL || Y_row == NULL ||
        beta == NULL || se == NULL || zscore == NULL || pvalue == NULL) {
         fprintf(stderr, "ERROR (sparse): Received NULL pointer for required input/output array.\n");
        return 8; // Invalid input parameter
    }


    // --- Variable Declarations ---
    double *XtX = NULL;              // [p x p], symmetric matrix X^T*X + lambda*I
    double *Xt = NULL;               // [p x n], Explicit X^T
    double *T = NULL;                // [p x n], Solution matrix T = (XtX+lambda*I)^-1 * Xt
    double *T_transpose = NULL;      // [n x p], Explicit T^T needed for sparse mm
    double *beta_transpose = NULL;   // [m x p], temporary storage for beta^T = Y^T * T^T

    size_t total_elements = (size_t)n_features * n_samples; // Total elements in output arrays

    // Permutation related arrays
    double *sum_b = NULL;            // [p x m], sum of permuted betas
    double *sum_b2 = NULL;           // [p x m], sum of squares of permuted betas
    double *count_ge = NULL;         // [p x m], count |beta_perm| >= |beta_obs|
    double *abs_beta = NULL;         // [p x m], absolute value of observed beta
    int *perm = NULL;                // [n], array for permutation indices
    double *T_transpose_perm = NULL; // [n x p], permuted T^T

    // MKL/VSL specific variables
    sparse_matrix_t Y_handle = NULL; // MKL sparse matrix handle
    struct matrix_descr descr;       // Descriptor for sparse matrix (general type)
    VSLStreamStatePtr stream = NULL; // VSL Random number stream state pointer

    int info;                      // LAPACK return code
    sparse_status_t status;        // MKL sparse status
    int vsl_status;                // VSL status
    int exit_status = 0;           // Final status to return (0 = success)

    // Define constants for BLAS/LAPACK calls
    const double alpha = 1.0;
    const double beta_blas = 0.0;
    const double EPSILON = 1e-15;  // Numerical tolerance value


    // --- Memory Allocation ---
    XtX              = (double*)mkl_malloc((size_t)n_features * n_features * sizeof(double), 64); CHECK_ALLOC_SPARSE(XtX);
    Xt               = (double*)mkl_malloc((size_t)n_features * n_genes * sizeof(double), 64);    CHECK_ALLOC_SPARSE(Xt);
    T                = (double*)mkl_malloc((size_t)n_features * n_genes * sizeof(double), 64);    CHECK_ALLOC_SPARSE(T);
    T_transpose      = (double*)mkl_malloc((size_t)n_genes * n_features * sizeof(double), 64);    CHECK_ALLOC_SPARSE(T_transpose);
    beta_transpose   = (double*)mkl_malloc((size_t)n_samples * n_features * sizeof(double), 64);  CHECK_ALLOC_SPARSE(beta_transpose);
    abs_beta         = (double*)mkl_malloc(total_elements * sizeof(double), 64);                  CHECK_ALLOC_SPARSE(abs_beta);
    perm             = (int*)mkl_malloc((size_t)n_genes * sizeof(int), 64);                       CHECK_ALLOC_SPARSE(perm);
    T_transpose_perm = (double*)mkl_malloc((size_t)n_genes * n_features * sizeof(double), 64);    CHECK_ALLOC_SPARSE(T_transpose_perm);

    sum_b    = (double*)mkl_calloc(total_elements, sizeof(double), 64); CHECK_ALLOC_SPARSE(sum_b);
    sum_b2   = (double*)mkl_calloc(total_elements, sizeof(double), 64); CHECK_ALLOC_SPARSE(sum_b2);
    count_ge = (double*)mkl_calloc(total_elements, sizeof(double), 64); CHECK_ALLOC_SPARSE(count_ge);


    // --- Computations ---

    // 1) Compute XtX = X^T * X + lambda*I
    cblas_dsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                n_features, n_genes,
                alpha, X, n_features, // Input X is n x p, lda=p
                beta_blas, XtX, n_features); // Output XtX is p x p, ldc=p

    // Fill lower triangle and add lambda to diagonal
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_features; i++) {
        for (int j = 0; j < i; j++) {
            XtX[i * n_features + j] = XtX[j * n_features + i];
        }
        XtX[i * n_features + i] += lambda_val;
    }

    // 2) Cholesky factorization: XtX+lambda*I = U^T * U
    info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'U', n_features, XtX, n_features);
    CHECK_LAPACK_STATUS_SPARSE(info, "LAPACKE_dpotrf");

    // 3a) Calculate X^T explicitly (p x n)
    #pragma omp parallel for schedule(static) collapse(2)
    for (int i = 0; i < n_genes; i++) {
        for (int j = 0; j < n_features; j++) {
            Xt[j * n_genes + i] = X[i * n_features + j]; // X is n x p
        }
    }

    // 3b) Copy X^T into the solution buffer T, as dpotrs overwrites the RHS
    memcpy(T, Xt, (size_t)n_features * n_genes * sizeof(double));

    // 4) Solve (U^T*U) * T = X^T for T (p x n) using dpotrs
    // A = Factorized XtX (p x p), B = Input X^T / Output T (p x n)
    // n = p (order of A), nrhs = n (cols in B/T), lda = p, ldb = n
    info = LAPACKE_dpotrs(LAPACK_ROW_MAJOR, 'U', n_features, n_genes,
                          XtX, n_features,  // Factorized XtX (p x p), lda=p
                          T, n_genes);      // Input X^T / Output T (p x n), ldb=n_genes (col length)
    CHECK_LAPACK_STATUS_SPARSE(info, "LAPACKE_dpotrs (solve for T)");

    // 5) Explicitly compute T^T (n x p) from T (p x n) for sparse multiply
    #pragma omp parallel for schedule(static) collapse(2)
    for (int i = 0; i < n_features; i++) {
        for (int j = 0; j < n_genes; j++) {
             T_transpose[j * n_features + i] = T[i * n_genes + j]; // T is p x n
        }
    }

    // 6) Create MKL sparse handle for Y (CSR format, 0-based indexing)
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    status = mkl_sparse_d_create_csr(
        &Y_handle, SPARSE_INDEX_BASE_ZERO, n_genes, n_samples,
        (MKL_INT*)Y_row, (MKL_INT*)(Y_row + 1),
        (MKL_INT*)Y_col, (double*)Y_vals);
    CHECK_SPARSE_STATUS(status, "mkl_sparse_d_create_csr");


    // 7) Compute observed beta^T = Y^T * T^T (Result in beta_transpose buffer)
    // Y^T is (m x n), T^T is (n x p). Result beta^T is (m x p).
    status = mkl_sparse_d_mm(
        SPARSE_OPERATION_TRANSPOSE, // Use Y^T (m x n)
        alpha, Y_handle, descr,
        SPARSE_LAYOUT_ROW_MAJOR,   // Layout of T^T
        T_transpose, n_features, n_features, // T^T (n x p), cols=p, lda=p
        beta_blas, beta_transpose, n_features); // Result beta^T (m x p), cols=p, ldb=p
    CHECK_SPARSE_STATUS(status, "mkl_sparse_d_mm (observed beta)");

    // 8) Transpose beta_transpose [m x p] result into final beta [p x m] output buffer
    #pragma omp parallel for schedule(static) collapse(2)
    for (int i = 0; i < n_samples; i++) {
        for (int k = 0; k < n_features; k++) {
            beta[k * n_samples + i] = beta_transpose[i * n_features + k];
        }
    }

    // Calculate absolute values of observed beta
    #pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < total_elements; idx++) {
        abs_beta[idx] = fabs(beta[idx]);
    }


    // --- VSL Setup for Permutations ---
    int vsl_seed = 42; // Fixed seed for reproducibility
    vsl_status = vslNewStream(&stream, VSL_BRNG_MT19937, vsl_seed);
    CHECK_VSL_STATUS_SPARSE(vsl_status, "vslNewStream");


    // --- Permutation Loop ---
    // Process permutations with dynamic scheduling for better load balance
    #pragma omp parallel
    {
        // Thread-local random indices
        int random_index;
        int tid = omp_get_thread_num();
        int local_exit_status = 0; // Thread-local error status
        
        // Thread-local arrays
        int *local_perm = NULL;
        double *local_T_transpose_perm = NULL;
        double *local_beta_transpose = NULL;
        
        // Allocate thread-local arrays
        local_perm = (int*)malloc(n_genes * sizeof(int));
        local_T_transpose_perm = (double*)malloc((size_t)n_genes * n_features * sizeof(double));
        local_beta_transpose = (double*)malloc((size_t)n_samples * n_features * sizeof(double));
        
        if (!local_perm || !local_T_transpose_perm || !local_beta_transpose) {
            fprintf(stderr, "Thread %d: Failed to allocate thread-local memory for sparse\n", tid);
            local_exit_status = 1;
            // Continue with main loop, will be skipped due to error check
        }

        // Only one thread initializes the permutation array for the first permutation
        #pragma omp single
        {
            for (int j = 0; j < n_genes; j++) perm[j] = j;
        }

        #pragma omp for schedule(dynamic, 1)
        for (int r = 0; r < n_rand; r++) {
             // Exit this thread's loop if another thread reported an error or local error
            if (exit_status != 0 || local_exit_status != 0) continue;

            // Each thread reinitializes perm for its assigned permutation
            memcpy(local_perm, perm, n_genes * sizeof(int));

            // Fisher-Yates shuffle using VSL random integers
            #pragma omp critical(vsl_rng)
            {
                 if (exit_status == 0) { // Only proceed if no global error yet
                    for (int j = n_genes - 1; j > 0; j--) {
                        vsl_status = viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, 1, &random_index, 0, j + 1);
                        if (vsl_status != VSL_STATUS_OK) {
                            fprintf(stderr, "Thread %d: Error in viRngUniform at r=%d, j=%d\n", tid, r, j);
                            exit_status = 6;
                            local_exit_status = 6;
                            break;
                        }
                        // Swap local_perm[j] with local_perm[random_index]
                        int tmp = local_perm[j];
                        local_perm[j] = local_perm[random_index];
                        local_perm[random_index] = tmp;
                    }
                 }
            } // end critical section

            // Check for error during shuffle or from another thread
            if (exit_status != 0 || local_exit_status != 0) continue;

            // Apply permutation to rows of T_transpose (which is T^T) to get T_transpose_perm
            // This creates P*T^T implicitly where P is permutation matrix
            for (int j = 0; j < n_genes; j++) {
                int permuted_row_idx = local_perm[j];
                memcpy(local_T_transpose_perm + (size_t)j * n_features, // local P*T^T (n x p)
                       T_transpose + (size_t)permuted_row_idx * n_features, // T^T (n x p)
                       n_features * sizeof(double));
            }

            // Thread-local temporary beta_transpose calculation
            // Compute beta_perm^T = Y^T * (P*T^T)
            sparse_status_t local_status = mkl_sparse_d_mm(
                SPARSE_OPERATION_TRANSPOSE, alpha, Y_handle, descr, // Y^T (m x n)
                SPARSE_LAYOUT_ROW_MAJOR,                           // Layout of P*T^T
                local_T_transpose_perm, n_features, n_features,    // P*T^T (n x p), cols=p, lda=p
                beta_blas, local_beta_transpose, n_features);       // Result beta_perm^T (m x p), cols=p, ldb=p

            if (local_status != SPARSE_STATUS_SUCCESS) {
                #pragma omp critical(error_report)
                {
                    if (exit_status == 0) { // Report only the first error
                       fprintf(stderr, "Thread %d: Error in mkl_sparse_d_mm at r=%d, status=%d\n", tid, r, local_status);
                       exit_status = 5;
                    }
                }
                continue; // Skip accumulation for this thread iteration
            }

            // Accumulate statistics - use collapse for better parallelism
            for (int i = 0; i < n_samples; i++) {
                for (int k = 0; k < n_features; k++) {
                    double v_perm = local_beta_transpose[i * n_features + k];
                    size_t beta_idx = (size_t)k * n_samples + i;

                    #pragma omp atomic update
                    sum_b[beta_idx] += v_perm;
                    #pragma omp atomic update
                    sum_b2[beta_idx] += v_perm * v_perm;

                    if (fabs(v_perm) >= abs_beta[beta_idx] - EPSILON) {
                        #pragma omp atomic update
                        count_ge[beta_idx] += 1.0;
                    }
                }
            }
        } // End of parallel for over permutations
        
        // Free thread-local resources
        free(local_perm);
        free(local_T_transpose_perm);
        free(local_beta_transpose);
        
    } // End of parallel region

    // Exit early if errors occurred during parallel permutation processing
    if (exit_status != 0) {
        goto cleanup_sparse;
    }

    // --- Finalize statistics: SE, Z-score, P-value ---
    double n_rand_plus_1 = (double)(n_rand + 1);
    double n_rand_d = (double)n_rand;

    #pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < total_elements; idx++) {
        double mean_perm = sum_b[idx] / n_rand_d;
        double var_perm = (sum_b2[idx] / n_rand_d) - (mean_perm * mean_perm);

        // Improved numerical stability for variance using relative epsilon
        double rel_epsilon = fmax(EPSILON, fabs(mean_perm) * EPSILON);

        if (var_perm < rel_epsilon) {
            if (var_perm < -rel_epsilon) {
                #pragma omp critical(variance_warning)
                {
                    fprintf(stderr, "Warning (sparse): Negative variance calculated (%e) at index %zu. Clamping to 0.\n",
                            var_perm, idx);
                }
            }
            var_perm = 0.0; // Clamp near-zero variance including small negative values
        }

        se[idx] = sqrt(var_perm);

        // Improved p-value calculation
        double p_raw = (count_ge[idx] + 1.0) / n_rand_plus_1;
        pvalue[idx] = fmin(1.0, fmax(0.0, p_raw)); // Clamp p-value to [0, 1]

        // Improved z-score calculation
        if (se[idx] > rel_epsilon) {
            zscore[idx] = (beta[idx] - mean_perm) / se[idx];
        } else {
             if (fabs(beta[idx] - mean_perm) < rel_epsilon) {
                 zscore[idx] = 0.0;
             } else {
                 zscore[idx] = (beta[idx] > mean_perm) ? INFINITY : -INFINITY;
             }
        }
    }

// --- Cleanup ---
cleanup_sparse: // Unique label for sparse function cleanup
     // fprintf(stderr, "DEBUG: Entering sparse cleanup block (exit_status=%d).\n", exit_status); fflush(stderr); // Mark entry to cleanup
    // Clean up VSL resources
    if (stream != NULL) {
        vslDeleteStream(&stream);
    }

    // Clean up MKL sparse matrix
    if (Y_handle != NULL) {
        mkl_sparse_destroy(Y_handle);
    }

    // Free all aligned memory using mkl_free
    if (XtX != NULL) { mkl_free(XtX); }
    if (Xt != NULL) { mkl_free(Xt); } // Cleanup Xt
    if (T != NULL) { mkl_free(T); }   // Cleanup T
    if (T_transpose != NULL) { mkl_free(T_transpose); }
    if (beta_transpose != NULL) { mkl_free(beta_transpose); }
    if (abs_beta != NULL) { mkl_free(abs_beta); }
    if (perm != NULL) { mkl_free(perm); }
    if (T_transpose_perm != NULL) { mkl_free(T_transpose_perm); }
    if (sum_b != NULL) { mkl_free(sum_b); }
    if (sum_b2 != NULL) { mkl_free(sum_b2); }
    if (count_ge != NULL) { mkl_free(count_ge); }

    // If sparse calculation fails, fill outputs with NaN (beta might be valid)
    if (exit_status != 0 && total_elements > 0) {
         // fprintf(stderr, "DEBUG: Filling sparse outputs with NAN due to error.\n"); fflush(stderr);
         #pragma omp parallel for schedule(static)
         for (size_t idx = 0; idx < total_elements; idx++) {
             se[idx] = NAN;
             zscore[idx] = NAN;
             pvalue[idx] = NAN;
         }
         // Beta is calculated before permutations, so only NaN it if error <= 4
         if (exit_status <= 4) {
              #pragma omp parallel for schedule(static)
              for (size_t idx = 0; idx < total_elements; idx++) {
                   beta[idx] = NAN;
              }
         }
    }
    // fprintf(stderr, "DEBUG: Exiting ridge_mkl_sparse (final status %d).\n", exit_status); fflush(stderr); // Mark final exit

    return exit_status; // Return 0 on success, non-zero on error
}

// --- Helper Function: Permute T columns ---
static void permute_T_columns(double *T_permuted, const double *T, const int *indices,
                              int p, int n) {
    // Permutes columns of T (p x n) according to indices array (length n)
    // T and T_permuted are row-major arrays (p*n elements)
    #pragma omp parallel for if(p*n > MIN_PARALLEL_SIZE) schedule(static) collapse(2)
    for (int i = 0; i < p; i++) {     // Iterate through rows (features)
        for (int j = 0; j < n; j++) { // Iterate through columns (original gene index)
            int permuted_col = indices[j]; // Get the new column index for original column j
            if (permuted_col < 0 || permuted_col >= n) { // Bounds check
                #pragma omp critical (permute_error)
                { fprintf(stderr, "ERROR (permute_T_columns): Invalid permuted index %d (n=%d).\n", permuted_col, n); }
                // Handle error? Maybe set to NaN or skip? For now, just copy original.
                 T_permuted[i*n + j] = T[i*n + j]; // Or NAN
            } else {
                T_permuted[i*n + j] = T[i*n + permuted_col];
            }
        }
    }
}

// --- Helper Function: Calculate T-test Statistics ---
static int calculate_ttest_direct(
    const double *X, const double *Y, const double *T,
    const double *beta, double *se, double *zscore,
    double *pvalue, int n, int p, int m,
    double lambda, double *df_out
) {
    // --- Setup & Resource Allocation ---
    *df_out = NAN; // Initialize output df
    int exit_status = 0;

    double *Y_hat = NULL;
    double *residuals = NULL;
    double *sigma2 = NULL;
    double *TXT = NULL;           // T * X^T for calculating trace(H)
    double *T_row_sq_sum = NULL;  // Row sums of squares of T
    double *XtX = NULL;           // For OLS case
    double *XtX_inv = NULL;       // For OLS case
    int *ipiv = NULL;             // For OLS case inversion
    double *X_transpose_ttest = NULL; // For explicit X^T in Ridge case
    double *eig_values = NULL;    // For eigenvalue-based df calculation
    double *eig_work = NULL;      // Work array for eigenvalue calculation


    // Allocate memory for common arrays
    Y_hat = (double*)mkl_malloc((size_t)n * m * sizeof(double), 64);
    residuals = (double*)mkl_malloc((size_t)n * m * sizeof(double), 64);
    sigma2 = (double*)mkl_malloc(m * sizeof(double), 64);

    if (!Y_hat || !residuals || !sigma2) {
        fprintf(stderr, "ERROR: Memory allocation failed in t-test calculation (common)\n");
        exit_status = 1;
        goto ttest_cleanup;
    }

    // Zero-initialize sigma2 array
    memset(sigma2, 0, m * sizeof(double));

    // --- Calculate Y_hat = X * beta ---
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, m, p,
                1.0, X, p, beta, m,
                0.0, Y_hat, m);

    // --- Calculate Residuals = Y - Y_hat ---
    memcpy(residuals, Y, (size_t)n * m * sizeof(double));

    // Subtract Y_hat from residuals (which contains Y)
    #pragma omp parallel for schedule(static) collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            residuals[i*m + j] -= Y_hat[i*m + j];
        }
    }

    // --- Calculate Degrees of Freedom (df) and Variance (sigma2) ---
    double df = NAN;

    if (lambda < EPS) { // OLS Case
        df = (double)(n > p ? n - p : 1);
        if (df < 1) df = 1.0;

        // Calculate sigma2 (variance) for each column
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < m; j++) {
            double sum_sq = 0.0;
            for (int i = 0; i < n; i++) {
                double r = residuals[i*m + j];
                sum_sq += r * r;
            }
            sigma2[j] = sum_sq / df;
        }

        // For OLS, calculate standard errors using (X'X)^-1
        XtX = (double*)mkl_malloc((size_t)p * p * sizeof(double), 64);
        XtX_inv = (double*)mkl_malloc((size_t)p * p * sizeof(double), 64);
        if (!XtX || !XtX_inv) {
            fprintf(stderr, "ERROR: Memory allocation failed for XtX matrices\n");
            exit_status = 1;
            goto ttest_cleanup;
        }

        // Compute X'X
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    p, p, n,
                    1.0, X, p, X, p,
                    0.0, XtX, p);

        // Invert X'X using LAPACK
        memcpy(XtX_inv, XtX, (size_t)p * p * sizeof(double));

        // LU factorization and inversion
        ipiv = (int*)mkl_malloc(p * sizeof(int), 64);
        if (!ipiv) {
            fprintf(stderr, "ERROR: Memory allocation failed for ipiv\n");
            exit_status = 1;
            goto ttest_cleanup;
        }

        int info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, p, p, XtX_inv, p, ipiv);
        if (info != 0) {
            fprintf(stderr, "ERROR: LAPACKE_dgetrf failed with info = %d in t-test OLS\n", info);
            exit_status = 3; // Factorization error
            goto ttest_cleanup; // Jump to local cleanup
        }

        info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, p, XtX_inv, p, ipiv);
        if (info != 0) {
            fprintf(stderr, "ERROR: LAPACKE_dgetri failed with info = %d in t-test OLS\n", info);
            exit_status = 4; // Inversion error
            goto ttest_cleanup; // Jump to local cleanup
        }

        // Calculate standard errors
        #pragma omp parallel for schedule(static) collapse(2)
        for (int i = 0; i < p; i++) {
            for (int j = 0; j < m; j++) {
                double diag_val = XtX_inv[i*p + i];
                double se_var = sigma2[j] * diag_val;
                if (se_var < EPS && se_var >= 0) se_var = 0.0; // Clamp small positive variance to zero
                se[i*m + j] = (se_var >= 0) ? sqrt(se_var) : NAN; // Check for negative variance before sqrt
            }
        }

    } else { // Ridge Case
        // --- Method 1: Using eigenvalue decomposition of X'X for more stable df ---
        XtX = (double*)mkl_malloc((size_t)p * p * sizeof(double), 64);
        if (!XtX) {
            fprintf(stderr, "ERROR: Memory allocation failed for XtX in ridge df calculation\n");
            exit_status = 1;
            goto ttest_cleanup;
        }

        // Compute X'X
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    p, p, n,
                    1.0, X, p, X, p,
                    0.0, XtX, p);

        // Allocate for eigenvalue computation
        eig_values = (double*)mkl_malloc(p * sizeof(double), 64);
        // Allocate workspace (size determined by LAPACKE documentation)
        eig_work = (double*)mkl_malloc((4*p) * sizeof(double), 64);
        if (!eig_values || !eig_work) {
            fprintf(stderr, "ERROR: Memory allocation failed for eigenvalue calculation\n");
            exit_status = 1;
            goto ttest_cleanup;
        }

        // Copy XtX since DSYEV overwrites input
        XtX_inv = (double*)mkl_malloc((size_t)p * p * sizeof(double), 64);
        if (!XtX_inv) {
            fprintf(stderr, "ERROR: Memory allocation failed for XtX copy\n");
            exit_status = 1;
            goto ttest_cleanup;
        }
        memcpy(XtX_inv, XtX, (size_t)p * p * sizeof(double));

        // Compute eigenvalues (using DSYEV)
        // Note: MKL's LAPACKE_dsyev would be better, this is just for illustration
        char jobz = 'N'; // 'N' = eigenvalues only
        char uplo = 'U'; // 'U' = upper triangle of A is stored
        int info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, jobz, uplo, p, XtX_inv, p, eig_values);
        if (info != 0) {
            fprintf(stderr, "ERROR: Eigenvalue computation failed with code %d\n", info);
            // Fall back to the old trace method if eigendecomposition fails
            fprintf(stderr, "Warning: Falling back to trace method for df calculation\n");
        } else {
            // Calculate effective df using eigenvalues: df = sum(λ_i / (λ_i + λ))
            df = 0.0;
            for (int i = 0; i < p; i++) {
                if (eig_values[i] > 0) { // Ensure eigenvalue is positive
                    df += eig_values[i] / (eig_values[i] + lambda);
                }
            }
            fprintf(stderr, "INFO: Ridge df calculated using eigenvalues: %.2f\n", df);
            
            // Sanity check on df
            if (df <= 0 || df >= n) {
                fprintf(stderr, "WARNING: Eigenvalue df calculation gave suspicious value (%.2f). Falling back to trace method.\n", df);
                df = NAN; // Reset df to trigger fallback
            }
        }

        // If eigenvalue method failed or gave suspicious results, fall back to trace method
        if (isnan(df)) {
            // For ridge regression, calculate effective degrees of freedom using trace
            TXT = (double*)mkl_malloc((size_t)p * p * sizeof(double), 64);
            if (!TXT) {
                fprintf(stderr, "ERROR: Memory allocation failed for TXT matrix\n");
                exit_status = 1;
                goto ttest_cleanup;
            }

            // We need explicit X' for the trace calculation
            X_transpose_ttest = (double*)mkl_malloc((size_t)p * n * sizeof(double), 64);
            if (!X_transpose_ttest) {
                fprintf(stderr, "ERROR: Memory allocation failed for X_transpose_ttest\n");
                exit_status = 1;
                goto ttest_cleanup;
            }
            
            // Transpose X (n x p) to X_transpose_ttest (p x n)
            #pragma omp parallel for schedule(static) collapse(2)
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < p; j++) {
                    X_transpose_ttest[j * n + i] = X[i * p + j];
                }
            }

            // Calculate TXT = T * X' using explicit X'
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        p, p, n,
                        1.0, T, n,              // T (p x n), lda=n
                        X_transpose_ttest, n,   // X' (n x p), ldb=n
                        0.0, TXT, p);           // TXT (p x p), ldc=p

            // Calculate trace(H) = trace(T*X^T) with better numerical stability
            double trace_H = 0.0;
            for (int i = 0; i < p; i++) {
                trace_H += TXT[i*p + i];
            }

            // Apply bounds on trace(H) to prevent numerical issues
            if (trace_H < 0) {
                fprintf(stderr, "WARNING: Negative trace(H) = %.2f detected, clamping to 0\n", trace_H);
                trace_H = 0;
            }
            if (trace_H > n) {
                fprintf(stderr, "WARNING: trace(H) = %.2f > n = %d detected, clamping to n\n", trace_H, n);
                trace_H = n;
            }

            df = (double)n - trace_H;
            fprintf(stderr, "INFO: Ridge df calculated using trace method: %.2f (trace_H = %.2f)\n", df, trace_H);
        }

        // Final check on df validity
        if (df <= 0) {
            fprintf(stderr, "WARNING: Effective df <= 0 (%.2f). Setting df=1.\n", df);
            df = 1.0;
        }

        // Calculate sigma2 (variance) for each column
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < m; j++) {
            double sum_sq = 0.0;
            for (int i = 0; i < n; i++) {
                double r = residuals[i*m + j];
                sum_sq += r * r;
            }
            sigma2[j] = sum_sq / df;
        }

        // For ridge, calculate row sums of squares of T
        T_row_sq_sum = (double*)mkl_malloc(p * sizeof(double), 64);
        if (!T_row_sq_sum) {
            fprintf(stderr, "ERROR: Memory allocation failed for T_row_sq_sum\n");
            exit_status = 1;
            goto ttest_cleanup;
        }

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < p; i++) {
            double row_sum_sq = 0.0;
            for (int j = 0; j < n; j++) {
                double T_ij = T[i*n + j];
                row_sum_sq += T_ij * T_ij;
            }
            T_row_sq_sum[i] = row_sum_sq;
        }

        // Calculate standard errors
        #pragma omp parallel for schedule(static) collapse(2)
        for (int i = 0; i < p; i++) {
            for (int j = 0; j < m; j++) {
                double se_var = sigma2[j] * T_row_sq_sum[i];
                // Use a relative epsilon scaled by the magnitude of the variance
                double rel_eps = fmax(EPS, fabs(sigma2[j] * T_row_sq_sum[i]) * EPS * 10);
                if (se_var < rel_eps && se_var >= -rel_eps) se_var = 0.0; // Clamp small variance to zero
                se[i*m + j] = (se_var >= 0) ? sqrt(se_var) : NAN; // Check for negative variance before sqrt
            }
        }
    } // End OLS/Ridge blocks

    // --- Calculate t-statistics and p-values ---
    if (isnan(df)) {
        fprintf(stderr, "ERROR: Degrees of freedom is NaN after calculation.\n");
        exit_status = 7; // T-test error

        // Fill outputs with NaN
        size_t pm_size = (size_t)p * m;
        #pragma omp parallel for schedule(static)
        for (size_t idx = 0; idx < pm_size; idx++) {
            zscore[idx] = NAN;
            pvalue[idx] = NAN;
        }
    } else {
        // Calculate t-statistics and p-values
        size_t pm_size = (size_t)p * m;
        #pragma omp parallel for schedule(static)
        for (size_t idx = 0; idx < pm_size; idx++) {
            double beta_val = beta[idx];
            double se_val = se[idx];

            // Calculate t-statistic with protection against zero SE
            double rel_eps = fmax(EPS, fabs(beta_val) * EPS);
            if (!isnan(se_val) && se_val > rel_eps) {
                zscore[idx] = beta_val / se_val;
            } else {
                zscore[idx] = (fabs(beta_val) < rel_eps) ? 0.0 :
                             (beta_val > 0 ? INFINITY : -INFINITY);
            }

            // Calculate p-value
            if (isfinite(zscore[idx])) {
                // Call improved t-distribution CDF
                double t_abs = fabs(zscore[idx]);
                pvalue[idx] = 2.0 * (1.0 - cdf_tdist(t_abs, df));
                pvalue[idx] = fmax(0.0, fmin(1.0, pvalue[idx])); // Clamp to [0,1]
            } else {
                pvalue[idx] = (zscore[idx] == 0.0) ? 1.0 : 0.0; // Handle infinity case
            }
        }
    }

    *df_out = df; // Store final df value

ttest_cleanup:
    // Clean up resources
    if (Y_hat) mkl_free(Y_hat);
    if (residuals) mkl_free(residuals);
    if (sigma2) mkl_free(sigma2);
    if (TXT) mkl_free(TXT);
    if (T_row_sq_sum) mkl_free(T_row_sq_sum);
    if (XtX) mkl_free(XtX);
    if (XtX_inv) mkl_free(XtX_inv);
    if (ipiv) mkl_free(ipiv);
    if (X_transpose_ttest) mkl_free(X_transpose_ttest);
    if (eig_values) mkl_free(eig_values);
    if (eig_work) mkl_free(eig_work);

    return exit_status;
}

// Improved t-distribution CDF function - more accurate for small df
static double cdf_tdist(double t, double df) {
    // For large df (>30), use normal approximation
    if (df > 30.0) {
        double z = t;
        return 0.5 * (1.0 + erf(z / sqrt(2.0)));
    }
    
    // For smaller df, use a better approximation
    // Based on approximation from "Handbook of Mathematical Functions"
    // by Abramowitz and Stegun, formula 26.7.8
    double x = df / (df + t*t);
    double a = df / 2.0;
    
    // Incomplete beta function approximation
    // This is still an approximation - for production use, 
    // consider using a specialized statistical library
    double y = a * log(x) - log(a) - 0.5 * log(1.0 - x);
    y = exp(y);
    
    // Adjustment for approximation
    double result = 1.0 - 0.5 * y;
    
    // Ensure result is in [0,1]
    return fmax(0.0, fmin(1.0, result));
}