/*
 * ridge_gsl.c - Optimized C implementation for ridge regression using GSL
 * Supports permutation testing (via T-column permutation) and t-tests.
 * Designed for direct memory access from Python/Cython.
 * Includes OpenMP parallelization and custom parallel DGEMM.
 */

// Define GSL feature test macros *before* any GSL includes
#define GSL_DISABLE_DEPRECATED // Often recommended
// #define GSL_CBLAS_ENUMS_ONLY   // Let's REMOVE this and rely on include order

#include "ridge_gsl.h" // Function prototypes and definitions
#include <stdio.h>      // For fprintf, stderr, fflush
#include <stdlib.h>     // For malloc, free, abort, calloc
#include <string.h>     // For memcpy, memset
#include <math.h>       // For fabs, sqrt, isnan, isinf, copysign, round, NAN, INFINITY, fmax, fmin
#include <time.h>       // For time() seeding
#include <unistd.h>     // For getpid()

// OpenMP header for parallelization directives
#ifdef _OPENMP
#include <omp.h>
#endif

// Include GSL Headers FIRST to get its CBLAS definitions
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>      // GSL's CBLAS interface (including enums)
#include <gsl/gsl_linalg.h>    // Linear algebra (LU, Cholesky, inversion)
#include <gsl/gsl_cdf.h>       // Cumulative Distribution Functions (t-dist)
#include <gsl/gsl_math.h>      // GSL math constants and functions (GSL_POSINF, isnan, isinf)
#include <gsl/gsl_errno.h>     // GSL error handling
#include <gsl/gsl_rng.h>       // Random Number Generators
#include <gsl/gsl_randist.h>   // Random distributions and shuffling


// OpenBLAS header for thread control (conditional)
// Include AFTER GSL headers. It might have include guards preventing redefinition,
// or we rely on the compiler using the first definition encountered (GSL's).
#ifdef HAVE_OPENBLAS
#include <cblas.h> // OpenBLAS CBLAS header (may conflict if not guarded)
// Declare OpenBLAS thread control functions
extern int openblas_get_num_threads(void);
extern void openblas_set_num_threads(int num_threads);
#endif


// --- Constants ---
#define EPS 1e-12 // Epsilon for floating-point comparisons
#define MIN_PARALLEL_SIZE 5000 // Threshold for enabling OpenMP parallel for loops
#define GSL_SUCCESS 0 // Explicit definition for GSL success code

// --- Error Codes ---
#define RIDGE_SUCCESS 0
#define RIDGE_ERROR_NULL_POINTER 1
#define RIDGE_ERROR_INVALID_DIMENSIONS 2
#define RIDGE_ERROR_ALLOCATION_FAILED 3
#define RIDGE_ERROR_MATRIX_OPERATION 4
#define RIDGE_ERROR_DECOMPOSITION 5
#define RIDGE_ERROR_INVERSION 6
#define RIDGE_ERROR_PERMUTATION 7
#define RIDGE_ERROR_TTEST 8
#define RIDGE_ERROR_DGEMM_FAILED 9

// --- OpenBLAS Thread Control Functions ---
#ifdef HAVE_OPENBLAS
int ridge_gsl_set_blas_threads(int num_threads) {
    int prev_threads = 1;
    // Check if function pointer is valid (safer than header guards)
    if (openblas_get_num_threads) {
        prev_threads = openblas_get_num_threads();
    } else {
        fprintf(stderr, "WARNING: openblas_get_num_threads symbol not found, assuming 1 thread.\n");
    }

    if (num_threads >= 0) { // Allow 0 for default
        if(openblas_set_num_threads) {
            openblas_set_num_threads(num_threads);
        } else {
             if (num_threads > 1) fprintf(stderr, "WARNING: openblas_set_num_threads symbol not found.\n");
        }
    }
    return prev_threads > 0 ? prev_threads : 1;
}

int ridge_gsl_get_blas_threads() {
    int num_threads = 1;
     if (openblas_get_num_threads) {
        num_threads = openblas_get_num_threads();
    }
    return num_threads > 0 ? num_threads : 1;
}
#else
// Fallback implementations
int ridge_gsl_set_blas_threads(int num_threads) {
    if (num_threads > 1) {
         fprintf(stderr, "WARNING: OpenBLAS not available/linked (HAVE_OPENBLAS not defined), cannot set BLAS threads > 1.\n");
         fflush(stderr);
    }
    return 1;
}

int ridge_gsl_get_blas_threads() {
    return 1;
}
#endif // HAVE_OPENBLAS


// --- Custom Parallel DGEMM Implementation ---
// Multi-threaded blocked implementation of matrix multiplication C = alpha*Op(A)*Op(B) + beta*C
// Uses CBLAS enums (assuming defined by GSL or OpenBLAS cblas.h)
static int parallel_blocked_dgemm(CBLAS_TRANSPOSE_t TransA, CBLAS_TRANSPOSE_t TransB,
                                 double alpha, const gsl_matrix *A, const gsl_matrix *B,
                                 double beta, gsl_matrix *C) {

    size_t m_dim, n_dim, k_dim;
    size_t lda, ldb, ldc;

    // Determine dimensions based on transpose operations
    if (TransA == CblasNoTrans) {
        m_dim = A->size1; k_dim = A->size2; lda = A->tda;
    } else { // CblasTrans or CblasConjTrans
        m_dim = A->size2; k_dim = A->size1; lda = A->tda;
    }

    if (TransB == CblasNoTrans) {
        n_dim = B->size2; ldb = B->tda;
        if (B->size1 != k_dim) return GSL_EBADLEN; // Inner dimensions must match
    } else { // CblasTrans or CblasConjTrans
        n_dim = B->size1; ldb = B->tda;
        if (B->size2 != k_dim) return GSL_EBADLEN; // Inner dimensions must match
    }

    // Check output matrix C dimensions
    if (C->size1 != m_dim || C->size2 != n_dim) return GSL_EBADLEN;
    ldc = C->tda;

    // Scale C by beta (Parallelized)
    #pragma omp parallel for schedule(static) if(m_dim*n_dim > MIN_PARALLEL_SIZE)
    for (size_t i = 0; i < m_dim; i++) {
        // Use pointer arithmetic for potential minor optimization
        double *C_row = C->data + i * ldc;
        if (beta == 0.0) {
            memset(C_row, 0, n_dim * sizeof(double)); // Faster zeroing?
        } else if (beta != 1.0) {
            for (size_t j = 0; j < n_dim; j++) {
                C_row[j] *= beta;
            }
        }
    }

    // --- Perform alpha * Op(A) * Op(B) calculation ---

    // Optimize the most common case: NoTrans, NoTrans
    if (TransA == CblasNoTrans && TransB == CblasNoTrans) {
        const size_t BLOCK_SIZE_M = 64;
        const size_t BLOCK_SIZE_N = 64;
        const size_t BLOCK_SIZE_K = 64;

        #pragma omp parallel
        {
            #pragma omp for collapse(2) schedule(dynamic)
            for (size_t i = 0; i < m_dim; i += BLOCK_SIZE_M) {
                for (size_t j = 0; j < n_dim; j += BLOCK_SIZE_N) {
                    size_t block_m = (i + BLOCK_SIZE_M > m_dim) ? m_dim - i : BLOCK_SIZE_M;
                    size_t block_n = (j + BLOCK_SIZE_N > n_dim) ? n_dim - j : BLOCK_SIZE_N;

                    for (size_t k = 0; k < k_dim; k += BLOCK_SIZE_K) {
                        size_t block_k = (k + BLOCK_SIZE_K > k_dim) ? k_dim - k : BLOCK_SIZE_K;

                        for (size_t bi = 0; bi < block_m; ++bi) {
                            const double *A_ptr = A->data + (i + bi) * lda + k;
                            double *C_ptr_base = C->data + (i + bi) * ldc + j;
                            for (size_t bk = 0; bk < block_k; ++bk) {
                                double a_val_alpha = alpha * A_ptr[bk]; // A->data[(i + bi) * lda + (k + bk)];
                                const double *B_ptr = B->data + (k + bk) * ldb + j;
                                double *C_ptr = C_ptr_base;
                                // Loop unrolling or vectorization pragmas could go here
                                for (size_t bj = 0; bj < block_n; ++bj) {
                                    // C->data[(i + bi) * ldc + (j + bj)] += a_val_alpha * B->data[(k + bk) * ldb + (j + bj)];
                                    C_ptr[bj] += a_val_alpha * B_ptr[bj];
                                }
                            }
                        }
                    } // end k block loop
                } // end j block loop
            } // end i block loop (omp for)
        } // end omp parallel
        return GSL_SUCCESS;

    } else {
        // Fall back to simpler (non-blocked) parallel loops for other transpose cases
        #pragma omp parallel
        {
            if (TransA == CblasNoTrans && TransB == CblasTrans) {
                #pragma omp for schedule(static)
                for (size_t i = 0; i < m_dim; ++i) {
                    for (size_t j = 0; j < n_dim; ++j) {
                        double sum = 0.0;
                        const double *A_row = A->data + i * lda;
                        const double *B_row = B->data + j * ldb; // B is transposed, use row j
                        for (size_t k = 0; k < k_dim; ++k) {
                            sum += A_row[k] * B_row[k]; // A[i,k] * B[j,k] -> A[i,k] * B'[k,j]
                        }
                        C->data[i * ldc + j] += alpha * sum;
                    }
                }
            } else if (TransA == CblasTrans && TransB == CblasNoTrans) {
                 #pragma omp for schedule(static)
                 for (size_t i = 0; i < m_dim; ++i) {
                    for (size_t j = 0; j < n_dim; ++j) {
                        double sum = 0.0;
                        for (size_t k = 0; k < k_dim; ++k) {
                            sum += A->data[k * lda + i] * B->data[k * ldb + j]; // A[k,i] * B[k,j]
                        }
                        C->data[i * ldc + j] += alpha * sum;
                    }
                }
            } else { // TransA == CblasTrans && TransB == CblasTrans
                #pragma omp for schedule(static)
                 for (size_t i = 0; i < m_dim; ++i) {
                    for (size_t j = 0; j < n_dim; ++j) {
                        double sum = 0.0;
                         for (size_t k = 0; k < k_dim; ++k) {
                            sum += A->data[k * lda + i] * B->data[j * ldb + k]; // A[k,i] * B[j,k] -> A[k,i] * B'[k,j]
                        }
                        C->data[i * ldc + j] += alpha * sum;
                    }
                }
            }
        } // end omp parallel
        return GSL_SUCCESS;
    }
}


// --- Forward Declarations for Helper Functions ---
static void permute_T_columns(double *T_permuted, const double *T, const int *indices, const size_t p, const size_t n);
static int calculate_ttest_direct(const double *X_data, const double *Y_data, const double *T_data, const double *beta_data,
                                  double *se_data, double *zscore_data, double *pvalue_data,
                                  const size_t n, const size_t p, const size_t m, double lambda,
                                  double *df_out);

// --- Main Ridge Regression Function ---
int ridge_gsl_reg(
    double *X_vec, double *Y_vec,
    int *n_pt, int *p_pt, int *m_pt,
    double *lambda_pt, double *nrand_pt,
    double *beta_vec, double *se_vec,
    double *zscore_vec, double *pvalue_vec,
    double *df_out
) {
    // --- 1. Pointer Checks ---
    if (!n_pt || !p_pt || !m_pt || !lambda_pt || !nrand_pt ||
        !X_vec || !Y_vec || !beta_vec || !se_vec || !zscore_vec || !pvalue_vec || !df_out) {
        fprintf(stderr, "FATAL ERROR (ridge_gsl_reg): NULL input pointer detected.\n"); fflush(stderr);
        if (df_out) *df_out = NAN;
        return RIDGE_ERROR_NULL_POINTER;
    }

    // --- 2. Get Dimensions and Parameters ---
    size_t n = (size_t)*n_pt;
    size_t p = (size_t)*p_pt;
    size_t m = (size_t)*m_pt;
    double lambda = *lambda_pt;
    int nrand_i = (int)round(*nrand_pt);
    *df_out = NAN;
    double calculated_df = NAN;

    // --- 3. Dimension Validation ---
    if (n == 0 || p == 0 || m == 0) {
        fprintf(stderr, "FATAL ERROR (ridge_gsl_reg): Invalid dimensions n=%zu, p=%zu, m=%zu.\n", n, p, m); fflush(stderr);
        *df_out = NAN;
        if (p > 0 && m > 0) {
            size_t pm_size = p * m;
            #pragma omp parallel for if(pm_size > MIN_PARALLEL_SIZE*10) schedule(static)
            for(size_t k=0; k<pm_size; ++k) {
                 if(beta_vec) beta_vec[k] = NAN; if(se_vec) se_vec[k] = NAN;
                 if(zscore_vec) zscore_vec[k] = NAN; if(pvalue_vec) pvalue_vec[k] = NAN;
            }
        }
        return RIDGE_ERROR_INVALID_DIMENSIONS;
    }

    // --- 4. Create GSL Matrix Views ---
    gsl_matrix_view X_view = gsl_matrix_view_array(X_vec, n, p);
    gsl_matrix_view Y_view = gsl_matrix_view_array(Y_vec, n, m);
    gsl_matrix_view beta_view = gsl_matrix_view_array(beta_vec, p, m);

    gsl_matrix *X = &X_view.matrix;
    gsl_matrix *Y = &Y_view.matrix;
    gsl_matrix *beta = &beta_view.matrix;

    // --- 5. Allocate Working Matrices ---
    gsl_matrix *XtX = NULL;
    gsl_matrix *XtX_lambda = NULL;
    gsl_matrix *XtX_inv = NULL;
    gsl_matrix *T = NULL;
    gsl_permutation *perm = NULL;

    int return_code = RIDGE_SUCCESS;
    volatile int local_error_flag = 0;

    XtX = gsl_matrix_alloc(p, p);
    XtX_lambda = gsl_matrix_alloc(p, p);
    XtX_inv = gsl_matrix_alloc(p, p);
    T = gsl_matrix_alloc(p, n);
    perm = gsl_permutation_alloc(p);

    if (!XtX || !XtX_lambda || !XtX_inv || !T || !perm) {
        fprintf(stderr, "FATAL ERROR (ridge_gsl_reg): Failed core GSL matrix allocation.\n"); fflush(stderr);
        return_code = RIDGE_ERROR_ALLOCATION_FAILED; goto cleanup_core;
    }

    // --- 6. Calculate T = (X'X + lambda*I)^-1 * X' ---
    // *** Use custom parallel dgemm ***
    if (parallel_blocked_dgemm(CblasTrans, CblasNoTrans, 1.0, X, X, 0.0, XtX) != GSL_SUCCESS) {
        fprintf(stderr, "ERROR (ridge_gsl_reg): parallel_blocked_dgemm for XtX failed.\n"); fflush(stderr);
        return_code = RIDGE_ERROR_DGEMM_FAILED; goto cleanup_core;
    }
    if (gsl_matrix_memcpy(XtX_lambda, XtX) != GSL_SUCCESS) {
        fprintf(stderr, "ERROR (ridge_gsl_reg): gsl_matrix_memcpy failed.\n"); fflush(stderr);
        return_code = RIDGE_ERROR_MATRIX_OPERATION; goto cleanup_core;
    }
    for (size_t i = 0; i < p; i++) {
        gsl_matrix_set(XtX_lambda, i, i, gsl_matrix_get(XtX_lambda, i, i) + lambda);
    }

    int signum;
    int chol_status = gsl_linalg_cholesky_decomp1(XtX_lambda);

    if (chol_status == GSL_SUCCESS) {
        if (gsl_linalg_cholesky_invert(XtX_lambda) != GSL_SUCCESS) {
            fprintf(stderr, "ERROR (ridge_gsl_reg): gsl_linalg_cholesky_invert failed.\n"); fflush(stderr);
            return_code = RIDGE_ERROR_INVERSION; goto cleanup_core;
        }
        if (gsl_matrix_memcpy(XtX_inv, XtX_lambda) != GSL_SUCCESS) {
             fprintf(stderr, "ERROR (ridge_gsl_reg): gsl_matrix_memcpy for XtX_inv failed.\n"); fflush(stderr);
             return_code = RIDGE_ERROR_MATRIX_OPERATION; goto cleanup_core;
        }
    } else {
        if(chol_status != GSL_EDOM) {
            fprintf(stderr, "ERROR (ridge_gsl_reg): Cholesky decomposition failed unexpectedly (errno %d: %s). Trying LU.\n", chol_status, gsl_strerror(chol_status));
        } else {
             fprintf(stderr, "WARNING (ridge_gsl_reg): Cholesky decomposition failed (matrix not positive definite). Falling back to LU decomposition.\n");
        }
        fflush(stderr);
        if (gsl_matrix_memcpy(XtX_lambda, XtX) != GSL_SUCCESS) {
             fprintf(stderr, "ERROR (ridge_gsl_reg): memcpy reset for XtX_lambda (LU) failed.\n"); fflush(stderr);
             return_code = RIDGE_ERROR_MATRIX_OPERATION; goto cleanup_core;
        }
        for (size_t i = 0; i < p; i++) {
            gsl_matrix_set(XtX_lambda, i, i, gsl_matrix_get(XtX_lambda, i, i) + lambda);
        }

        if (gsl_linalg_LU_decomp(XtX_lambda, perm, &signum) != GSL_SUCCESS) {
            fprintf(stderr, "ERROR (ridge_gsl_reg): gsl_linalg_LU_decomp failed.\n"); fflush(stderr);
            return_code = RIDGE_ERROR_DECOMPOSITION; goto cleanup_core;
        }
        if (gsl_linalg_LU_invert(XtX_lambda, perm, XtX_inv) != GSL_SUCCESS) {
            fprintf(stderr, "ERROR (ridge_gsl_reg): gsl_linalg_LU_invert failed.\n"); fflush(stderr);
            return_code = RIDGE_ERROR_INVERSION; goto cleanup_core;
        }
    }

    // Calculate T = XtX_inv * X'
    // *** Use custom parallel dgemm ***
    if (parallel_blocked_dgemm(CblasNoTrans, CblasTrans, 1.0, XtX_inv, X, 0.0, T) != GSL_SUCCESS) {
        fprintf(stderr, "ERROR (ridge_gsl_reg): parallel_blocked_dgemm for T failed.\n"); fflush(stderr);
        return_code = RIDGE_ERROR_DGEMM_FAILED; goto cleanup_core;
    }

    // --- 7. Calculate beta = T * Y ---
    // *** Use custom parallel dgemm ***
    if (parallel_blocked_dgemm(CblasNoTrans, CblasNoTrans, 1.0, T, Y, 0.0, beta) != GSL_SUCCESS) {
        fprintf(stderr, "ERROR (ridge_gsl_reg): parallel_blocked_dgemm for beta failed.\n"); fflush(stderr);
        return_code = RIDGE_ERROR_DGEMM_FAILED; goto cleanup_core;
    }

    // --- 8. Perform Significance Test ---
    if (nrand_i > 0) {
        // --- Permutation Test ---
        double *aver = NULL;
        double *aver_sq = NULL;
        double *pvalue_counts = NULL;
        size_t pm_size = p * m;

        aver = (double*)calloc(pm_size, sizeof(double));
        aver_sq = (double*)calloc(pm_size, sizeof(double));
        pvalue_counts = (double*)calloc(pm_size, sizeof(double));

        if (!aver || !aver_sq || !pvalue_counts ) {
            fprintf(stderr, "FATAL ERROR (ridge_gsl_reg): Failed allocation for permutation accumulators.\n"); fflush(stderr);
            return_code = RIDGE_ERROR_ALLOCATION_FAILED; goto cleanup_perm;
        }

        #pragma omp parallel if(nrand_i > 1 && pm_size > MIN_PARALLEL_SIZE)
        {
            // Thread-local variables
            double *local_beta_rand = (double*)malloc(pm_size * sizeof(double));
            double *local_T_permuted = (double*)malloc(p * n * sizeof(double));
            int *local_indices = (int*)malloc(n * sizeof(int));
            gsl_rng *local_rng = gsl_rng_alloc(gsl_rng_mt19937);
            double *local_aver = (double*)calloc(pm_size, sizeof(double));
            double *local_aver_sq = (double*)calloc(pm_size, sizeof(double));
            double *local_pvalue_counts = (double*)calloc(pm_size, sizeof(double));
            int thread_id = omp_get_thread_num();
            int thread_error = 0;

            if (!local_beta_rand || !local_T_permuted || !local_indices || !local_rng || !local_aver || !local_aver_sq || !local_pvalue_counts) {
                #pragma omp critical
                { fprintf(stderr, "ERROR (ridge_gsl_reg): Thread %d failed thread-local alloc.\n", thread_id); fflush(stderr); }
                thread_error = 1;
                #pragma omp atomic write
                local_error_flag = 1;
            } else {
                for (size_t i = 0; i < n; i++) { local_indices[i] = (int)i; }
                unsigned long seed = (unsigned long)time(NULL) ^ (unsigned long)getpid() ^ (unsigned long)((thread_id + 1) * 314159);
                gsl_rng_set(local_rng, seed);
            }

            gsl_matrix_view local_T_perm_view;
            gsl_matrix_view local_beta_rand_view;
            gsl_matrix *local_T_perm_mat = NULL;
            gsl_matrix *local_beta_rand_mat = NULL;
            if (!thread_error) {
                 local_T_perm_view = gsl_matrix_view_array(local_T_permuted, p, n);
                 local_beta_rand_view = gsl_matrix_view_array(local_beta_rand, p, m);
                 local_T_perm_mat = &local_T_perm_view.matrix;
                 local_beta_rand_mat = &local_beta_rand_view.matrix;
            }

            #pragma omp for schedule(dynamic) nowait
            for (int i = 0; i < nrand_i; i++) {
                int global_err_check;
                #pragma omp atomic read
                global_err_check = local_error_flag;
                if (thread_error || global_err_check) continue;

                gsl_ran_shuffle(local_rng, local_indices, n, sizeof(int));
                permute_T_columns(local_T_permuted, T->data, local_indices, p, n);

                // *** Use custom parallel dgemm ***
                if (parallel_blocked_dgemm(CblasNoTrans, CblasNoTrans, 1.0, local_T_perm_mat, Y, 0.0, local_beta_rand_mat) != GSL_SUCCESS) {
                     #pragma omp critical
                     { fprintf(stderr, "ERROR (ridge_gsl_reg): Thread %d parallel_blocked_dgemm failed in perm %d.\n", thread_id, i); fflush(stderr); }
                     thread_error = 1;
                     #pragma omp atomic write
                     local_error_flag = 1;
                     continue;
                }

                for (size_t k = 0; k < pm_size; k++) {
                    double br_val = local_beta_rand[k];
                    double b_val = beta_vec[k];
                    local_aver[k] += br_val;
                    local_aver_sq[k] += br_val * br_val;
                    if (fabs(br_val) >= fabs(b_val) - EPS) {
                        local_pvalue_counts[k] += 1.0;
                    }
                }
            } // end omp for

            if (!thread_error && local_aver) {
                #pragma omp critical
                {
                    int global_err_check_crit;
                    #pragma omp atomic read
                    global_err_check_crit = local_error_flag;
                    if (!global_err_check_crit) {
                        for (size_t k = 0; k < pm_size; k++) {
                            aver[k] += local_aver[k];
                            aver_sq[k] += local_aver_sq[k];
                            pvalue_counts[k] += local_pvalue_counts[k];
                        }
                    }
                }
            }

            free(local_beta_rand);
            free(local_T_permuted);
            free(local_indices);
            if (local_rng) gsl_rng_free(local_rng);
            free(local_aver);
            free(local_aver_sq);
            free(local_pvalue_counts);

        } // end omp parallel

        if(local_error_flag) {
             fprintf(stderr, "ERROR (ridge_gsl_reg): At least one thread encountered an error during permutations.\n"); fflush(stderr);
             return_code = RIDGE_ERROR_PERMUTATION;
             goto cleanup_perm;
        }


        // --- 9. Finalize Permutation Statistics ---
        double norm_factor = (nrand_i > 0) ? (1.0 / nrand_i) : 1.0;
        #pragma omp parallel for if(pm_size > MIN_PARALLEL_SIZE) schedule(static)
        for (size_t k = 0; k < pm_size; k++) {
            double mean_perm = aver[k] * norm_factor;
            double mean_sq_perm = aver_sq[k] * norm_factor;
            double var_perm = mean_sq_perm - (mean_perm * mean_perm);

            if (var_perm < 0.0 && var_perm > -EPS) var_perm = 0.0;
            else if (var_perm < -EPS) {
                 #pragma omp critical
                 { fprintf(stderr, "WARNING (ridge_gsl_reg): Negative variance (%.2e) at element %zu. SE set to NaN.\n", var_perm, k); fflush(stderr); }
                 var_perm = NAN;
            }

            se_vec[k] = sqrt(var_perm);

            if (!isnan(se_vec[k]) && se_vec[k] > EPS) {
                zscore_vec[k] = (beta_vec[k] - mean_perm) / se_vec[k];
            } else if (!isnan(se_vec[k])) {
                zscore_vec[k] = (fabs(beta_vec[k] - mean_perm) < EPS) ? 0.0 : copysign(INFINITY, beta_vec[k] - mean_perm);
            } else {
                 zscore_vec[k] = NAN;
            }

            if (nrand_i > 0) {
                 pvalue_vec[k] = (pvalue_counts[k] + 1.0) / (nrand_i + 1.0);
                 pvalue_vec[k] = fmax(0.0, fmin(1.0, pvalue_vec[k]));
            } else {
                 pvalue_vec[k] = NAN;
            }
        }
        *df_out = NAN;

    // --- Cleanup for permutation-specific arrays ---
    cleanup_perm:
        free(aver);
        free(aver_sq);
        free(pvalue_counts);
        if (return_code == RIDGE_SUCCESS && local_error_flag) {
            return_code = RIDGE_ERROR_PERMUTATION;
        }
        if(return_code != RIDGE_SUCCESS) goto cleanup_core;

    } else { // nrand_i <= 0
        // --- T-Test ---
        int ttest_status = calculate_ttest_direct(X_vec, Y_vec, T->data, beta_vec, se_vec, zscore_vec, pvalue_vec, n, p, m, lambda, &calculated_df);
        if (ttest_status != RIDGE_SUCCESS) {
            fprintf(stderr, "ERROR (ridge_gsl_reg): T-test calculation failed with status %d.\n", ttest_status); fflush(stderr);
            if (return_code == RIDGE_SUCCESS) return_code = RIDGE_ERROR_TTEST;
            *df_out = NAN;
        } else {
            *df_out = calculated_df;
        }
        // Fall through to cleanup_core
    }


cleanup_core:
    gsl_matrix_free(XtX);
    gsl_matrix_free(XtX_lambda);
    gsl_matrix_free(XtX_inv);
    gsl_matrix_free(T);
    gsl_permutation_free(perm);

    // NaN filling on error
    if (return_code != RIDGE_SUCCESS) {
        if (p > 0 && m > 0) {
            size_t pm_size = p * m;
            int fill_needed = 1;
            if (return_code == RIDGE_ERROR_PERMUTATION && nrand_i > 0) {
                 fill_needed = 0; // Avoid double NaNing if perm test failed
            }

            if(fill_needed) {
                #pragma omp parallel for if(pm_size > MIN_PARALLEL_SIZE*10) schedule(static)
                for(size_t k=0; k<pm_size; ++k) {
                    int should_nan_stats = (return_code >= RIDGE_ERROR_MATRIX_OPERATION);
                    int should_nan_beta = (return_code <= RIDGE_ERROR_ALLOCATION_FAILED ||
                                           return_code == RIDGE_ERROR_INVALID_DIMENSIONS ||
                                           return_code == RIDGE_ERROR_DGEMM_FAILED ||
                                           return_code == RIDGE_ERROR_MATRIX_OPERATION || // If core matrix ops failed
                                           return_code == RIDGE_ERROR_DECOMPOSITION ||
                                           return_code == RIDGE_ERROR_INVERSION );

                    if(should_nan_stats) {
                        if(se_vec) se_vec[k] = NAN;
                        if(zscore_vec) zscore_vec[k] = NAN;
                        if(pvalue_vec) pvalue_vec[k] = NAN;
                    }
                    if(should_nan_beta) {
                         if(beta_vec) beta_vec[k] = NAN;
                         // If beta is NaN, stats should also be NaN
                         if(se_vec) se_vec[k] = NAN;
                         if(zscore_vec) zscore_vec[k] = NAN;
                         if(pvalue_vec) pvalue_vec[k] = NAN;
                    }
                }
            }
        }
        if (df_out) *df_out = NAN;
        fprintf(stderr, "INFO: ridge_gsl_reg C function exited with error code %d. Output arrays may contain NaN.\n", return_code); fflush(stderr);
    }

    return return_code;
}

// --- Helper Function: Permute T columns ---
static void permute_T_columns(double *T_permuted, const double *T, const int *indices,
                              const size_t p, const size_t n) {
    #pragma omp parallel for if(p*n > MIN_PARALLEL_SIZE) schedule(static) collapse(2)
    for (size_t i = 0; i < p; i++) {
        for (size_t j = 0; j < n; j++) {
            int permuted_col = indices[j];
            if (permuted_col < 0 || (size_t)permuted_col >= n) {
                #pragma omp critical (PermuteError)
                { fprintf(stderr, "ERROR (permute_T_columns): Invalid permuted index %d (n=%zu) at row %zu, col %zu.\n", permuted_col, n, i, j); fflush(stderr); }
                T_permuted[i*n + j] = NAN;
            } else {
                T_permuted[i*n + j] = T[i*n + permuted_col];
            }
        }
    }
}


// --- Helper Function: Calculate T-test Statistics ---
static int calculate_ttest_direct(
    const double *X_data, const double *Y_data, const double *T_data, const double *beta_data,
    double *se_data, double *zscore_data, double *pvalue_data,
    const size_t n, const size_t p, const size_t m, double lambda,
    double *df_out
) {
    *df_out = NAN;

    gsl_matrix_view X_view = gsl_matrix_view_array((double*)X_data, n, p);
    gsl_matrix_view Y_view = gsl_matrix_view_array((double*)Y_data, n, m);
    gsl_matrix_view T_view = gsl_matrix_view_array((double*)T_data, p, n);
    gsl_matrix_view beta_view = gsl_matrix_view_array((double*)beta_data, p, m);
    gsl_matrix *X_mat = &X_view.matrix;
    gsl_matrix *Y_mat = &Y_view.matrix;
    gsl_matrix *T_mat = &T_view.matrix;
    gsl_matrix *beta_mat = &beta_view.matrix;

    gsl_matrix *Y_hat = NULL;
    gsl_matrix *residuals = NULL;
    double *sigma2 = NULL;
    gsl_matrix *XtX = NULL;
    gsl_matrix *XtX_inv = NULL;
    gsl_permutation *perm_ttest = NULL;
    gsl_matrix *TX = NULL;
    double *T_row_sq_sum = NULL;

    int status = RIDGE_SUCCESS;
    double df = NAN;

    Y_hat = gsl_matrix_alloc(n, m);
    residuals = gsl_matrix_alloc(n, m);
    sigma2 = (double*)malloc(m * sizeof(double));
    if (!Y_hat || !residuals || !sigma2) {
         fprintf(stderr, "ERROR (ttest): Allocation failed for Y_hat/residuals/sigma2.\n"); fflush(stderr);
         status = RIDGE_ERROR_ALLOCATION_FAILED; goto ttest_cleanup;
    }
    memset(sigma2, 0, m * sizeof(double));

    // Calculate Y_hat = X @ beta
    // *** Uses custom parallel dgemm ***
    if (parallel_blocked_dgemm(CblasNoTrans, CblasNoTrans, 1.0, X_mat, beta_mat, 0.0, Y_hat) != GSL_SUCCESS) {
        fprintf(stderr, "ERROR (ttest): parallel_blocked_dgemm for Y_hat failed.\n"); fflush(stderr);
        status = RIDGE_ERROR_DGEMM_FAILED; goto ttest_cleanup;
    }

    // Calculate Residuals = Y - Y_hat
    if (gsl_matrix_memcpy(residuals, Y_mat) != GSL_SUCCESS) {
        fprintf(stderr, "ERROR (ttest): memcpy for residuals failed.\n"); fflush(stderr);
        status = RIDGE_ERROR_MATRIX_OPERATION; goto ttest_cleanup;
    }
    if (gsl_matrix_sub(residuals, Y_hat) != GSL_SUCCESS) {
         fprintf(stderr, "ERROR (ttest): matrix subtraction for residuals failed.\n"); fflush(stderr);
         status = RIDGE_ERROR_MATRIX_OPERATION; goto ttest_cleanup;
    }

    // Calculate Degrees of Freedom (df) and Variance (sigma2)
    if (fabs(lambda) < EPS) { // OLS Case
        df = (double)fmax(1.0, (double)n - (double)p);
        if (n <= p && status == RIDGE_SUCCESS) {
             fprintf(stderr, "WARNING (ttest-OLS): n <= p (%zu <= %zu). DF set to 1. T-test results may be unreliable.\n", n, p); fflush(stderr);
        }

        #pragma omp parallel for if(m > 8) schedule(static)
        for (size_t j = 0; j < m; j++) {
            double sum_sq_resid = 0.0;
            for (size_t i = 0; i < n; i++) {
                double r = gsl_matrix_get(residuals, i, j);
                sum_sq_resid += r * r;
            }
            sigma2[j] = fmax(0.0, sum_sq_resid / df);
        }

        XtX = gsl_matrix_alloc(p, p);
        XtX_inv = gsl_matrix_alloc(p, p);
        perm_ttest = gsl_permutation_alloc(p);
        if (!XtX || !XtX_inv || !perm_ttest) {
            fprintf(stderr, "ERROR (ttest-OLS): Allocation failed for XtX/XtX_inv/perm.\n"); fflush(stderr);
            status = RIDGE_ERROR_ALLOCATION_FAILED; goto ttest_cleanup;
        }

        // Calculate (X'X)
        // *** Uses custom parallel dgemm ***
        if (parallel_blocked_dgemm(CblasTrans, CblasNoTrans, 1.0, X_mat, X_mat, 0.0, XtX) != GSL_SUCCESS) {
             fprintf(stderr, "ERROR (ttest-OLS): parallel_blocked_dgemm for XtX failed.\n"); fflush(stderr);
             status = RIDGE_ERROR_DGEMM_FAILED; goto ttest_cleanup;
        }
        int signum_ttest;
        if (gsl_linalg_LU_decomp(XtX, perm_ttest, &signum_ttest) != GSL_SUCCESS) {
            fprintf(stderr, "ERROR (ttest-OLS): LU decomposition failed.\n"); fflush(stderr);
            status = RIDGE_ERROR_DECOMPOSITION; goto ttest_cleanup;
        }
        if (gsl_linalg_LU_invert(XtX, perm_ttest, XtX_inv) != GSL_SUCCESS) {
             fprintf(stderr, "ERROR (ttest-OLS): LU invert failed.\n"); fflush(stderr);
             status = RIDGE_ERROR_INVERSION; goto ttest_cleanup;
        }

        #pragma omp parallel for if(p*m > MIN_PARALLEL_SIZE) schedule(dynamic) collapse(2)
        for (size_t i = 0; i < p; i++) {
            for (size_t j = 0; j < m; j++) {
                double diag_val = gsl_matrix_get(XtX_inv, i, i);
                if (diag_val < -EPS) {
                    #pragma omp critical (NegDiagWarnOLS)
                    { fprintf(stderr, "WARNING (ttest-OLS): Negative diag(XtX_inv) (%.2e) at %zu. SE set NaN.\n", diag_val, i); fflush(stderr); }
                    se_data[i*m + j] = NAN;
                } else {
                    if (diag_val < 0.0) diag_val = 0.0;
                    double se_var = sigma2[j] * diag_val;
                    se_data[i*m + j] = sqrt(se_var);
                }
            }
        }

    } else { // Ridge Case
        double trace_H = 0.0;
        TX = gsl_matrix_alloc(p, p);
        if (!TX) {
            fprintf(stderr, "ERROR (ttest-Ridge): Allocation failed for TX.\n"); fflush(stderr);
            status = RIDGE_ERROR_ALLOCATION_FAILED; goto ttest_cleanup;
        }
        // Calculate TX = T * X
        // *** Uses custom parallel dgemm ***
        if (parallel_blocked_dgemm(CblasNoTrans, CblasNoTrans, 1.0, T_mat, X_mat, 0.0, TX) != GSL_SUCCESS) {
             fprintf(stderr, "ERROR (ttest-Ridge): parallel_blocked_dgemm for TX failed.\n"); fflush(stderr);
             status = RIDGE_ERROR_DGEMM_FAILED; goto ttest_cleanup;
        }
        for (size_t i = 0; i < p; i++) { trace_H += gsl_matrix_get(TX, i, i); }

        df = (double)n - trace_H;
        if (df <= 0) {
             if (status == RIDGE_SUCCESS) {
                 fprintf(stderr, "WARNING (ttest-Ridge): Eff. DF <= 0 (%.2f, n=%zu, trH=%.2f). Clamped to 1.0.\n", df, n, trace_H); fflush(stderr);
             }
             df = 1.0;
        }

        #pragma omp parallel for if(m > 8) schedule(static)
        for (size_t j = 0; j < m; j++) {
            double sum_sq_resid = 0.0;
            for (size_t i = 0; i < n; i++) {
                double r = gsl_matrix_get(residuals, i, j);
                sum_sq_resid += r * r;
            }
            sigma2[j] = fmax(0.0, sum_sq_resid / df);
        }

        T_row_sq_sum = (double*)malloc(p * sizeof(double));
        if (!T_row_sq_sum) {
            fprintf(stderr, "ERROR (ttest-Ridge): Allocation failed for T_row_sq_sum.\n"); fflush(stderr);
            status = RIDGE_ERROR_ALLOCATION_FAILED; goto ttest_cleanup;
        }
        #pragma omp parallel for if(p > 100) schedule(static)
        for (size_t i = 0; i < p; i++) {
            double row_sum_sq = 0.0;
            const double* T_row_ptr = T_data + i*n;
            for (size_t k = 0; k < n; k++) {
                double T_ik = T_row_ptr[k];
                row_sum_sq += T_ik * T_ik;
            }
            T_row_sq_sum[i] = row_sum_sq;
        }

        #pragma omp parallel for if(p*m > MIN_PARALLEL_SIZE) schedule(dynamic) collapse(2)
        for (size_t i = 0; i < p; i++) {
            for (size_t j = 0; j < m; j++) {
                double se_var = sigma2[j] * T_row_sq_sum[i];
                 if (se_var < 0.0 && se_var > -EPS) se_var = 0.0;
                 else if (se_var < -EPS) {
                     #pragma omp critical (NegSEVarWarnRidge)
                     { fprintf(stderr, "WARNING (ttest-Ridge): Neg SE variance (%.2e) for feat %zu, sample %zu. SE set NaN.\n", se_var, i, j); fflush(stderr); }
                     se_data[i*m + j] = NAN;
                     continue;
                 }
                 se_data[i*m + j] = sqrt(se_var);
            }
        }
    } // End OLS/Ridge blocks

    // Calculate t-statistics and p-values
    if (isnan(df)) {
        if(status == RIDGE_SUCCESS) status = RIDGE_ERROR_TTEST;
        size_t pm_size = p * m;
        #pragma omp parallel for if(pm_size > MIN_PARALLEL_SIZE*10) schedule(static)
        for (size_t k = 0; k < pm_size; k++) { se_data[k]=NAN; zscore_data[k]=NAN; pvalue_data[k]=NAN; }
    } else {
        size_t pm_size = p * m;
        #pragma omp parallel for if(pm_size > MIN_PARALLEL_SIZE) schedule(dynamic)
        for (size_t k = 0; k < pm_size; k++) {
            double beta_val = beta_data[k];
            double se_val = se_data[k];
            double t_stat;
            double p_val = NAN;

            if (isnan(se_val)) { t_stat = NAN; }
            else if (se_val > EPS) { t_stat = beta_val / se_val; }
            else { t_stat = (fabs(beta_val) < EPS) ? 0.0 : copysign(INFINITY, beta_val); }

            zscore_data[k] = t_stat;

            if (isfinite(t_stat) && df > 0) {
                p_val = 2.0 * gsl_cdf_tdist_Q(fabs(t_stat), df);
                p_val = fmax(0.0, fmin(1.0, p_val));
            } else if (isinf(t_stat)) { p_val = 0.0; }
            else { p_val = NAN; }

            pvalue_data[k] = p_val;
        }
    }

    *df_out = df;

ttest_cleanup:
    gsl_matrix_free(Y_hat);
    gsl_matrix_free(residuals);
    free(sigma2);
    gsl_matrix_free(XtX);
    gsl_matrix_free(XtX_inv);
    gsl_permutation_free(perm_ttest);
    gsl_matrix_free(TX);
    free(T_row_sq_sum);

    return status;
}