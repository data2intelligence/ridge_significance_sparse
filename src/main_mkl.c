// mkl/main.c
// Test harness for ridge_mkl C functions with customizable matrix dimensions.
// Modified to profile different OpenMP/MKL thread combinations and
// use the same underlying Y data (generated sparsely, then used densely/sparsified).
// Includes improved sparse data generation.

// Define POSIX source compatibility BEFORE including any headers
#define _POSIX_C_SOURCE 200809L // Or _GNU_SOURCE for Linux often works

#include <stdio.h>
#include <stdlib.h>
#include <string.h> // For memset, strcmp, strdup, strtok_r
#include <math.h>    // For NAN, fabs, isnan, labs
#include <time.h>    // For seeding random number generator, time()
#include <ctype.h>   // For isdigit

#ifdef _OPENMP
#include <omp.h>     // For OpenMP thread control and timing (omp_*)
#else
// Define dummy OpenMP functions if not compiled with OpenMP
#define omp_get_max_threads() 1
#define omp_set_num_threads(n) (void)n
#define omp_get_wtime() ((double)clock() / CLOCKS_PER_SEC) // Fallback timer
#define omp_get_thread_num() 0
#endif

// Include the MKL backend header
#include "ridge_mkl.h"

// --- Configuration Defaults ---
#define DEFAULT_GENES 2000
#define DEFAULT_FEATURES 300
#define DEFAULT_SAMPLES 25
#define DEFAULT_PERM 10
#define DEFAULT_SPARSITY 0.05 // Default to sparse generation
#define DEFAULT_LAMBDA 10.0
#define DEFAULT_MKL_THREADS -1 // Use MKL default
#define MAX_THREAD_SETTINGS 32 // Max number of thread counts per type

// --- Helper Function Prototypes ---
void print_usage(const char* program_name);
void print_results_summary(const char* prefix, int status, size_t p, size_t m,
                           const double* beta, const double* se,
                           const double* zscore, const double* pvalue, double df);
void generate_chunked_random_data(double* data, size_t n_rows, size_t n_cols,
                                 double min_val, double max_val);
// Updated prototype name to reflect improvement
void generate_sparse_fill_dense_oversample(double* dense_data, size_t n_rows, size_t n_cols, double density,
                                           double min_val, double max_val, size_t* nnz_count);
int convert_dense_to_csr(const double* dense_data, size_t n_rows, size_t n_cols,
                         double** vals_out, int** cols_out, int** rows_out, size_t* nnz_out);
void print_memory_info(size_t n, size_t p, size_t m, double density);
int parse_thread_list(const char* arg, int* list, int max_len);


// --- Main Function ---
int main(int argc, char* argv[]) {
    printf("Starting MKL C backend test with custom matrix dimensions...\n");
    printf("Profiling different OMP/MKL thread combinations.\n");
    printf("Using unified Y data (generate sparse -> use dense & sparse).\n");

    // Default settings
    size_t n_genes = DEFAULT_GENES;
    size_t n_features = DEFAULT_FEATURES;
    size_t n_samples = DEFAULT_SAMPLES;
    int perm_count = DEFAULT_PERM;
    double sparsity = DEFAULT_SPARSITY;
    double lambda_val = DEFAULT_LAMBDA;
    int run_perm_test = 1; // 1 = Permutation (default)
    int run_ttest = 0;     // 0 = T-Test (must be explicitly requested for dense)
    int run_dense = 1;     // Run dense tests by default
    int run_sparse = 1;    // Run sparse tests by default

    // Thread settings storage
    int omp_thread_list[MAX_THREAD_SETTINGS];
    int mkl_thread_list[MAX_THREAD_SETTINGS];
    int num_omp_settings = 0;
    int num_mkl_settings = 0;
    int got_omp_arg = 0;
    int got_mkl_arg = 0;

    // --- Initialize variables used in cleanup path ---
    double* X_data = NULL;
    double* Y_data = NULL;
    double* beta_d = NULL, *se_d = NULL, *z_d = NULL, *pv_d = NULL; // Dense results
    double* beta_s = NULL, *se_s = NULL, *z_s = NULL, *pv_s = NULL; // Sparse results
    double* Y_sparse_vals = NULL;
    int*    Y_sparse_col = NULL;
    int*    Y_sparse_row = NULL;
    int overall_status = 1; // Default to error until success
    int csr_status = -1;     // Status for CSR conversion

    // Process command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--genes") == 0 && i+1 < argc) {
            n_genes = atol(argv[i+1]); i++;
        } else if (strcmp(argv[i], "--features") == 0 && i+1 < argc) {
            n_features = atol(argv[i+1]); i++;
        } else if (strcmp(argv[i], "--samples") == 0 && i+1 < argc) {
            n_samples = atol(argv[i+1]); i++;
        } else if (strcmp(argv[i], "--perm") == 0 && i+1 < argc) {
            perm_count = atoi(argv[i+1]); i++;
            if (perm_count > 0) {
                 run_perm_test = 1;
                 run_ttest = 0; // Permutation overrides ttest request
            } else {
                 fprintf(stderr,"Warning: --perm requires positive value, using default %d.\n", DEFAULT_PERM);
                 perm_count = DEFAULT_PERM;
                 run_perm_test = 1;
                 run_ttest = 0;
            }
        } else if (strcmp(argv[i], "--sparse") == 0 && i+1 < argc) {
            sparsity = atof(argv[i+1]);
            if (sparsity < 0.0 || sparsity > 1.0) {
                fprintf(stderr, "Warning: Sparsity must be between 0.0 and 1.0. Using default (%.2f).\n", DEFAULT_SPARSITY);
                sparsity = DEFAULT_SPARSITY;
            }
             i++;
        } else if (strcmp(argv[i], "--lambda") == 0 && i+1 < argc) {
            lambda_val = atof(argv[i+1]);
            if (lambda_val < 0.0) {
                fprintf(stderr, "Warning: Lambda must be non-negative. Using default (%.1f).\n", DEFAULT_LAMBDA);
                lambda_val = DEFAULT_LAMBDA;
            }
             i++;
        } else if (strcmp(argv[i], "--ttest") == 0) {
            if (run_perm_test && perm_count > 0) {
                 fprintf(stderr,"Warning: --ttest ignored when --perm is set > 0.\n");
            } else {
                run_ttest = 1; // Request T-test instead of permutation
                run_perm_test = 0;
                perm_count = 0; // T-test uses nrand=0 signal in dense function
            }
        } else if (strcmp(argv[i], "--omp-threads") == 0 && i+1 < argc) {
            num_omp_settings = parse_thread_list(argv[i+1], omp_thread_list, MAX_THREAD_SETTINGS);
            if (num_omp_settings <= 0) {
                fprintf(stderr, "Error: Invalid --omp-threads list '%s'. Use comma-separated numbers (e.g., 1,2,4).\n", argv[i+1]);
                return 1;
            }
            got_omp_arg = 1;
            i++;
        } else if (strcmp(argv[i], "--mkl-threads") == 0 && i+1 < argc) { // Renamed from --threads
            num_mkl_settings = parse_thread_list(argv[i+1], mkl_thread_list, MAX_THREAD_SETTINGS);
             if (num_mkl_settings <= 0) {
                fprintf(stderr, "Error: Invalid --mkl-threads list '%s'. Use comma-separated numbers (e.g., 1,2,4).\n", argv[i+1]);
                return 1;
            }
            got_mkl_arg = 1;
            i++;
        } else if (strcmp(argv[i], "--no-dense") == 0) {
             run_dense = 0;
        } else if (strcmp(argv[i], "--no-sparse") == 0) {
             run_sparse = 0;
        } else if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

     // Validate test combination
    if (!run_dense && !run_sparse) {
         fprintf(stderr, "Error: Cannot use both --no-dense and --no-sparse. No tests would run.\n");
         return 1;
    }
     if (run_ttest && !run_dense) {
         fprintf(stderr, "Warning: --ttest requires dense run. Ignoring --ttest.\n");
         run_ttest = 0;
         run_perm_test = 1; // Fallback to permutation if ttest was the only mode
         perm_count = DEFAULT_PERM;
     }
     if (run_ttest && run_sparse) {
         fprintf(stderr, "Info: T-test is only performed for the dense case.\n");
     }
     if (!run_perm_test && !run_ttest) { // Should not happen with current logic, but safe fallback
         fprintf(stderr, "Info: No test type specified, defaulting to permutation test.\n");
         run_perm_test = 1;
         perm_count = DEFAULT_PERM;
     }

    // Validate dimensions
    if (n_genes == 0 || n_features == 0 || n_samples == 0) {
        fprintf(stderr, "Error: Matrix dimensions must be positive.\n");
        print_usage(argv[0]);
        return 1;
    }

    // --- Set Default Thread Lists if not provided ---
    if (!got_omp_arg) {
        #ifdef _OPENMP
        int max_omp = omp_get_max_threads();
        omp_thread_list[0] = 1;
        num_omp_settings = 1;
        if (max_omp > 1 && max_omp <= MAX_THREAD_SETTINGS) { // Only add max if different and fits
            int found = 0;
            for(int k=0; k<num_omp_settings; ++k) if(omp_thread_list[k] == max_omp) found = 1;
            if (!found) {
               omp_thread_list[num_omp_settings++] = max_omp;
            }
        }
        printf("INFO: No --omp-threads specified, defaulting to: ");
        for(int i=0; i<num_omp_settings; ++i) printf("%d%s", omp_thread_list[i], (i == num_omp_settings - 1) ? "" : ",");
        printf("\n");
        #else
        omp_thread_list[0] = 1;
        num_omp_settings = 1;
        printf("INFO: OpenMP not enabled. Using 1 OMP thread.\n");
        #endif
    }
    if (!got_mkl_arg) {
        int current_mkl_threads = ridge_mkl_get_threads(); // Get MKL default
        mkl_thread_list[0] = current_mkl_threads;
        num_mkl_settings = 1;
        printf("INFO: No --mkl-threads specified, defaulting to MKL default: %d\n", current_mkl_threads);
    }

    // --- Display Configuration ---
    printf("\n--- Configuration ---\n");
    printf("Test(s) Run:    %s%s%s\n",
           (run_dense && run_perm_test) ? "DensePerm " : "",
           (run_dense && run_ttest) ? "DenseTTest " : "",
           run_sparse ? "SparsePerm" : "");
    if (run_perm_test) printf("Permutations:   %d\n", perm_count);
    printf("Dimensions:     X(%zu x %zu), Y(%zu x %zu)\n", n_genes, n_features, n_genes, n_samples);
    printf("Y Density:      %.2f%%\n", sparsity * 100.0);
    printf("Lambda:         %.2f\n", lambda_val);
    printf("OMP Threads:    [");
    for(int i=0; i<num_omp_settings; ++i) printf("%d%s", omp_thread_list[i], (i == num_omp_settings - 1) ? "" : ", ");
    printf("]\n");
    printf("MKL Threads:    [");
    for(int i=0; i<num_mkl_settings; ++i) printf("%d%s", mkl_thread_list[i], (i == num_mkl_settings - 1) ? "" : ", ");
    printf("]\n");
    #ifndef _OPENMP
    printf("NOTE: Compiled without OpenMP support.\n");
    #endif
    #ifndef HAVE_MKL // Should be defined by Makefile
    printf("CRITICAL WARNING: Compiled without MKL support (HAVE_MKL not defined).\n");
    printf("                 MKL thread settings will have no effect, results invalid.\n");
    #endif
    printf("---------------------\n");

    print_memory_info(n_genes, n_features, n_samples, sparsity);

    // --- Allocate memory for test data (once) ---
    printf("\nAllocating memory for X, Y, and Results...\n");
    fflush(stdout);
    X_data = (double*)malloc(n_genes * n_features * sizeof(double));
    Y_data = (double*)malloc(n_genes * n_samples * sizeof(double)); // Always dense storage

    if (!X_data || !Y_data ) {
        fprintf(stderr, "FATAL: Memory allocation failed for input data.\n");
        goto cleanup; // Jump to cleanup
    }

    size_t pm_size = n_features * n_samples;

    if (run_dense) {
        beta_d = (double*)malloc(pm_size * sizeof(double));
        se_d   = (double*)malloc(pm_size * sizeof(double));
        z_d    = (double*)malloc(pm_size * sizeof(double));
        pv_d   = (double*)malloc(pm_size * sizeof(double));
        if (!beta_d || !se_d || !z_d || !pv_d) {
             fprintf(stderr, "FATAL: Memory allocation failed for dense results.\n"); goto cleanup;
        }
    }
     if (run_sparse) {
        beta_s = (double*)malloc(pm_size * sizeof(double));
        se_s   = (double*)malloc(pm_size * sizeof(double));
        z_s    = (double*)malloc(pm_size * sizeof(double));
        pv_s   = (double*)malloc(pm_size * sizeof(double));
         if (!beta_s || !se_s || !z_s || !pv_s) {
             fprintf(stderr, "FATAL: Memory allocation failed for sparse results.\n"); goto cleanup;
        }
    }

    printf("Memory allocation successful.\n");

    // --- Generate Data (once) ---
    printf("Generating random data (this may take time for large matrices)...\n");
    fflush(stdout);
    // Use time-based seed for less predictable random data across runs
    // Keep srand(12345) if strict reproducibility is needed for debugging
    srand((unsigned int)time(NULL));

    printf("Generating X data...\n");
    generate_chunked_random_data(X_data, n_genes, n_features, 0.0, 3.0);

    printf("Generating Y data (density %.2f%%)...\n", sparsity * 100.0);
    size_t y_nnz_count = 0;
    if (sparsity < 1.0) {
        // Use the improved sparse generation function
        generate_sparse_fill_dense_oversample(Y_data, n_genes, n_samples, sparsity, 0.0, 10.0, &y_nnz_count);
        // The function now prints the actual vs target nnz counts
        // printf("Generated Y with %zu non-zero elements (%.2f%% density)\n",
        //       y_nnz_count, 100.0 * y_nnz_count / (n_genes * n_samples));
    } else {
        generate_chunked_random_data(Y_data, n_genes, n_samples, 0.0, 10.0);
        y_nnz_count = n_genes * n_samples; // All elements are potentially non-zero
        printf("Generated dense Y.\n");
    }
    printf("Data generation complete.\n");

    // --- Convert Y to CSR if sparse test is needed ---
    size_t  actual_nnz = 0;
    if (run_sparse) {
        printf("Converting dense Y to CSR format...\n"); fflush(stdout);
        double convert_start = omp_get_wtime();
        csr_status = convert_dense_to_csr(Y_data, n_genes, n_samples,
                                          &Y_sparse_vals, &Y_sparse_col, &Y_sparse_row,
                                          &actual_nnz);
        double convert_end = omp_get_wtime();
        if (csr_status != 0) {
            fprintf(stderr, "FATAL: Failed to convert Y data to CSR format.\n");
            goto cleanup;
        }
        // Check if CSR nnz matches the count from generation (should be very close now)
        if (sparsity < 1.0 && labs((long)actual_nnz - (long)y_nnz_count) != 0) {
             printf("INFO: CSR conversion resulted in %zu non-zeros (generation count was %zu).\n", actual_nnz, y_nnz_count);
        } else {
             printf("CSR conversion successful (%zu non-zeros). Time: %.2f s\n", actual_nnz, convert_end - convert_start);
        }
        fflush(stdout);
    }


    // --- Prepare Parameters for MKL calls ---
    int n_int = (int)n_genes;
    int p_int = (int)n_features;
    int m_int = (int)n_samples;
    int n_rand_perm_int = run_perm_test ? perm_count : 0; // Perm count for perm test
    int n_rand_ttest_int = 0;                            // 0 signal for t-test
    double df_out = NAN;
    // Initialize status codes to a known 'not run' state different from success (0)
    int status_d = -10, status_s = -10;


    // --- Profiling Loop ---
    printf("\n--- Starting Profiling Runs ---\n");
    printf("%-12s | %-12s | %-12s | %-15s | %-12s | %-15s | %s\n",
           "OMP Threads", "MKL Threads", "Dense Status", "Dense Time (s)", "Sparse Status", "Sparse Time (s)", "Result Summary");
    printf("--------------------------------------------------------------------------------------------------------------------\n");
    fflush(stdout);

    int initial_mkl_threads = ridge_mkl_get_threads(); // Store initial MKL setting

    for (int i = 0; i < num_omp_settings; ++i) {
        int current_omp_threads = omp_thread_list[i];

        // Skip if OpenMP not enabled and threads > 1 requested
        #ifndef _OPENMP
        if (current_omp_threads > 1) {
             printf("%-10d | %-10s | %-12s | %-15s | %-12s | %-15s | %s\n",
                   current_omp_threads, "N/A", "N/A", "N/A", "N/A", "N/A", "SKIP: OpenMP not enabled");
            continue;
        }
        #endif

        for (int j = 0; j < num_mkl_settings; ++j) {
            int requested_mkl_threads = mkl_thread_list[j];

            // --- Set Threads ---
            #ifdef _OPENMP
            omp_set_num_threads(current_omp_threads);
            #endif
            ridge_mkl_set_threads(requested_mkl_threads);
            int actual_mkl_threads = ridge_mkl_get_threads(); // Get the number MKL is actually using


            // --- Run and Time ---
            double elapsed_dense = -1.0, elapsed_sparse = -1.0;
            double start_time;

            // Reset status for this run
            status_d = -1; // Mark as skipped initially
            status_s = -1; // Mark as skipped initially

            // --- Dense Test ---
            if (run_dense) {
                df_out = NAN; // Reset df for each run
                int nrand_dense = run_ttest ? n_rand_ttest_int : n_rand_perm_int;

                start_time = omp_get_wtime();
                // Use initialized result buffers beta_d, se_d etc.
                status_d = ridge_mkl_dense(X_data, Y_data, // Use the unified Y_data
                                          n_int, p_int, m_int,
                                          lambda_val, nrand_dense,
                                          beta_d, se_d, z_d, pv_d, &df_out);
                elapsed_dense = omp_get_wtime() - start_time;
            }

            // --- Sparse Test ---
             if (run_sparse) {
                 // Sparse only runs permutation test
                 if (run_perm_test) {
                     // Ensure CSR conversion succeeded before running
                     if (csr_status == 0) {
                         start_time = omp_get_wtime();
                         // Use initialized result buffers beta_s, se_s etc.
                         status_s = ridge_mkl_sparse(X_data, n_int, p_int,
                                                     Y_sparse_vals, Y_sparse_col, Y_sparse_row,
                                                     m_int, (int)actual_nnz, lambda_val, n_rand_perm_int,
                                                     beta_s, se_s, z_s, pv_s);
                         elapsed_sparse = omp_get_wtime() - start_time;
                     } else {
                         status_s = -3; // Indicate CSR conversion failed
                     }
                 } else {
                      status_s = -2; // Indicate t-test requested but not supported/run for sparse
                 }
            }


            // --- Print Results Row ---
             printf("%-10d | %-10d | ", current_omp_threads, actual_mkl_threads);
             // Dense Results
             if (status_d == -1) printf("%-12s | %-15s | ", "SKIP", "N/A");
             else printf("%-12d | %-15.4f | ", status_d, elapsed_dense);
             // Sparse Results
             if (status_s == -1) printf("%-12s | %-15s | ", "SKIP", "N/A");
             else if (status_s == -2) printf("%-12s | %-15s | ", "N/A(T)", "N/A");
             else if (status_s == -3) printf("%-12s | %-15s | ", "ERR(CSR)", "N/A");
             else printf("%-12d | %-15.4f | ", status_s, elapsed_sparse);

             // Result Summary (show first element of dense perm/ttest and sparse perm)
             if (status_d == 0) {
                 print_results_summary(run_ttest ? "DT" : "DP", status_d, p_int, m_int, beta_d, se_d, z_d, pv_d, df_out);
             }
             if (status_s == 0) {
                  if (status_d == 0) printf(" || "); // Separator if dense also ran
                  print_results_summary("SP", status_s, p_int, m_int, beta_s, se_s, z_s, pv_s, NAN);
             }
             if (status_d != 0 && status_s != 0 && status_s > -2) { // Print fail unless skipped/not applicable
                 printf("Failed");
             }
             if (status_d == -1 && status_s == -1) {
                 printf("Skipped"); // If both skipped
             }
             printf("\n");
             fflush(stdout);

        } // End MKL threads loop
    } // End OMP threads loop

    printf("--------------------------------------------------------------------------------------------------------------------\n");

    // Determine overall status for return code
    // Success (0) only if all requested tests ran and succeeded.
    overall_status = 0; // Assume success initially
    if (run_dense && status_d != 0) overall_status = status_d; // Dense failed
    if (run_sparse && status_s != 0 && status_s > -2) { // Sparse failed (ignore skip/N/A codes)
       if (overall_status == 0) overall_status = status_s; // Report sparse failure if dense was ok
       else overall_status = -99; // Indicate multiple failures
    }


    // Restore initial MKL thread setting
    ridge_mkl_set_threads(initial_mkl_threads);
    printf("Restored MKL threads to %d.\n", initial_mkl_threads);

cleanup: // Unified cleanup label

    // --- Cleanup ---
    printf("\nCleaning up allocated memory...\n");
    free(X_data);
    free(Y_data);
    // Dense results
    free(beta_d); free(se_d); free(z_d); free(pv_d);
    // Sparse results
    free(beta_s); free(se_s); free(z_s); free(pv_s);
    // CSR arrays
    free(Y_sparse_vals); free(Y_sparse_col); free(Y_sparse_row);


    printf("\nMKL C backend profiling finished. Overall Status: %d\n", overall_status);
    return overall_status; // Return 0 on success, non-zero on error
}


// --- Helper Function Implementations ---

// Parses a comma-separated list of integers
int parse_thread_list(const char* arg, int* list, int max_len) {
    int count = 0;
    char* token;
    char* str = strdup(arg); // Duplicate string as strtok_r modifies it
    if (!str) return -1;     // Allocation failed

    char* rest = str;
    while ((token = strtok_r(rest, ",", &rest)) != NULL && count < max_len) {
        // Basic validation: check if it's a positive integer
        int valid = 1;
        char* endptr;
        long val_l = strtol(token, &endptr, 10);

        // Check for conversion errors, non-digit characters left, range
        if (endptr == token || *endptr != '\0' || val_l <= 0 || val_l > 10240) {
             valid = 0;
        }

        if (valid) {
            list[count++] = (int)val_l;
        } else {
            fprintf(stderr, "Warning: Skipping invalid thread count '%s' in list.\n", token);
        }
    }

    free(str); // Free the duplicated string
    return count;
}


void print_usage(const char* program_name) {
    printf("\nUsage: %s [options]\n", program_name);
    printf("Options:\n");
    printf("  --genes N         Number of genes/observations [default: %d]\n", DEFAULT_GENES);
    printf("  --features N      Number of features [default: %d]\n", DEFAULT_FEATURES);
    printf("  --samples N       Number of samples [default: %d]\n", DEFAULT_SAMPLES);
    printf("  --perm N          Number of permutations [default: %d]. Runs Permutation Test (dense/sparse).\n", DEFAULT_PERM);
    printf("  --ttest           Run T-test instead of Permutation Test (dense case only).\n");
    printf("  --sparse VAL      Sparsity level (0.0-1.0) for generating Y matrix [default: %.2f]\n", DEFAULT_SPARSITY);
    printf("  --lambda VAL      Ridge regularization parameter [default: %.1f]\n", DEFAULT_LAMBDA);
    printf("  --omp-threads L   Comma-separated list of OpenMP threads for C loops (e.g., 1,2,4,8) [default: 1,MAX]\n");
    printf("  --mkl-threads L   Comma-separated list of MKL threads (e.g., 1,2,4) [default: MKL_DEFAULT]\n");
    printf("  --no-dense        Skip dense tests (Permutation and T-Test).\n");
    printf("  --no-sparse       Skip sparse permutation test.\n");
    printf("  --help            Show this help message\n");
    printf("\nNotes:\n");
    printf("  - Y matrix is generated once (potentially sparsely), then used for both dense and sparse tests.\n");
    printf("  - For sparse tests, the dense Y is converted to CSR format.\n");
    printf("  - T-test (--ttest) only applies to the dense run.\n");
    printf("\nExamples:\n");
    printf("  # Profile dense/sparse perm test (Y=5%% sparse) with 1, 8 OMP threads and MKL default threads\n");
    printf("  %s --genes 4000 --features 500 --sparse 0.05 --perm 50 --omp-threads 1,8\n\n", program_name);
    printf("  # Profile dense t-test with 4 OMP threads and 1, 2, 4 MKL threads\n");
    printf("  %s --genes 2000 --features 300 --ttest --omp-threads 4 --mkl-threads 1,2,4 --no-sparse\n\n", program_name);
}


// Helper to generate random data (parallelized)
void generate_chunked_random_data(double* data, size_t n_rows, size_t n_cols,
                                 double min_val, double max_val) {
    double range = max_val - min_val;
    size_t total_elements = n_rows * n_cols;
    printf("    Generating %.2f M dense elements... ", (double)total_elements / 1.0e6);
    fflush(stdout);
    unsigned int base_seed = (unsigned int)time(NULL); // Seed based on time

    #pragma omp parallel
    {
       // Use omp_get_wtime() for a potentially higher resolution seed component per thread
       unsigned int thread_seed = base_seed + omp_get_thread_num() * 101 + (unsigned int)(omp_get_wtime()*1000);
       #pragma omp for schedule(static)
       for (size_t k = 0; k < total_elements; ++k) {
            // Use rand_r for thread safety
            data[k] = min_val + ((double)rand_r(&thread_seed) / RAND_MAX) * range;
       }
    }
    printf("Done.\n");
}

// Generates a sparse matrix by filling a dense array (parallelized)
// Improved version: Slightly oversamples placements to get closer to target density.
void generate_sparse_fill_dense_oversample(double* dense_data, size_t n_rows, size_t n_cols, double density,
                                           double min_val, double max_val, size_t* nnz_count) {
    double range = max_val - min_val;
    size_t total_elements = n_rows * n_cols;
    // Ensure target_nonzero_elements doesn't exceed total_elements
    size_t target_nonzero_elements = (size_t)(total_elements * density);
    if (target_nonzero_elements > total_elements) target_nonzero_elements = total_elements;
    if (target_nonzero_elements == 0 && density > 0 && total_elements > 0) target_nonzero_elements = 1; // Ensure at least one if density > 0

    // --- Oversampling Strategy ---
    // Heuristic: Increase attempts more aggressively for higher densities where collisions are more likely.
    // Simple linear scaling: 1.0 (low density) up to ~1.2 (high density)
    double oversample_factor = 1.0 + (0.2 * density);
    // Alternative (slightly more complex): 1.0 / (1.0 - density * 0.2) ? // Could get too large

    // Prevent excessive oversampling, especially if density is 1.0
    if (oversample_factor > 1.5) oversample_factor = 1.5; // Cap factor
    if (density >= 1.0) oversample_factor = 1.0; // No oversampling needed for dense
    if (target_nonzero_elements <= 1) oversample_factor = 1.0; // No oversampling for 0 or 1 element

    size_t attempts = (size_t)(target_nonzero_elements * oversample_factor);
    // Ensure at least one attempt if target > 0, and don't exceed total elements * reasonable factor
    if (attempts == 0 && target_nonzero_elements > 0) attempts = 1;
    if (attempts > total_elements * 2) attempts = total_elements * 2; // Safety cap

    printf("    Generating sparse fill (target %.2f%% density, %zu nnz, attempting %zu placements)... ",
           density * 100.0, target_nonzero_elements, attempts);
    fflush(stdout);

    // 1. Initialize to zero (parallel)
    #pragma omp parallel for schedule(static)
    for(size_t k=0; k < total_elements; ++k) {
        dense_data[k] = 0.0;
    }

    // 2. Randomly place non-zero elements (parallel) - with oversampling
    // Use time-based seed combined with thread ID for better randomness across threads/runs
    unsigned int base_seed = (unsigned int)time(NULL) + 1;

    #pragma omp parallel
    {
       unsigned int thread_seed = base_seed + omp_get_thread_num() * 101 + (unsigned int)(omp_get_wtime()*1000);
       #pragma omp for schedule(static) // Static assignment of attempts to threads
       for (size_t k = 0; k < attempts; ++k) { // Loop 'attempts' times
           // Generate random index (ensure it's within bounds)
           size_t idx = (size_t)(((double)rand_r(&thread_seed) / RAND_MAX) * total_elements);
           idx = (idx >= total_elements) ? total_elements - 1 : idx; // Clamp to max index

           // Generate value
           double val = min_val + ((double)rand_r(&thread_seed) / RAND_MAX) * range;
           // Avoid exact zero if the range [min_val, max_val] includes zero, as this breaks sparsity count
           if (val == 0.0 && min_val <= 0.0 && max_val >= 0.0) {
               // Assign small non-zero value randomly positive or negative
               val = (rand_r(&thread_seed) % 2 == 0) ? 1e-9 : -1e-9;
           }
           // Place the value - overwrites are expected and part of this method
           dense_data[idx] = val;
       }
    }


    // 3. Count actual non-zeros (parallel) - unchanged
    size_t actual_nnz = 0;
    #pragma omp parallel for reduction(+:actual_nnz) schedule(static)
    for(size_t k=0; k<total_elements; ++k) {
        if (dense_data[k] != 0.0) {
            actual_nnz++;
        }
    }
    *nnz_count = actual_nnz;

     // Report difference from target
     long long diff = (long long)actual_nnz - (long long)target_nonzero_elements;
     double diff_percent = (target_nonzero_elements > 0) ? (100.0 * (double)diff / target_nonzero_elements) : 0.0;
     printf("Done (actual %zu nnz, target %zu, diff %+lld [%+.2f%%]).\n",
            actual_nnz, target_nonzero_elements, diff, diff_percent);
}

// Converts a dense matrix (row-major) to CSR format (0-based indexing)
int convert_dense_to_csr(const double* dense_data, size_t n_rows, size_t n_cols,
                         double** vals_out, int** cols_out, int** rows_out, size_t* nnz_out)
{
    *vals_out = NULL;
    *cols_out = NULL;
    *rows_out = NULL;
    *nnz_out = 0;

    // First pass: Count non-zeros (can be parallelized)
    size_t nnz = 0;
    #pragma omp parallel for reduction(+:nnz) schedule(static)
    for (size_t i = 0; i < n_rows * n_cols; ++i) {
        if (dense_data[i] != 0.0) {
            nnz++;
        }
    }
    *nnz_out = nnz;

    // Allocate CSR arrays
    // Use calloc to initialize row_ptr easily (though we overwrite later)
    *vals_out = (double*)malloc(nnz * sizeof(double));
    *cols_out = (int*)malloc(nnz * sizeof(int));
    *rows_out = (int*)calloc((n_rows + 1), sizeof(int)); // calloc initializes to 0

    if (!(*vals_out) || !(*cols_out) || !(*rows_out)) {
        free(*vals_out); free(*cols_out); free(*rows_out); // Free partially allocated
        *vals_out = NULL; *cols_out = NULL; *rows_out = NULL;
        *nnz_out = 0;
        fprintf(stderr, "ERROR: CSR allocation failed (nnz=%zu)\n", nnz);
        return 1; // Allocation failure
    }

    // Second pass: Populate CSR arrays (This part is inherently sequential per row pointer update)
    size_t current_nnz = 0;
    // rows_out[0] is already 0 from calloc
    for (size_t i = 0; i < n_rows; ++i) {
        for (size_t j = 0; j < n_cols; ++j) {
            double val = dense_data[i * n_cols + j];
            if (val != 0.0) {
                // Bounds check for safety, although nnz should be correct
                if (current_nnz >= nnz) {
                    fprintf(stderr,"ERROR: CSR conversion bounds error at row %zu, col %zu (current_nnz=%zu, total_nnz=%zu).\n",
                            i, j, current_nnz, nnz);
                     free(*vals_out); free(*cols_out); free(*rows_out);
                     return 2; // Logic error
                }
                (*vals_out)[current_nnz] = val;
                (*cols_out)[current_nnz] = (int)j; // Cast size_t to int
                current_nnz++;
            }
        }
        (*rows_out)[i + 1] = (int)current_nnz; // Store count at end of row i
    }

     // Final check
    if (current_nnz != nnz) {
         fprintf(stderr,"ERROR: CSR final nnz count mismatch (%zu vs %zu)\n", current_nnz, nnz);
         free(*vals_out); free(*cols_out); free(*rows_out);
         return 3; // Mismatch error
    }


    return 0; // Success
}


// Print memory usage information
void print_memory_info(size_t n, size_t p, size_t m, double density) {
    double X_size_gb = ((double)n * p * sizeof(double)) / (1024.0 * 1024.0 * 1024.0);
    double Y_dense_size_gb = ((double)n * m * sizeof(double)) / (1024.0 * 1024.0 * 1024.0);
    // CSR size estimate: nnz * (double + int) + (n+1) * int
    size_t est_nnz = (size_t)(n * m * density);
    if (est_nnz == 0 && density > 0.0) est_nnz = 1; // Avoid 0 nnz if density > 0
    double Y_sparse_csr_gb = ( (double)est_nnz * (sizeof(double) + sizeof(int)) + (double)(n+1) * sizeof(int) ) / (1024.0*1024.0*1024.0);
    // Output buffers (one set dense, one set sparse) - Assuming both might run
    double output_buffers_gb = ((double)p * m * 4 * sizeof(double) * 2) / (1024.0 * 1024.0 * 1024.0);
    // MKL intermediates are hard to estimate precisely, depends heavily on algorithm paths
    // Let's use a rough estimate similar to GSL's temp calculation as a proxy
    double temp_mkl_gb = ( ((double)p*p*2) + ((double)p*n) + ((double)p*m*2) ) * sizeof(double) / (1024.0*1024.0*1024.0); // Rough guess: XtX, T, beta_perm etc.
    double total_gb = X_size_gb + Y_dense_size_gb + Y_sparse_csr_gb + output_buffers_gb + temp_mkl_gb;

    printf("\n--- Memory Estimate (Rough) ---\n");
    printf("  Input X:        ~%.3f GB\n", X_size_gb);
    printf("  Input Y (dense):~%.3f GB\n", Y_dense_size_gb);
    printf("  Input Y (CSR):  ~%.3f GB (at %.1f%% density)\n", Y_sparse_csr_gb, density * 100.0);
    printf("  Output Buffers: ~%.3f GB (Dense+Sparse)\n", output_buffers_gb);
    printf("  Est. Temp MKL:  ~%.3f GB\n", temp_mkl_gb);
    printf("  Estimated Total:  ~%.3f GB\n", total_gb);
    printf("-----------------------------\n");

    if (total_gb > 8.0) { // Adjust threshold as needed
        printf("WARNING: Estimated memory usage is high (>8GB). Ensure sufficient RAM.\n");
    }
}

// Print results summary concisely for profiling table
void print_results_summary(const char* prefix, int status, size_t p, size_t m,
                           const double* beta, const double* se,
                           const double* zscore, const double* pvalue, double df)
{
    if (status != 0) {
        printf("%s:Fail(%d)", prefix, status);
        return;
    }
    if (!beta || !se || !zscore || !pvalue) {
        printf("%s:Err(NULL)", prefix);
        return;
    }

    // Find first valid index (non-NaN) if possible
    size_t idx = 0;
    size_t pm = p*m;
    // Check bounds before accessing elements
    if (pm == 0) {
         printf("%s:Empty", prefix);
         return;
    }
    while (idx < pm && (isnan(beta[idx]) || isnan(se[idx]) || isnan(zscore[idx]) || isnan(pvalue[idx]))) {
        idx++;
    }
    if (idx >= pm) idx = 0; // Fallback to 0 if all are NaN or empty

    printf("%s:[%.2e,%.2e,%.2e,%.2e%s]", prefix,
        beta[idx], se[idx], zscore[idx], pvalue[idx],
        (isnan(df) ? "" : ",df") // Don't print df value itself to save space
        );

}