// gsl/main.c
// Test harness for ridge_gsl C functions with control over matrix dimensions
// Modified to profile different OpenMP/BLAS thread combinations.
// Includes improved sparse data generation.

// Define POSIX source compatibility BEFORE including any headers
#define _POSIX_C_SOURCE 200809L // Or _GNU_SOURCE for Linux often works

#include <stdio.h>
#include <stdlib.h>
#include <string.h>  // For memset, strcmp, strdup, strtok_r
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

// Include the GSL backend header
#include "ridge_gsl.h"

// --- Configuration Defaults ---
#define DEFAULT_GENES 2000
#define DEFAULT_FEATURES 300
#define DEFAULT_SAMPLES 25
#define DEFAULT_PERM 10
#define DEFAULT_SPARSITY 1.0 // Default to dense generation for GSL tests
#define DEFAULT_LAMBDA 10.0
#define MAX_THREAD_SETTINGS 32 // Max number of thread counts per type

// --- Helper Function Prototypes ---
void print_usage(const char* program_name);
void print_results_summary(int status, size_t p, size_t m,
                           const double* beta, const double* se,
                           const double* zscore, const double* pvalue, double df);
void generate_chunked_random_data(double* data, size_t n_rows, size_t n_cols,
                                 double min_val, double max_val);
// Updated prototype name
void generate_sparse_fill_dense_oversample(double* dense_data, size_t n_rows, size_t n_cols, double density,
                                           double min_val, double max_val, size_t* nnz_count);
void print_memory_info(size_t n, size_t p, size_t m);
int parse_thread_list(const char* arg, int* list, int max_len);


// --- Main Function ---
int main(int argc, char* argv[]) {
    printf("Starting GSL C backend test with custom matrix dimensions...\n");
    printf("Profiling different thread combinations.\n");

    // Default settings
    size_t n_genes = DEFAULT_GENES;
    size_t n_features = DEFAULT_FEATURES;
    size_t n_samples = DEFAULT_SAMPLES;
    int perm_count = DEFAULT_PERM;
    double sparsity = DEFAULT_SPARSITY;
    double lambda_val = DEFAULT_LAMBDA;
    int run_test_type = 1; // 1 = Permutation, 0 = T-Test

    // Thread settings storage
    int omp_thread_list[MAX_THREAD_SETTINGS];
    int blas_thread_list[MAX_THREAD_SETTINGS];
    int num_omp_settings = 0;
    int num_blas_settings = 0;
    int got_omp_arg = 0;
    int got_blas_arg = 0;

    // Initialize pointers
    double* X_data = NULL;
    double* Y_data = NULL;
    double* beta_out = NULL;
    double* se_out = NULL;
    double* zscore_out = NULL;
    double* pvalue_out = NULL;
    int overall_status = 1; // Assume error until success
    int status_in_loop = -99; // Use a temporary status inside the loop, init non-zero

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
             if (perm_count <= 0) {
                 fprintf(stderr,"Warning: --perm requires positive value, running T-test instead.\n");
                 run_test_type = 0; perm_count = 0;
             } else {
                 run_test_type = 1;
             }
        } else if (strcmp(argv[i], "--sparse") == 0 && i+1 < argc) {
            sparsity = atof(argv[i+1]);
            if (sparsity < 0.0 || sparsity > 1.0) {
                fprintf(stderr, "Warning: Sparsity must be between 0.0 and 1.0. Using default (%.1f).\n", DEFAULT_SPARSITY);
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
            run_test_type = 0; // Request T-test instead of permutation
            perm_count = 0;    // T-test uses nrand=0
        } else if (strcmp(argv[i], "--omp-threads") == 0 && i+1 < argc) {
            num_omp_settings = parse_thread_list(argv[i+1], omp_thread_list, MAX_THREAD_SETTINGS);
            if (num_omp_settings <= 0) {
                fprintf(stderr, "Error: Invalid --omp-threads list '%s'. Use comma-separated numbers (e.g., 1,2,4).\n", argv[i+1]);
                return 1;
            }
            got_omp_arg = 1;
            i++;
        } else if (strcmp(argv[i], "--blas-threads") == 0 && i+1 < argc) {
            num_blas_settings = parse_thread_list(argv[i+1], blas_thread_list, MAX_THREAD_SETTINGS);
             if (num_blas_settings <= 0) {
                fprintf(stderr, "Error: Invalid --blas-threads list '%s'. Use comma-separated numbers (e.g., 1,2,4).\n", argv[i+1]);
                return 1;
            }
            got_blas_arg = 1;
            i++;
        } else if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
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
        if (max_omp > 1 && max_omp <= MAX_THREAD_SETTINGS) {
            int found = 0;
            for(int k=0; k<num_omp_settings; ++k) if(omp_thread_list[k] == max_omp) found = 1;
            if (!found) {
               omp_thread_list[num_omp_settings++] = max_omp;
            }
        }
        printf("INFO: No --omp-threads specified, defaulting to: ");
        for(int k=0; k<num_omp_settings; ++k) printf("%d%s", omp_thread_list[k], (k == num_omp_settings - 1) ? "" : ",");
        printf("\n");
        #else
        omp_thread_list[0] = 1;
        num_omp_settings = 1;
        printf("INFO: OpenMP not enabled. Using 1 OMP thread.\n");
        #endif
    }
    if (!got_blas_arg) {
        blas_thread_list[0] = 1; // Default to 1 BLAS thread is usually safest for profiling OMP
        num_blas_settings = 1;
        printf("INFO: No --blas-threads specified, defaulting to: 1\n");
    }

    // --- Display Configuration ---
    printf("\n--- Configuration ---\n");
    printf("Test Type:      %s (%d permutations)\n",
           run_test_type == 1 ? "Permutation" : "T-Test",
           run_test_type == 1 ? perm_count : 0);
    printf("Dimensions:     X(%zu x %zu), Y(%zu x %zu)\n", n_genes, n_features, n_genes, n_samples);
    printf("Y Density:      %.2f%%\n", sparsity * 100.0);
    printf("Lambda:         %.2f\n", lambda_val);
    printf("OMP Threads:    [");
    for(int i=0; i<num_omp_settings; ++i) printf("%d%s", omp_thread_list[i], (i == num_omp_settings - 1) ? "" : ", ");
    printf("]\n");
    printf("BLAS Threads:   [");
    for(int i=0; i<num_blas_settings; ++i) printf("%d%s", blas_thread_list[i], (i == num_blas_settings - 1) ? "" : ", ");
    printf("]\n");
    #ifndef _OPENMP
    printf("NOTE: Compiled without OpenMP support.\n");
    #endif
    #ifndef HAVE_OPENBLAS // Assuming HAVE_OPENBLAS is defined via Makefile if linked
    printf("NOTE: Compiled without OpenBLAS support (or HAVE_OPENBLAS not defined).\n");
    printf("      BLAS thread settings may have no effect.\n");
    #endif
    printf("---------------------\n");

    print_memory_info(n_genes, n_features, n_samples);

    // --- Allocate memory for test data (once) ---
    printf("\nAllocating memory for X, Y, and Results...\n");
    fflush(stdout);
    X_data = (double*)malloc(n_genes * n_features * sizeof(double));
    Y_data = (double*)malloc(n_genes * n_samples * sizeof(double));
    size_t pm_size = n_features * n_samples;
    beta_out = (double*)malloc(pm_size * sizeof(double));
    se_out = (double*)malloc(pm_size * sizeof(double));
    zscore_out = (double*)malloc(pm_size * sizeof(double));
    pvalue_out = (double*)malloc(pm_size * sizeof(double));

    if (!X_data || !Y_data || !beta_out || !se_out || !zscore_out || !pvalue_out) {
        fprintf(stderr, "FATAL: Memory allocation failed.\n");
        goto cleanup;
    }
    printf("Memory allocation successful.\n");

    // --- Generate Data (once) ---
    printf("Generating random data (this may take time for large matrices)...\n");
    fflush(stdout);
    srand((unsigned int)time(NULL)); // Use time-based seed

    printf("Generating X data...\n");
    generate_chunked_random_data(X_data, n_genes, n_features, 0.0, 3.0);

    printf("Generating Y data (density %.2f%%)...\n", sparsity * 100.0);
    size_t y_nnz_count = 0; // Variable to hold actual nnz count from generation
    if (sparsity < 1.0) {
        // Use the improved sparse generation function
        generate_sparse_fill_dense_oversample(Y_data, n_genes, n_samples, sparsity, 0.0, 10.0, &y_nnz_count);
    } else {
        generate_chunked_random_data(Y_data, n_genes, n_samples, 0.0, 10.0);
    }
    printf("Data generation complete.\n");

    // --- Prepare Parameters for ridge_gsl_reg ---
    int n_int = (int)n_genes;
    int p_int = (int)n_features;
    int m_int = (int)n_samples;
    double n_rand_val = run_test_type == 1 ? (double)perm_count : 0.0;
    double df_out = NAN;

    // --- Profiling Loop ---
    printf("\n--- Starting Profiling Runs ---\n");
    printf("%-12s | %-12s | %-15s | %-8s | %s\n", "OMP Threads", "BLAS Threads", "Time (seconds)", "Status", "Result Summary");
    printf("-------------------------------------------------------------------------------\n");
    fflush(stdout);

    int initial_blas_threads = ridge_gsl_get_blas_threads(); // Store initial BLAS setting
    // No need for last_run_status here, use status_in_loop

    for (int i = 0; i < num_omp_settings; ++i) {
        int current_omp_threads = omp_thread_list[i];

        #ifndef _OPENMP
        if (current_omp_threads > 1) {
            printf("%-12d | %-12s | %-15s | %-8s | %s\n",
                   current_omp_threads, "N/A", "N/A", "SKIP", "OpenMP not enabled");
            continue;
        }
        #endif

        for (int j = 0; j < num_blas_settings; ++j) {
            int current_blas_threads = blas_thread_list[j];

            // --- Set Threads ---
            #ifdef _OPENMP
            omp_set_num_threads(current_omp_threads);
            #endif
            int actual_blas_threads_before = ridge_gsl_set_blas_threads(current_blas_threads);
            int actual_blas_threads_after = ridge_gsl_get_blas_threads();

            if (actual_blas_threads_after != current_blas_threads && current_blas_threads > 1) {
                 if(actual_blas_threads_after > 1 || current_blas_threads > 1) {
                    fprintf(stderr,"Warning: Requested %d BLAS threads, but GSL backend reports using %d.\n",
                            current_blas_threads, actual_blas_threads_after);
                 }
            }


            // --- Run and Time ---
            double start_time = omp_get_wtime();

            int current_status = ridge_gsl_reg(X_data, Y_data, &n_int, &p_int, &m_int,
                                              &lambda_val, &n_rand_val,
                                              beta_out, se_out, zscore_out, pvalue_out, &df_out);
            status_in_loop = current_status; // Update status from this iteration

            double end_time = omp_get_wtime();
            double elapsed_time = end_time - start_time;

            // --- Print Results Row ---
            printf("%-12d | %-12d | %-15.4f | %-8d | ",
                   current_omp_threads,
                   actual_blas_threads_after,
                   elapsed_time,
                   current_status); // Print status from this specific run
            if (current_status == 0) {
                 print_results_summary(current_status, p_int, m_int, beta_out, se_out, zscore_out, pvalue_out, df_out);
            } else {
                printf("Failed - check stderr.");
            }
            printf("\n");
            fflush(stdout);

             if(actual_blas_threads_after != actual_blas_threads_before) {
                 ridge_gsl_set_blas_threads(actual_blas_threads_before);
             }
        } // End BLAS threads loop
    } // End OMP threads loop

    printf("-------------------------------------------------------------------------------\n");

    // Restore initial BLAS thread setting just in case
    ridge_gsl_set_blas_threads(initial_blas_threads);

    // Determine overall status based on the *last* run executed
    overall_status = (status_in_loop == 0) ? 0 : 1;


cleanup: // Unified cleanup label
    // --- Cleanup ---
    printf("\nCleaning up allocated memory...\n");
    free(X_data);
    free(Y_data);
    free(beta_out);
    free(se_out);
    free(zscore_out);
    free(pvalue_out);

    // ** Use status_in_loop here **
    printf("\nGSL C backend profiling finished. Final Run Status: %d\n", status_in_loop);
    return overall_status; // Return 0 only if last run was successful
}


// --- Helper Function Implementations ---

// Parses a comma-separated list of integers
int parse_thread_list(const char* arg, int* list, int max_len) {
    int count = 0;
    char* token;
    // Use strdup for safe modification by strtok_r
    char* str_copy = strdup(arg);
    if (!str_copy) {
        perror("Failed to duplicate string in parse_thread_list");
        return -1; // Indicate error
    }

    char* rest = str_copy;
    while ((token = strtok_r(rest, ",", &rest)) != NULL && count < max_len) {
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

    free(str_copy); // Free the duplicated string
    return count;
}


void print_usage(const char* program_name) {
    printf("\nUsage: %s [options]\n", program_name);
    printf("Options:\n");
    printf("  --genes N         Number of genes/observations [default: %d]\n", DEFAULT_GENES);
    printf("  --features N      Number of features [default: %d]\n", DEFAULT_FEATURES);
    printf("  --samples N       Number of samples [default: %d]\n", DEFAULT_SAMPLES);
    printf("  --perm N          Number of permutations [default: %d]. Runs Permutation Test.\n", DEFAULT_PERM);
    printf("  --ttest           Run T-test instead of Permutation Test.\n");
    printf("  --sparse VAL      Sparsity level (0.0-1.0) for Y matrix [default: %.1f]\n", DEFAULT_SPARSITY);
    printf("  --lambda VAL      Ridge regularization parameter [default: %.1f]\n", DEFAULT_LAMBDA);
    printf("  --omp-threads L   Comma-separated list of OpenMP threads (e.g., 1,2,4,8) [default: 1,MAX]\n");
    printf("  --blas-threads L  Comma-separated list of BLAS threads (e.g., 1,2,4) [default: 1]\n");
    printf("                    (Requires linking with threaded BLAS like OpenBLAS and using HAVE_OPENBLAS)\n");
    printf("  --help            Show this help message\n");
    printf("\nExamples:\n");
    printf("  # Profile permutation test with 1, 2, 4, 8 OMP threads, keeping BLAS single-threaded\n");
    printf("  %s --genes 8000 --features 1200 --perm 100 --omp-threads 1,2,4,8 --blas-threads 1\n\n", program_name);
    printf("  # Profile permutation test, trying different BLAS threads with 4 OMP threads\n");
    printf("  %s --genes 8000 --features 1200 --perm 100 --omp-threads 4 --blas-threads 1,2,4\n\n", program_name);
     printf("  # Profile T-test with max OMP threads and 1 BLAS thread\n");
    printf("  %s --genes 8000 --features 1200 --ttest --omp-threads %d --blas-threads 1\n\n", program_name, omp_get_max_threads());
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
    double oversample_factor = 1.0 + (0.2 * density); // Simple linear scaling
    if (oversample_factor > 1.5) oversample_factor = 1.5; // Cap factor
    if (density >= 1.0) oversample_factor = 1.0;
    if (target_nonzero_elements <= 1) oversample_factor = 1.0;

    size_t attempts = (size_t)(target_nonzero_elements * oversample_factor);
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
    unsigned int base_seed = (unsigned int)time(NULL) + 1;

    #pragma omp parallel
    {
       unsigned int thread_seed = base_seed + omp_get_thread_num() * 101 + (unsigned int)(omp_get_wtime()*1000);
       #pragma omp for schedule(static)
       for (size_t k = 0; k < attempts; ++k) {
           size_t idx = (size_t)(((double)rand_r(&thread_seed) / RAND_MAX) * total_elements);
           idx = (idx >= total_elements) ? total_elements - 1 : idx;

           double val = min_val + ((double)rand_r(&thread_seed) / RAND_MAX) * range;
           if (val == 0.0 && min_val <= 0.0 && max_val >= 0.0) {
               val = (rand_r(&thread_seed) % 2 == 0) ? 1e-9 : -1e-9;
           }
           dense_data[idx] = val;
       }
    }

    // 3. Count actual non-zeros (parallel)
    size_t actual_nnz = 0;
    #pragma omp parallel for reduction(+:actual_nnz) schedule(static)
    for(size_t k=0; k<total_elements; ++k) {
        if (dense_data[k] != 0.0) {
            actual_nnz++;
        }
    }
    *nnz_count = actual_nnz;

     long long diff = (long long)actual_nnz - (long long)target_nonzero_elements;
     double diff_percent = (target_nonzero_elements > 0) ? (100.0 * (double)diff / target_nonzero_elements) : 0.0;
     printf("Done (actual %zu nnz, target %zu, diff %+lld [%+.2f%%]).\n",
            actual_nnz, target_nonzero_elements, diff, diff_percent);
}


// Print memory usage information
void print_memory_info(size_t n, size_t p, size_t m) {
    double X_size_gb = ((double)n * p * sizeof(double)) / (1024 * 1024 * 1024);
    double Y_size_gb = ((double)n * m * sizeof(double)) / (1024 * 1024 * 1024);
    double output_buffers_gb = ((double)p * m * 4 * sizeof(double)) / (1024 * 1024 * 1024);
    // Estimate intermediate GSL matrix sizes (rough estimate)
    double temp_gsl_gb = ( ((double)p*p*3) + ((double)p*n) + ((double)n*m*2) + ((double)p*p) ) * sizeof(double) / (1024*1024*1024);
    double total_gb = X_size_gb + Y_size_gb + output_buffers_gb + temp_gsl_gb;

    printf("\n--- Memory Estimate ---\n");
    printf("  Input X:        ~%.3f GB\n", X_size_gb);
    printf("  Input Y:        ~%.3f GB\n", Y_size_gb);
    printf("  Output Buffers: ~%.3f GB\n", output_buffers_gb);
    printf("  Est. Temp GSL:  ~%.3f GB\n", temp_gsl_gb);
    printf("  Estimated Total:  ~%.3f GB\n", total_gb);
    printf("-----------------------\n");

    if (total_gb > 8.0) {
        printf("WARNING: Estimated memory usage is high (>8GB). Ensure sufficient RAM.\n");
    }
}

// Print results summary concisely for profiling table
// Added const qualifiers to pointers where data shouldn't be modified
void print_results_summary(int status, size_t p, size_t m,
                           const double* beta, const double* se,
                           const double* zscore, const double* pvalue, double df)
{
    if (status != 0) {
        printf("Fail(%d)", status); // Removed prefix, assuming context is clear from table
        return;
    }
    if (!beta || !se || !zscore || !pvalue) {
        printf("Err(NULL)");
        return;
    }

    // Find first valid index (non-NaN) if possible
    size_t idx = 0;
    size_t pm = p*m;
    if (pm == 0) {
         printf("Empty");
         return;
    }
    // Loop to find first valid set of stats
    while (idx < pm && (isnan(beta[idx]) || isnan(se[idx]) || isnan(zscore[idx]) || isnan(pvalue[idx]))) {
        idx++;
    }
    // If all stats are NaN, fall back to index 0 (beta might still be valid)
    if (idx >= pm) {
         idx = 0;
         // If beta[0] is also NaN, report as such
         if (isnan(beta[idx])) {
             printf("All NaN");
             return;
         }
    }

    // Print summary - more robust check for NaN df
    printf("Beta[0]=%.2e, SE[0]=%.2e, Z[0]=%.2e, PV[0]=%.2e%s",
        beta[idx], se[idx], zscore[idx], pvalue[idx],
        (isnan(df) ? "" : ", DF") // Just indicate if DF is present
        );
    // Optionally print DF value if needed and not NaN
    // if (!isnan(df)) {
    //     printf("=%.1f", df);
    // }
}