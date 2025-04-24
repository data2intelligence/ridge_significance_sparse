# ridge_inference/ridge_gsl.pyx
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: infer_types=True
# cython: profile=False
# (Comment: Disable profiling for release builds)

import numpy as np
cimport numpy as cnp
import warnings
import time
import logging
from scipy import sparse as sps

# Import the Python API for NumPy
_np_init = False
if not _np_init:
    cnp.import_array()
    _np_init = True

logger = logging.getLogger(__name__)

# --- OpenMP Configuration ---
cdef bint _has_openmp_gsl = False
cdef bint _omp_warning_issued = False
try:
    from openmp cimport omp_get_max_threads, omp_set_num_threads
    _ = omp_get_max_threads()
    _has_openmp_gsl = True
    logger.debug("Successfully cimported OpenMP functions for GSL wrapper.")
except (ImportError, Exception) as e:
    _has_openmp_gsl = False

# --- C Function Declarations ---
cdef extern from "ridge_gsl.h" nogil:
    int ridge_gsl_reg(double *X_vec, double *Y_vec,
                      int *n_pt, int *p_pt, int *m_pt,
                      double *lambda_pt, double *nrand_pt,
                      double *beta_vec, double *se_vec,
                      double *zscore_vec, double *pvalue_vec,
                      double *df_out)
    int ridge_gsl_set_blas_threads(int num_threads) nogil
    int ridge_gsl_get_blas_threads() nogil

# --- Type definitions ---
ctypedef cnp.float64_t DTYPE_t
ctypedef int INT_t

# --- Thread Control Functions ---
cpdef int set_gsl_blas_threads(int num_threads):
    cdef int current_threads_on_err = 1; 
    cdef int prev_threads = 1;
    if num_threads < 0:
        warnings.warn("BLAS threads must be non-negative.")
        with nogil: current_threads_on_err = ridge_gsl_get_blas_threads()
        return current_threads_on_err
    with nogil: prev_threads = ridge_gsl_set_blas_threads(num_threads)
    logger.debug(f"GSL BLAS threads requested: {num_threads}, previous: {prev_threads}.")
    return prev_threads

cpdef int get_gsl_blas_threads():
    cdef int current_threads = 1;
    with nogil: current_threads = ridge_gsl_get_blas_threads()
    return current_threads if current_threads > 0 else 1

cpdef int get_gsl_omp_max_threads():
    global _omp_warning_issued
    if not _has_openmp_gsl:
        if not _omp_warning_issued: warnings.warn("OpenMP support not available in this GSL Cython build."); _omp_warning_issued = True
        return 1
    cdef int max_threads = 1;
    with nogil: max_threads = omp_get_max_threads()
    return max_threads if max_threads > 0 else 1

cpdef int set_gsl_omp_threads(int num_threads):
    global _omp_warning_issued
    if not _has_openmp_gsl:
        if not _omp_warning_issued: warnings.warn("OpenMP support not available, cannot set threads."); _omp_warning_issued = True
        return 1
    if num_threads <= 0: warnings.warn("OMP threads must be positive."); return get_gsl_omp_max_threads()
    cdef int prev_threads = 1; 
    cdef int current_threads = 1;
    with nogil:
        prev_threads = omp_get_max_threads(); omp_set_num_threads(num_threads); current_threads = omp_get_max_threads()
    logger.debug(f"GSL OMP threads requested: {num_threads}, previous max: {prev_threads}, current max: {current_threads}.")
    return prev_threads if prev_threads > 0 else 1

# --- Main Ridge Regression Wrapper (GSL Backend) ---
def ridge_regression_gsl(X, Y, double lambda_val=5e5, int n_rand=1000, int blas_threads=-1, int omp_threads=-1):
    """
    Performs ridge regression with significance testing using the GSL backend.

    Calculates beta coefficients, standard errors (SE), Z-scores/T-statistics,
    and p-values for the association between columns of X (features) and
    columns of Y (samples/outcomes). Supports permutation testing or analytical
    t-tests for significance.

    Args:
        X (np.ndarray): Input data matrix (n_genes x n_features), dense NumPy array (float64).
        Y (np.ndarray): Output data matrix (n_genes x n_samples), dense NumPy array (float64).
        lambda_val (float): Ridge regularization parameter (lambda >= 0). Default: 5e5.
        n_rand (int): Number of permutations for permutation testing.
                      If n_rand <= 0, a t-test is performed instead. Default: 1000.
        blas_threads (int): Number of threads for BLAS operations (e.g., matrix multiply).
                            -1 uses the current BLAS default. Requires linking with a
                            threaded BLAS library (like OpenBLAS) and building with
                            HAVE_OPENBLAS defined. Default: -1.
        omp_threads (int): Number of threads for OpenMP parallel sections within the C code
                           (e.g., permutation loops, result calculation).
                           -1 uses the current OpenMP default (often set by OMP_NUM_THREADS).
                           Default: -1.

    Returns:
        dict: A dictionary containing:
            - 'beta' (np.ndarray): Beta coefficients (p x m).
            - 'se' (np.ndarray): Standard errors (p x m).
            - 'zscore' (np.ndarray): Z-scores (permutation) or T-statistics (t-test) (p x m).
            - 'pvalue' (np.ndarray): P-values (p x m).
            - 'df' (float): Degrees of freedom (only included if t-test was run and successful).
            - 'method_used' (str): Identifier 'gsl_cython'.
            - 'test_type' (str): 'permutation' or 't-test'.
            - 'blas_threads_used' (int): Actual number of BLAS threads reported by the backend.
            - 'omp_threads_used' (int): Actual number of OMP threads reported by OpenMP runtime.
            - 'execution_time' (float): Wall-clock time (seconds) spent inside the C function call.

    Raises:
        TypeError: If X or Y are not dense NumPy arrays or cannot be converted to float64.
        ValueError: If X/Y dimensions are incompatible, zero, or n_rand is negative.
        RuntimeError: If the underlying C function reports an error.
    """
    logger.info("Executing ridge_regression_gsl (GSL backend)...")
    start_time = time.time()
    # --- CDEF Declarations ---
    cdef cnp.ndarray[DTYPE_t, ndim=2, mode='c'] X_c, Y_c;
    cdef INT_t n_c, p_c, m_c;
    cdef double nrand_c, lambda_c, df_c = np.nan;
    cdef cnp.ndarray[DTYPE_t, ndim=2, mode='c'] beta_out, se_out, zscore_out, pvalue_out;
    cdef double* X_data_ptr; 
    cdef double* Y_data_ptr;
    cdef double* beta_data_ptr; 
    cdef double* se_data_ptr;
    cdef double* zscore_data_ptr; 
    cdef double* pvalue_data_ptr;
    cdef int actual_omp = 1; 
    cdef int actual_blas = 1;
    cdef int prev_omp = -1; 
    cdef int prev_blas = -1;
    cdef int status = 0;

    # --- Input Validation & Prep ---
    prep_start_time = time.time()
    if sps.issparse(X) or sps.issparse(Y): raise TypeError("GSL backend requires dense NumPy arrays for X and Y.")
    if not isinstance(X, np.ndarray) or not isinstance(Y, np.ndarray): raise TypeError("GSL backend requires dense NumPy arrays for X and Y.")
    if X.ndim != 2 or Y.ndim != 2: raise ValueError("Inputs X and Y must be 2D.")
    if X.shape[0] != Y.shape[0]: raise ValueError(f"X rows ({X.shape[0]}) != Y rows ({Y.shape[0]})")

    try:
        X_c = np.require(X, dtype=np.float64, requirements=['C', 'A', 'W'])
        Y_c = np.require(Y, dtype=np.float64, requirements=['C', 'A', 'W'])
    except Exception as e: raise TypeError(f"Failed input array conversion/validation: {e}") from e

    if lambda_val < 0: warnings.warn("lambda_val < 0. Clamping to 0."); lambda_val = 0.0

    # *** SYNTAX FIX APPLIED HERE ***
    if X_c.shape[0] == 0 or X_c.shape[1] == 0 or Y_c.shape[1] == 0:
        raise ValueError(f"Input arrays have zero dimensions: X=({X_c.shape[0]}, {X_c.shape[1]}), Y=({Y_c.shape[0]}, {Y_c.shape[1]})")
    # ********************************

    n_c = <INT_t>X_c.shape[0]; p_c = <INT_t>X_c.shape[1]; m_c = <INT_t>Y_c.shape[1]
    test_type = 't-test' if n_rand <= 0 else 'permutation'
    nrand_c = float(n_rand); lambda_c = float(lambda_val)
    logger.debug(f"GSL Input shapes: X({n_c}, {p_c}), Y({n_c}, {m_c}). Lambda={lambda_c:.2e}, Nrand={nrand_c:.0f} ({test_type})")

    beta_out = np.empty((p_c, m_c), dtype=np.float64, order='C'); se_out = np.empty((p_c, m_c), dtype=np.float64, order='C')
    zscore_out = np.empty((p_c, m_c), dtype=np.float64, order='C'); pvalue_out = np.empty((p_c, m_c), dtype=np.float64, order='C')

    beta_data_ptr = &beta_out[0, 0] if beta_out.size > 0 else NULL
    se_data_ptr = &se_out[0, 0] if se_out.size > 0 else NULL
    zscore_data_ptr = &zscore_out[0, 0] if zscore_out.size > 0 else NULL
    pvalue_data_ptr = &pvalue_out[0, 0] if pvalue_out.size > 0 else NULL
    X_data_ptr = &X_c[0, 0] if X_c.size > 0 else NULL
    Y_data_ptr = &Y_c[0, 0] if Y_c.size > 0 else NULL
    logger.debug(f"GSL Prep took {time.time() - prep_start_time:.4f}s")

    # --- Thread Management ---
    thread_mgmt_start_time = time.time()
    if omp_threads > 0: prev_omp = set_gsl_omp_threads(omp_threads)
    actual_omp = get_gsl_omp_max_threads()
    if blas_threads >= 0: prev_blas = set_gsl_blas_threads(blas_threads)
    actual_blas = get_gsl_blas_threads()
    logger.debug(f"GSL Thread setup took {time.time() - thread_mgmt_start_time:.4f}s. OMP Max={actual_omp}, BLAS Actual={actual_blas}")

    # --- Call C function ---
    logger.debug("Calling C function ridge_gsl_reg...")
    c_call_start_time = time.time()
    if X_data_ptr == NULL or Y_data_ptr == NULL or \
       (p_c > 0 and m_c > 0 and (beta_data_ptr == NULL or se_data_ptr == NULL or zscore_data_ptr == NULL or pvalue_data_ptr == NULL)):
       logger.error("FATAL ERROR: NULL data pointer detected before calling C function.")
       raise MemoryError("Internal error: Data pointer became NULL before C call.")

    with nogil:
        status = ridge_gsl_reg(X_data_ptr, Y_data_ptr, &n_c, &p_c, &m_c,
                               &lambda_c, &nrand_c,
                               beta_data_ptr, se_data_ptr, zscore_data_ptr, pvalue_data_ptr,
                               &df_c)
    c_call_duration = time.time() - c_call_start_time
    logger.info(f"C function ridge_gsl_reg exec: {c_call_duration:.4f}s, status: {status}")

    # --- Restore threads ---
    restore_start_time = time.time()
    if prev_omp != -1: set_gsl_omp_threads(prev_omp)
    if prev_blas != -1: set_gsl_blas_threads(prev_blas)
    logger.debug(f"GSL Thread restore took {time.time() - restore_start_time:.4f}s")

    # --- Handle errors ---
    if status != 0:
        err_map = {1:"NULL ptr", 2:"Dims", 3:"Alloc", 4:"MatrixOp", 5:"Decomp", 6:"Invert", 7:"Perm", 8:"TTest", 9:"DGEMM"}
        msg = err_map.get(status, f"Unknown {status}")
        beta_out.fill(np.nan); se_out.fill(np.nan); zscore_out.fill(np.nan); pvalue_out.fill(np.nan); df_c = np.nan
        logger.error(f"GSL C function failed: {msg} (code {status})")
        raise RuntimeError(f"GSL C function failed: {msg} (code {status})")

    # --- Return results ---
    result_dict = {'beta': beta_out, 'se': se_out, 'zscore': zscore_out, 'pvalue': pvalue_out,
                   'method_used': 'gsl_cython', 'test_type': test_type,
                   'blas_threads_used': actual_blas, 'omp_threads_used': actual_omp, 'execution_time': c_call_duration}
    if test_type == 't-test' and not np.isnan(df_c): result_dict['df'] = df_c
    logger.info(f"ridge_regression_gsl finished successfully in {time.time() - start_time:.4f}s")
    return result_dict