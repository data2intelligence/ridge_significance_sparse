# ridge_inference/ridge_mkl.pyx
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: infer_types=True

import numpy as np
cimport numpy as cnp
from scipy import sparse as sps
import time
import warnings
import logging

# Import the Python API for NumPy
cnp.import_array()

logger = logging.getLogger(__name__)

# --- OpenMP Configuration ---
cdef bint _has_openmp_mkl = False
try:
    from openmp cimport omp_get_max_threads, omp_set_num_threads
    _ = omp_get_max_threads()
    _has_openmp_mkl = True
    logger.debug("Successfully cimported OpenMP functions for MKL wrapper.")
except ImportError: _has_openmp_mkl = False
except Exception as e: _has_openmp_mkl = False; warnings.warn(f"OMP check failed (MKL): {e}")

# --- C Function Declarations ---
# Fixed function declarations to match ridge_mkl.h
cdef extern from "ridge_mkl.h" nogil:
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
    )
    int ridge_mkl_sparse(
        const double *X,
        int n_genes,
        int n_features,
        const double *Y_vals,
        const int *Y_col,
        const int *Y_row,
        int n_samples,
        int nnz,
        double lambda_val,
        int n_rand,
        double *beta,
        double *se,
        double *zscore,
        double *pvalue
    )
    int ridge_mkl_set_threads(int num_threads) nogil
    int ridge_mkl_get_threads() nogil

# --- Type definitions ---
ctypedef cnp.float64_t DTYPE_t
ctypedef cnp.int32_t ITYPE_t
ctypedef size_t SIZE_t

# --- Python-accessible Thread Control Functions ---
cpdef int set_mkl_threads(int num_threads):
    cdef int current_threads_on_err
    cdef int prev_threads
    if num_threads < 0:
        warnings.warn("MKL threads must be non-negative.")
        # Indent the 'with nogil' block correctly
        with nogil:
            current_threads_on_err = ridge_mkl_get_threads()
        return current_threads_on_err # Return after the nogil block

    # Proceed if num_threads is non-negative
    with nogil:
        prev_threads = ridge_mkl_set_threads(num_threads)
    if prev_threads == -1:
         warnings.warn("MKL thread control unavailable (MKL not linked or HAVE_MKL not defined?).")
         with nogil: return ridge_mkl_get_threads()
    logger.debug(f"MKL threads set request: {num_threads} (was {prev_threads}) via C helper.")
    return prev_threads

cpdef int get_mkl_threads():
    cdef int current_threads; 
    with nogil: 
        current_threads = ridge_mkl_get_threads()
    return current_threads if current_threads > 0 else 1

cpdef int get_mkl_omp_max_threads(): return omp_get_max_threads() if _has_openmp_mkl else 1

cpdef int set_mkl_omp_threads(int num_threads):
    if not _has_openmp_mkl: warnings.warn("OMP not available in MKL build."); return 1
    if num_threads <= 0: warnings.warn("OMP threads must be positive."); return get_mkl_omp_max_threads()
    prev = omp_get_max_threads(); omp_set_num_threads(num_threads); current = omp_get_max_threads()
    logger.debug(f"MKL OMP threads req: {num_threads} (was {prev}, now {current}).")
    return prev

# --- Main Ridge Regression Wrapper (MKL Backend) ---
def ridge_regression_mkl(X, Y, double lambda_val=5e5, int n_rand=1000, int mkl_threads=-1, int omp_threads=-1):
    logger.info("Executing ridge_regression_mkl (MKL backend)...")
    overall_start_time = time.time()

    # --- CDEF Declarations ---
    cdef cnp.ndarray[DTYPE_t, ndim=2, mode='c'] X_c
    cdef int n_genes, n_features, n_samples
    cdef int nnz = 0 
    cdef bint is_Y_sparse
    cdef cnp.ndarray[DTYPE_t, ndim=2, mode='c'] Y_c_dense = None
    cdef cnp.ndarray[DTYPE_t, ndim=1] Y_vals_c = None
    cdef cnp.ndarray[ITYPE_t, ndim=1] Y_indices_c = None, Y_indptr_c = None
    cdef const double* X_ptr
    cdef const double* Y_dense_ptr = NULL
    cdef const double* Y_vals_ptr = NULL
    cdef const int* Y_indices_ptr = NULL
    cdef const int* Y_indptr_ptr = NULL
    cdef double* beta_ptr = NULL
    cdef double* se_ptr = NULL
    cdef double* zscore_ptr = NULL
    cdef double* pvalue_ptr = NULL
    cdef double df_val = 0.0  # For t-test df output
    cdef int actual_omp=1, actual_mkl=1, prev_omp=-1, prev_mkl=-1, status=0

    # --- Input Validation & Prep ---
    prep_start_time = time.time()
    if lambda_val < 0: warnings.warn("lambda_val < 0. Clamping to 0."); lambda_val = 0.0
    if sps.issparse(X): raise TypeError("MKL backend currently requires dense X input.")
    try: X_c = np.require(X, dtype=np.float64, requirements=['C', 'A'])
    except Exception as e: raise TypeError(f"Failed X conversion: {e}") from e
    if X_c.ndim != 2: raise ValueError("Input X must be 2D.")
    n_genes = X_c.shape[0]; n_features = X_c.shape[1]

    is_Y_sparse = sps.isspmatrix_csr(Y)
    if not is_Y_sparse and sps.issparse(Y): Y = Y.tocsr(); is_Y_sparse = True

    if is_Y_sparse:
        if n_rand <= 0: raise ValueError(f"MKL sparse backend requires n_rand > 0, got {n_rand}.")
        Y_csr = Y
        if Y_csr.shape[0] != n_genes: raise ValueError(f"Sparse Y rows ({Y_csr.shape[0]}) != X rows ({n_genes}).")
        n_samples = Y_csr.shape[1]; nnz = Y_csr.nnz
        try:
            Y_vals_c = np.require(Y_csr.data, dtype=np.float64, requirements=['A'])
            Y_indices_c = np.require(Y_csr.indices, dtype=np.int32, requirements=['A'])
            Y_indptr_c = np.require(Y_csr.indptr, dtype=np.int32, requirements=['A'])
        except Exception as e: raise TypeError(f"Failed sparse Y type check: {e}") from e
    elif isinstance(Y, np.ndarray):
        if Y.ndim != 2: raise ValueError("Dense Y must be 2D.")
        if Y.shape[0] != n_genes: raise ValueError(f"Dense Y rows ({Y.shape[0]}) != X rows ({n_genes}).")
        is_Y_sparse = False
        try: Y_c_dense = np.require(Y, dtype=np.float64, requirements=['C', 'A'])
        except Exception as e: raise TypeError(f"Failed dense Y conversion: {e}") from e
        n_samples = Y_c_dense.shape[1]
    else: raise TypeError("Y must be NumPy ndarray or SciPy CSR sparse.")
    if n_genes == 0 or n_features == 0 or n_samples == 0: raise ValueError("Zero dimensions.")
    logger.debug(f"MKL shapes: X({n_genes},{n_features}), Y({n_genes},{n_samples}) {'CSR' if is_Y_sparse else 'Dense'}")

    cdef cnp.ndarray[DTYPE_t, ndim=2, mode='c'] beta_out = np.empty((n_features, n_samples), dtype=np.float64, order='C')
    cdef cnp.ndarray[DTYPE_t, ndim=2, mode='c'] se_out = np.empty((n_features, n_samples), dtype=np.float64, order='C')
    cdef cnp.ndarray[DTYPE_t, ndim=2, mode='c'] zscore_out = np.empty((n_features, n_samples), dtype=np.float64, order='C')
    cdef cnp.ndarray[DTYPE_t, ndim=2, mode='c'] pvalue_out = np.empty((n_features, n_samples), dtype=np.float64, order='C')

    X_ptr = &X_c[0, 0]
    if is_Y_sparse:
        Y_vals_ptr = &Y_vals_c[0] if nnz > 0 else NULL
        Y_indices_ptr = <const int*>&Y_indices_c[0] if nnz > 0 else NULL
        Y_indptr_ptr = <const int*>&Y_indptr_c[0]
    else: Y_dense_ptr = &Y_c_dense[0, 0]
    beta_ptr = &beta_out[0, 0]; se_ptr = &se_out[0, 0]; zscore_ptr = &zscore_out[0, 0]; pvalue_ptr = &pvalue_out[0, 0]
    logger.debug(f"MKL Prep took {time.time() - prep_start_time:.4f}s")

    # --- Thread Management ---
    thread_mgmt_start_time = time.time()
    if mkl_threads > 0: prev_mkl = set_mkl_threads(mkl_threads)
    actual_mkl = get_mkl_threads()
    if omp_threads > 0: prev_omp = set_mkl_omp_threads(omp_threads)
    actual_omp = get_mkl_omp_max_threads()
    logger.debug(f"MKL Thread setup took {time.time() - thread_mgmt_start_time:.4f}s. MKL={actual_mkl}, OMP={actual_omp}")

    # --- Call C function ---
    c_call_start_time = time.time()

    # Log outside nogil block
    if is_Y_sparse:
        logger.debug("Calling C function ridge_mkl_sparse...")
    else:
        logger.debug("Calling C function ridge_mkl_dense...")

    # Perform C function calls inside nogil block
    with nogil:
        if is_Y_sparse:
            status = ridge_mkl_sparse(
                X_ptr, 
                n_genes, 
                n_features,
                Y_vals_ptr, 
                Y_indices_ptr, 
                Y_indptr_ptr, 
                n_samples, 
                nnz, 
                lambda_val, 
                n_rand,
                beta_ptr, 
                se_ptr, 
                zscore_ptr, 
                pvalue_ptr
            )
        else:
            status = ridge_mkl_dense(
                X_ptr, 
                Y_dense_ptr, 
                n_genes, 
                n_features, 
                n_samples, 
                lambda_val, 
                n_rand,
                beta_ptr, 
                se_ptr, 
                zscore_ptr, 
                pvalue_ptr, 
                &df_val
            )

    c_call_duration = time.time() - c_call_start_time
    logger.info(f"C function ridge_mkl_{'sparse' if is_Y_sparse else 'dense'} exec: {c_call_duration:.4f}s, status: {status}")

    # --- Restore threads ---
    restore_start_time = time.time()
    if prev_mkl != -1: set_mkl_threads(prev_mkl)
    if prev_omp != -1: set_mkl_omp_threads(prev_omp)
    logger.debug(f"MKL Thread restore took {time.time() - restore_start_time:.4f}s")

    # --- Handle errors ---
    if status != 0:
        err_map = {1:"NULL ptr", 2:"Dims", 3:"Alloc", 4:"LAPACK", 5:"BLAS", 6:"SPBLAS", 7:"VSL", 8:"Perm", 9:"Unsupp", 10:"Input"}
        msg = err_map.get(status, f"Unknown {status}")
        beta_out.fill(np.nan); se_out.fill(np.nan); zscore_out.fill(np.nan); pvalue_out.fill(np.nan)
        logger.error(f"MKL C func failed: {msg} (code {status})")
        raise RuntimeError(f"MKL C func failed: {msg} (code {status})")

    # --- Return results ---
    logger.info(f"ridge_regression_mkl finished successfully in {time.time() - overall_start_time:.4f}s")
    result_dict = {
        'beta': beta_out, 
        'se': se_out, 
        'zscore': zscore_out, 
        'pvalue': pvalue_out,
        'method_used': 'mkl', 
        'test_type': 'permutation' if n_rand > 0 else 't-test',
        'is_Y_sparse_input': is_Y_sparse,
        'mkl_threads_used': actual_mkl, 
        'omp_threads_used': actual_omp, 
        'execution_time': c_call_duration
    }
    
    # Include df value for t-test
    if not is_Y_sparse and n_rand <= 0:
        result_dict['df'] = df_val
        
    return result_dict