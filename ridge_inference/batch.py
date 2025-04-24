# ridge_inference/batch.py

import numpy as np
import pandas as pd
import logging
import time
import math
import gc
from scipy import sparse as sps

# Import specific calculation functions and availability flags from core
try:
    from .core import (
        _calculate_T_numpy, _calculate_beta_numpy, _perform_permutation_numpy,
        _perform_ttest_numpy, # Keep t-test numpy implementation
        NUMBA_AVAILABLE, # Still need this flag, even if Numba batching removed
        _calculate_T_cupy, _calculate_beta_cupy, _perform_permutation_cupy, CUPY_AVAILABLE,
        _transfer_to_gpu,
        CuSparseError, LinAlgError as CuPyLinAlgError, OutOfMemoryError as CuPyOutOfMemoryError,
        cp # Import 'cp' alias from core if available
    )
    # Define GPU specific errors if CuPy is available
    GPU_ERRORS = (CuPyOutOfMemoryError, CuSparseError, CuPyLinAlgError) if CUPY_AVAILABLE else ()
except ImportError as e:
     logging.critical(f"CRITICAL ERROR: Could not import core ridge functions/cupy for batch processing: {e}", exc_info=True)
     NUMBA_AVAILABLE = False
     CUPY_AVAILABLE = False
     cp = None
     GPU_ERRORS = ()
     # Define dummy functions for those that might be missing
     def _dummy_func(*args, **kwargs): raise ImportError("Core function missing due to import error")
     _calculate_T_numpy = _dummy_func
     _calculate_beta_numpy = _dummy_func
     _perform_permutation_numpy = _dummy_func
     _perform_ttest_numpy = _dummy_func
     _calculate_T_cupy = _dummy_func
     _calculate_beta_cupy = _dummy_func
     _perform_permutation_cupy = _dummy_func
     _transfer_to_gpu = _dummy_func

# Import MKL/GSL flags to check if they exist, but don't use them for batching
try:
    from .backend_selection import MKL_AVAILABLE, GSL_CYTHON_AVAILABLE
except ImportError:
    MKL_AVAILABLE, GSL_CYTHON_AVAILABLE = False, False

logger = logging.getLogger(__name__)

# Define fallback errors, potentially excluding GPU if needed
FALLBACK_ERRORS_BASE = (
    NotImplementedError, RuntimeError, ImportError, FileNotFoundError, TypeError, ValueError, MemoryError, ArithmeticError,
)
FALLBACK_ERRORS = tuple(list(FALLBACK_ERRORS_BASE) + list(GPU_ERRORS))


# --- Helper to select backend functions (Removed MKL/GSL/Numba options) ---
def _get_backend_funcs(method_name):
    """Returns a dictionary of calculation functions for the chosen backend."""
    if method_name == 'python':
        return {
            'calc_T': _calculate_T_numpy,
            'calc_beta': _calculate_beta_numpy,
            'perm_test': _perform_permutation_numpy,
            't_test': _perform_ttest_numpy, # Retain t-test for potential future batching
            'transfer': None,
            'cleanup': lambda: gc.collect(),
            'is_gpu': False,
            'requires_dense_X': False, # Python iterative handles sparse X (though currently disabled in ridge_batch)
            'requires_dense_Y': False
        }
    elif method_name == 'gpu' and CUPY_AVAILABLE:
        if cp is None: raise RuntimeError("CuPy reported available, but 'cp' module is None.")
        def gpu_cleanup():
            try:
                if cp: cp.get_default_memory_pool().free_all_blocks()
            except Exception as pool_e:
                 logger.warning(f"CuPy batch cleanup: Could not free pool blocks: {pool_e}")
            gc.collect()

        return {
            'calc_T': _calculate_T_cupy,
            'calc_beta': _calculate_beta_cupy,
            'perm_test': _perform_permutation_cupy,
            't_test': None, # T-test not implemented for GPU
            'transfer': _transfer_to_gpu,
            'cleanup': gpu_cleanup,
            'is_gpu': True,
            'requires_dense_X': False, # Handles sparse X via transfer (densifies)
            'requires_dense_Y': False # Handles sparse Y via transfer/compute
        }
    elif method_name in ['mkl', 'gsl_cython', 'numba']:
         raise NotImplementedError(f"Backend '{method_name}' does not currently support batch processing via ridge_batch.")
    else:
        # Fallback to python if requested backend is invalid/unavailable
        if method_name != 'python':
             logger.warning(f"Requested backend '{method_name}' not available or invalid for batching, defaulting to 'python'.")
        return _get_backend_funcs('python') # Default to python


# --- Main Batch Function ---
def ridge_batch(X, Y, lambda_val, n_rand, method='auto', batch_size=5000, verbose=0):
    """
    Performs ridge regression in batches using Python or GPU backends.
    Calculates T matrix once. Handles sparse Y.

    Parameters
    ----------
    X : numpy.ndarray
        Input matrix (n_genes, n_features). Must be dense for batching.
    Y : numpy.ndarray or scipy.sparse matrix
        Output matrix (n_genes, n_samples). Can be dense or sparse.
    lambda_val : float
        Ridge regularization parameter.
    n_rand : int
        Number of permutations. Must be > 0 for batching.
    method : str
        Backend method ('python', 'gpu', 'auto'). Default 'auto' chooses GPU if available.
    batch_size : int
        Number of samples (columns of Y) to process in each batch.
    verbose : int
        Verbosity level (0=quiet, 1=info, 2=debug).

    Returns
    -------
    dict
        Results: 'beta', 'se', 'zscore', 'pvalue' (np.ndarrays),
        'method_used', 'execution_time', 'peak_gpu_pool_mb' (optional).
    """
    start_time_batch = time.time()
    is_X_sparse_orig = sps.issparse(X)

    # --- Check for Sparse X (still not supported in batch) ---
    if is_X_sparse_orig:
        raise NotImplementedError("Batch processing (`ridge_batch`) currently requires dense X input. Use standard `ridge` for sparse X.")

    n_genes, n_features = X.shape
    n_samples = Y.shape[1]

    if n_rand <= 0:
        raise NotImplementedError("Batch processing currently only supports permutation tests (n_rand > 0).")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    logger.info(f"Starting ridge_batch: n_samples={n_samples}, batch_size={batch_size}, method='{method}'")

    # --- Auto Method Selection (Python or GPU only for batching) ---
    selected_backend_name = method.lower()
    if selected_backend_name == 'auto':
        selected_backend_name = 'gpu' if CUPY_AVAILABLE else 'python'
        logger.info(f"Auto method selected batch backend: '{selected_backend_name}'")
    elif selected_backend_name not in ['python', 'gpu']:
        logger.warning(f"Requested method '{method}' is not supported for batching. Defaulting to 'python'.")
        selected_backend_name = 'python'

    # --- Get Backend Functions ---
    try:
        backend = _get_backend_funcs(selected_backend_name)
        logger.info(f"Using backend '{selected_backend_name}' for batch processing.")
    except NotImplementedError as nie:
        logger.error(f"Backend '{selected_backend_name}' is not supported for batch processing: {nie}")
        raise
    except Exception as e:
        logger.error(f"Failed to get backend functions for '{selected_backend_name}': {e}", exc_info=True)
        raise RuntimeError(f"Could not initialize batch backend '{selected_backend_name}'.") from e

    # --- Input Compatibility Check ---
    # Redundant now as only dense X allowed, but keep structure
    if backend.get('requires_dense_X', False) and is_X_sparse_orig:
         raise TypeError(f"Backend '{selected_backend_name}' requires dense X, but received sparse input.")


    # --- Pre-computation (T matrix, GPU transfers) ---
    T = None
    X_input_backend = X # Variable used by backends (might become GPU array)
    Y_input_backend = Y # Variable used by backends (might become GPU array)

    try:
        if backend['is_gpu']:
            if not CUPY_AVAILABLE or cp is None: raise RuntimeError("GPU backend selected but CuPy is not available.")
            if backend['transfer']:
                logger.info("Transferring data to GPU for batch processing...")
                # X is already dense here
                X_gpu, Y_gpu = backend['transfer'](X, Y)
                X_input_backend = X_gpu
                Y_input_backend = Y_gpu
                logger.info("GPU transfer complete. Calculating T matrix on GPU...")
                T = backend['calc_T'](X_input_backend, float(lambda_val))
                logger.info("GPU T matrix calculation complete.")
            else: raise RuntimeError("GPU backend selected but no transfer function found.")
        else: # Python backend
            logger.info(f"Calculating T matrix on CPU using '{selected_backend_name}' logic...")
            # Python backend _calculate_T requires dense X
            X_for_T = np.require(X, dtype=np.float64, requirements=['C', 'A'])
            T = backend['calc_T'](X_for_T, float(lambda_val))
            X_input_backend = X_for_T # Python uses CPU arrays
            Y_input_backend = Y
            logger.info("CPU T matrix calculation complete.")

    except Exception as e:
         logger.error(f"Failed during pre-computation (T matrix / GPU transfer) for backend '{selected_backend_name}': {e}", exc_info=True)
         if 'X_gpu' in locals(): del X_gpu
         if 'Y_gpu' in locals(): del Y_gpu
         if backend.get('cleanup'): backend['cleanup']()
         if T is not None: del T
         raise RuntimeError("Critical error during batch pre-computation.") from e

    # --- Batch Loop ---
    num_batches = math.ceil(n_samples / batch_size)
    all_beta, all_se, all_zscore, all_pvalue = [], [], [], []
    total_compute_time = 0
    peak_gpu_mem_mb_batch = np.nan

    # Progress bar setup
    iter_batches = range(num_batches)
    try:
        from tqdm import tqdm
        if verbose >= 1 or num_batches > 10:
             iter_batches = tqdm(iter_batches, desc=f"Processing Batches ({selected_backend_name})", total=num_batches, leave=False)
    except ImportError:
        if verbose >= 1: logger.info("tqdm not found, progress bar disabled.")

    # --- Main Loop ---
    for i in iter_batches:
        batch_start_time = time.time()
        start_col = i * batch_size
        end_col = min((i + 1) * batch_size, n_samples)
        current_batch_size = end_col - start_col
        if verbose > 1: logger.debug(f"Batch {i+1}/{num_batches}: Y cols {start_col}-{end_col-1} (size {current_batch_size})")

        Y_batch_backend = Y_input_backend[:, start_col:end_col]

        beta_batch_res, se_batch_res, zscore_batch_res, pvalue_batch_res = None, None, None, None

        # --- Execute Batch Computation ---
        try:
            logger.debug(f"Batch {i+1}: Calculating beta using precomputed T.")
            beta_batch = backend['calc_beta'](T, Y_batch_backend)

            logger.debug(f"Batch {i+1}: Performing permutation test.")
            se_batch, zscore_batch, pvalue_batch = backend['perm_test'](
                T, Y_batch_backend, beta_batch, n_rand
            )

            if backend['is_gpu']:
                if cp is None or not CUPY_AVAILABLE:
                     raise RuntimeError("GPU backend: CuPy (cp) not available for result transfer.")
                logger.debug(f"Batch {i+1}: Transferring results from GPU to CPU.")
                # Measure GPU memory usage
                try:
                    mempool = cp.get_default_memory_pool()
                    current_used = mempool.used_bytes() / (1024*1024)
                    if np.isnan(peak_gpu_mem_mb_batch) or current_used > peak_gpu_mem_mb_batch:
                        peak_gpu_mem_mb_batch = current_used
                except Exception as mem_e: logger.warning(f"Batch {i+1}: Failed to get GPU memory usage: {mem_e}")

                beta_batch_res = cp.asnumpy(beta_batch) if isinstance(beta_batch, cp.ndarray) else beta_batch
                se_batch_res = cp.asnumpy(se_batch) if isinstance(se_batch, cp.ndarray) else se_batch
                zscore_batch_res = cp.asnumpy(zscore_batch) if isinstance(zscore_batch, cp.ndarray) else zscore_batch
                pvalue_batch_res = cp.asnumpy(pvalue_batch) if isinstance(pvalue_batch, cp.ndarray) else pvalue_batch
                del beta_batch, se_batch, zscore_batch, pvalue_batch
                if backend.get('cleanup'): backend['cleanup']()
            else:
                beta_batch_res, se_batch_res, zscore_batch_res, pvalue_batch_res = beta_batch, se_batch, zscore_batch, pvalue_batch

            # --- Append results ---
            if beta_batch_res is None or se_batch_res is None or zscore_batch_res is None or pvalue_batch_res is None:
                raise RuntimeError(f"Backend failed to return all result matrices for batch {i+1}.")
            expected_shape = (n_features, current_batch_size)
            if beta_batch_res.shape != expected_shape: raise ValueError(f"Batch {i+1} beta shape mismatch: got {beta_batch_res.shape}, expected {expected_shape}")
            all_beta.append(beta_batch_res)
            all_se.append(se_batch_res)
            all_zscore.append(zscore_batch_res)
            all_pvalue.append(pvalue_batch_res)

            batch_duration = time.time() - batch_start_time
            total_compute_time += batch_duration
            if verbose > 0 and not isinstance(iter_batches, tqdm): logger.info(f"Batch {i+1}/{num_batches} completed in {batch_duration:.2f}s.")

        except FALLBACK_ERRORS as e:
             logger.error(f"Backend '{selected_backend_name}' failed on Batch {i+1}: {type(e).__name__} - {e}", exc_info=(verbose > 0))
             raise RuntimeError(f"Backend failed during batch processing on batch {i+1}. Cannot continue.") from e
        except Exception as e:
             logger.error(f"Unexpected error in backend '{selected_backend_name}' on Batch {i+1}: {e}", exc_info=True)
             raise RuntimeError(f"Unexpected backend error during batch processing on batch {i+1}. Cannot continue.") from e
        finally:
             del Y_batch_backend
             if 'beta_batch_res' in locals(): del beta_batch_res
             if 'se_batch_res' in locals(): del se_batch_res
             if 'zscore_batch_res' in locals(): del zscore_batch_res
             if 'pvalue_batch_res' in locals(): del pvalue_batch_res
             gc.collect()

    # --- Concatenate Results ---
    if not all_beta:
        raise RuntimeError("No batches were successfully processed or results collected.")
    logger.info(f"Concatenating results from {len(all_beta)} batches...")
    try:
        final_beta = np.hstack([np.asarray(b) for b in all_beta])
        final_se = np.hstack([np.asarray(s) for s in all_se])
        final_zscore = np.hstack([np.asarray(z) for z in all_zscore])
        final_pvalue = np.hstack([np.asarray(p) for p in all_pvalue])
    except ValueError as e:
        logger.error(f"Failed to concatenate batch results: {e}", exc_info=True)
        raise RuntimeError("Could not concatenate results from batches.") from e

    # --- Final Cleanup ---
    logger.debug("Performing final cleanup...")
    del T, X_input_backend, Y_input_backend, all_beta, all_se, all_zscore, all_pvalue
    if 'X_gpu' in locals(): del X_gpu # If GPU was used
    if 'Y_gpu' in locals(): del Y_gpu
    if backend.get('cleanup'): backend['cleanup']()
    gc.collect()

    total_time_batch = time.time() - start_time_batch
    logger.info(f"Ridge_batch completed using '{selected_backend_name}' in {total_time_batch:.2f}s (Compute time: {total_compute_time:.2f}s)")
    if not np.isnan(peak_gpu_mem_mb_batch): logger.info(f"Peak GPU memory usage (approx): {peak_gpu_mem_mb_batch:.2f} MB")

    return {
        'beta': final_beta, 'se': final_se, 'zscore': final_zscore, 'pvalue': final_pvalue,
        'method_used': f"{selected_backend_name}_batched",
        'execution_time': total_time_batch,
        'peak_gpu_pool_mb': peak_gpu_mem_mb_batch if not np.isnan(peak_gpu_mem_mb_batch) else None
    }