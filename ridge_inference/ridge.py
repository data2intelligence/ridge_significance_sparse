# ridge_inference/ridge.py

import gc
import time
import numpy as np
import pandas as pd
import traceback # Keep for potential detailed error logging if needed
from scipy import sparse as sps
from scipy import stats
import logging
import warnings # Use warnings module

# --- Define logger for this module ---
logger = logging.getLogger(__name__)

# --- Default settings ---
DEFAULT_LAMBDA = 5e5
DEFAULT_NRAND = 1000

# Import necessary components from the updated backend_selection
from .backend_selection import (
    select_optimal_backend,
    get_backend_function,
    get_available_backends,
    is_backend_available,
    set_backend,
    list_backends,
    set_threads,
    BACKEND_PREFERENCE,
    BACKEND_FALLBACK_ENABLED,
    MKL_AVAILABLE,
    GSL_CYTHON_AVAILABLE, # Updated flag name
    NUMBA_AVAILABLE,
    CUPY_AVAILABLE,
    PYTHON_AVAILABLE
)

# Re-export convenience functions
set_backend_preference = set_backend
list_available_backends = list_backends
set_thread_count = set_threads

# Define errors that might trigger fallback attempts
# Combine errors from different backends might be necessary
# GPU specific errors might be defined in core/batch
try:
    from .batch import GPU_ERRORS
    FALLBACK_ERRORS_BASE = (
        NotImplementedError, RuntimeError, ImportError, FileNotFoundError,
        TypeError, ValueError, MemoryError, ArithmeticError
    )
    FALLBACK_ERRORS = tuple(set(list(FALLBACK_ERRORS_BASE) + list(GPU_ERRORS)))
except ImportError:
     FALLBACK_ERRORS = (
        NotImplementedError, RuntimeError, ImportError, FileNotFoundError,
        TypeError, ValueError, MemoryError, ArithmeticError
    )


def ridge(X, Y, lambda_=DEFAULT_LAMBDA, alternative="two-sided", n_rand=DEFAULT_NRAND, method="auto", verbose=0):
    """
    Performs Ridge regression with significance testing (permutation or t-test).

    Automatically dispatches to the most suitable backend based on input data
    characteristics (dense/sparse) and available hardware/libraries (MKL, GSL,
    Numba, GPU, Python). Handles necessary data conversions for selected backends.

    Parameters
    ----------
    X : numpy.ndarray or scipy.sparse matrix
        Input matrix (n_genes, n_features). Sparse X requires 'python' backend.
    Y : numpy.ndarray or scipy.sparse matrix
        Output matrix (n_genes, n_samples). Can be dense or sparse.
    lambda_ : float, default=5e5
        Ridge regularization parameter (lambda >= 0).
    alternative : {'two-sided', 'greater', 'less'}, default='two-sided'
        Alternative hypothesis for the t-test (used only if n_rand=0 and backend is 'gsl_cython' or 'python').
    n_rand : int, default=1000
        Number of permutations. If 0, a t-test is performed.
    method : {'auto', 'mkl', 'gsl_cython', 'gsl', 'numba', 'gpu', 'python'}, default='auto'
        Desired backend. 'auto' selects best available. 'gsl' is alias for 'gsl_cython'.
    verbose : int, default=0
        Verbosity level (0=quiet, 1=info, 2=debug).

    Returns
    -------
    dict
        Results and metadata: 'beta', 'se', 'zscore', 'pvalue' (np.ndarrays),
        'method_used', 'execution_time', 'peak_gpu_pool_mb' (optional), 'df' (optional for t-test).
    """
    start_time = time.time()
    requested_method = method.lower() if isinstance(method, str) else "auto"
    logger.info(f"Starting ridge regression. Requested Method: {requested_method}, n_rand: {n_rand}, lambda: {lambda_}")

    # --- Input Validation ---
    if not hasattr(X, 'shape') or not hasattr(Y, 'shape') or X.ndim != 2 or Y.ndim != 2 or X.shape[0] != Y.shape[0]:
        raise ValueError(f"Invalid input shapes/dimensions: X={getattr(X,'shape','N/A')}, Y={getattr(Y,'shape','N/A')}")
    if n_rand < 0: raise ValueError("n_rand cannot be negative.")
    if lambda_ < 0:
        logger.warning(f"Negative lambda_ ({lambda_}) provided. Clamping to 0.")
        lambda_ = 0.0
    valid_alternatives = ["two-sided", "greater", "less"]
    if alternative not in valid_alternatives:
        raise ValueError(f"Invalid alternative '{alternative}'. Options: {valid_alternatives}")

    X_input = X.values if isinstance(X, pd.DataFrame) else X
    Y_input = Y.values if isinstance(Y, pd.DataFrame) else Y

    # --- Select Backend ---
    selected_backend = select_optimal_backend(
        X_input, Y_input, n_rand=n_rand, user_selection=requested_method
    )
    logger.info(f"Selected backend: '{selected_backend}'")

    # --- Prepare Inputs based on Selected Backend ---
    X_run = X_input
    Y_run = Y_input
    is_X_sparse = sps.issparse(X_input)
    is_Y_sparse = sps.issparse(Y_input)

    try:
        prep_start_time = time.time()
        if selected_backend == 'gsl_cython':
            if is_X_sparse:
                raise TypeError("GSL backend requires dense X input.") # Should have been caught by selection
            if is_Y_sparse:
                logger.warning("GSL backend requires dense Y. Densifying Y input...")
                Y_run = Y_input.toarray(order='C') # Densify Y for GSL C call
                logger.info(f"Densified Y for GSL backend (shape: {Y_run.shape}).")
            # Ensure X and Y (now dense) are correct type/layout
            X_run = np.require(X_run, dtype=np.float64, requirements=['C', 'A'])
            Y_run = np.require(Y_run, dtype=np.float64, requirements=['C', 'A'])

        elif selected_backend == 'mkl':
            if is_X_sparse:
                raise TypeError("MKL backend requires dense X input.") # Caught by selection, but safety
            # MKL Cython wrapper handles internal checks/conversions for Y
            # Ensure X is suitable here
            X_run = np.require(X_run, dtype=np.float64, requirements=['C', 'A'])
            # Y_run remains original (dense or sparse) - wrapper deals with it

        elif selected_backend == 'python':
             # Handles sparse X/Y internally, no specific prep needed here beyond initial conversion
             pass
        elif selected_backend == 'numba':
             if is_X_sparse:
                 raise TypeError("Numba backend requires dense X input.") # Caught by selection
             X_run = np.require(X_run, dtype=np.float64, requirements=['C', 'A'])
             # Y_run can be dense or sparse (wrapper/kernel handles it)
        elif selected_backend == 'gpu':
             # GPU wrapper handles transfers and densification internally
             pass
        else:
            raise ValueError(f"Internal error: Unexpected selected_backend '{selected_backend}' during preparation.")
        logger.debug(f"Input preparation for backend '{selected_backend}' took {time.time() - prep_start_time:.4f}s")

    except (MemoryError, TypeError) as prep_error:
         logger.error(f"Failed to prepare inputs for backend '{selected_backend}': {prep_error}", exc_info=True)
         raise RuntimeError(f"Input preparation failed for backend '{selected_backend}'.") from prep_error

    # --- Execute with Fallbacks ---
    result_dict = None
    final_method_used = "failed"
    errors = []
    available_backends = get_available_backends()
    user_specific = requested_method != 'auto' # Was a specific backend requested?

    # Current backend to try
    current_backend = selected_backend

    while result_dict is None: # Loop until success or all fallbacks fail
        logger.info(f"Attempting ridge regression with backend: '{current_backend}'")
        try:
            backend_func = get_backend_function(current_backend)
            # Prepare args - MKL/GSL wrappers expect lambda_val, n_rand directly
            func_kwargs = {'lambda_val': lambda_, 'n_rand': n_rand}
            # Add backend-specific thread args if needed (or rely on global/env settings)
            # Example: if current_backend == 'mkl': func_kwargs['mkl_threads'] = MKL_NUM_THREADS

            result_dict = backend_func(X_run, Y_run, **func_kwargs)
            final_method_used = result_dict.get('method_used', current_backend)
            logger.info(f"Backend '{current_backend}' SUCCEEDED.")
            break # Exit loop on success

        except FALLBACK_ERRORS as e:
            logger.warning(f"Backend '{current_backend}' failed: {type(e).__name__} - {e}")
            errors.append((current_backend, str(e)))

            # Check if fallback is allowed and possible
            if not BACKEND_FALLBACK_ENABLED or user_specific:
                 logger.error(f"Backend '{current_backend}' failed. Fallback is disabled or specific backend was requested.")
                 break # Exit loop, will raise error later

            # Determine next fallback backend
            fallback_order = [b for b in ['mkl', 'gpu', 'gsl_cython', 'numba', 'python'] if b != current_backend and is_backend_available(b)]
            logger.debug(f"Available fallbacks from '{current_backend}': {fallback_order}")

            next_backend = None
            # Find the first suitable fallback based on data constraints
            for fb in fallback_order:
                fb_compatible = True
                # Check t-test compatibility
                if n_rand == 0 and fb not in ['gsl_cython', 'python']: fb_compatible = False
                # Check sparse X compatibility
                if is_X_sparse and fb != 'python': fb_compatible = False
                # Check MKL t-test incompatibility
                if n_rand == 0 and fb == 'mkl': fb_compatible = False

                if fb_compatible:
                    next_backend = fb
                    break

            if next_backend:
                logger.info(f"--- Falling back to backend: '{next_backend}' ---")
                # Re-prepare inputs if necessary for the new backend
                # (Similar logic as initial preparation, but maybe simpler as X is now dense if not python)
                current_backend = next_backend
                X_run = X_input # Reset to original (or densified if GSL/MKL was tried)
                Y_run = Y_input # Reset to original
                is_X_sparse = sps.issparse(X_run) # Re-check X sparsity (should be False unless initial was python)
                is_Y_sparse = sps.issparse(Y_run)

                try:
                    prep_start_time = time.time()
                    if current_backend == 'gsl_cython':
                        if is_X_sparse: raise TypeError("Fallback Error: GSL requires dense X.")
                        if is_Y_sparse:
                             logger.warning("GSL fallback requires dense Y. Densifying Y...")
                             Y_run = Y_run.toarray(order='C')
                        X_run = np.require(X_run, dtype=np.float64, requirements=['C', 'A'])
                        Y_run = np.require(Y_run, dtype=np.float64, requirements=['C', 'A'])
                    elif current_backend == 'mkl':
                         if is_X_sparse: raise TypeError("Fallback Error: MKL requires dense X.")
                         X_run = np.require(X_run, dtype=np.float64, requirements=['C', 'A'])
                         # Y_run passed as is to MKL wrapper
                    # Python, Numba, GPU handle prep internally or already done
                    logger.debug(f"Input re-preparation for backend '{current_backend}' took {time.time() - prep_start_time:.4f}s")
                except (MemoryError, TypeError) as prep_error:
                    logger.error(f"Failed to prepare inputs for fallback backend '{current_backend}': {prep_error}", exc_info=True)
                    errors.append((current_backend, f"Prep failed: {prep_error}"))
                    # Need to select *another* fallback if possible, or break
                    # For simplicity, we break here if re-prep fails
                    logger.error("Cannot continue fallback due to input preparation error.")
                    result_dict = None # Ensure we exit the loop and report failure
                    break # Exit the while loop

                # Continue to the next iteration of the while loop to try 'next_backend'
                continue

            else: # No suitable fallback found
                 logger.error("No suitable fallback backend found.")
                 break # Exit loop

        except Exception as e_unexpected:
            logger.error(f"Unexpected error during execution with backend '{current_backend}': {e_unexpected}", exc_info=True)
            errors.append((current_backend, f"Unexpected: {e_unexpected}"))
            break # Exit loop

    # --- Check Results and Process ---
    if result_dict is None or not all(k in result_dict for k in ['beta', 'se', 'zscore', 'pvalue']):
        error_str = "; ".join([f"{b}: {e}" for b, e in errors]) if errors else "Unknown reason."
        raise RuntimeError(f"Ridge regression failed. All attempted backends failed. Errors: {error_str}")

    # --- Adjust p-values for t-test alternative hypothesis ---
    # Only adjust if t-test was run (n_rand == 0) AND df is available
    df_ttest = result_dict.get('df', None) # Get df if returned (GSL/Python t-test should return it)

    if n_rand == 0 and df_ttest is not None and not np.isnan(df_ttest):
        logger.debug(f"Adjusting t-test p-values for alternative='{alternative}' using df={df_ttest:.2f}...")
        pvalue_ttest = result_dict['pvalue']
        zscore_ttest = result_dict['zscore'] # t-statistic

        try:
            if alternative == "greater":
                pvalue_adjusted = stats.t.sf(zscore_ttest, df=df_ttest)
            elif alternative == "less":
                pvalue_adjusted = stats.t.cdf(zscore_ttest, df=df_ttest)
            else:  # two-sided
                pvalue_adjusted = 2 * stats.t.sf(np.abs(zscore_ttest), df=df_ttest)

            result_dict['pvalue'] = np.clip(np.nan_to_num(pvalue_adjusted, nan=1.0), 0.0, 1.0)
            logger.debug("P-value adjustment complete.")
        except Exception as pval_err:
             logger.error(f"Error adjusting t-test p-values: {pval_err}. Returning original p-values.", exc_info=True)
             # Keep original pvalues if adjustment fails
    elif n_rand == 0 and (df_ttest is None or np.isnan(df_ttest)):
        logger.warning("T-test performed but degrees of freedom (df) not available/valid. Cannot adjust p-values for alternative hypothesis.")

    # --- Final Validation of Results ---
    essential_keys = ['beta', 'se', 'zscore', 'pvalue']
    if not all(key in result_dict and isinstance(result_dict[key], np.ndarray) for key in essential_keys):
        missing = [k for k in essential_keys if k not in result_dict or not isinstance(result_dict[k], np.ndarray)]
        raise RuntimeError(f"Backend '{final_method_used}' returned incomplete/invalid results (Missing/Invalid: {missing})")

    final_n_features = X_input.shape[1]
    final_n_samples = Y_input.shape[1]
    expected_shape = (final_n_features, final_n_samples)

    for key in essential_keys:
        if result_dict[key].shape != expected_shape:
             raise ValueError(f"Backend '{final_method_used}' returned key '{key}' with incorrect shape. Got {result_dict[key].shape}, expected {expected_shape}")

    # --- Add Execution Metadata ---
    execution_time = time.time() - start_time
    result_dict['execution_time'] = execution_time # Total time for this function call
    result_dict['method_used'] = final_method_used # Final method that produced result
    if 'peak_gpu_pool_mb' not in result_dict: result_dict['peak_gpu_pool_mb'] = None
    # Keep 'df' in dict if it was added by the backend

    # --- Cleanup ---
    del X_input, Y_input, X_run, Y_run # Clean up original and potentially modified arrays
    gc.collect()

    logger.info(f"Ridge completed successfully in {execution_time:.2f}s using method: '{final_method_used}'")
    return result_dict