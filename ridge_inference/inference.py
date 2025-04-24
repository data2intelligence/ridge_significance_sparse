# ridge_inference/inference.py

import numpy as np
import pandas as pd
import logging
import time
from scipy import sparse as sps
from .utils import scale_dataframe_columns
from .loaders import load_signature_matrix
import warnings
import gc

# Import main ridge function and batch function
from .ridge import ridge, FALLBACK_ERRORS as RIDGE_FALLBACK_ERRORS
try:
    from .batch import ridge_batch, FALLBACK_ERRORS as BATCH_FALLBACK_ERRORS
    BATCH_AVAILABLE = True
    # Combine errors that might occur in either standard or batch mode
    FALLBACK_ERRORS = tuple(set(RIDGE_FALLBACK_ERRORS + BATCH_FALLBACK_ERRORS))
except ImportError:
    logging.warning("ridge_batch function not found, batch processing disabled.")
    BATCH_AVAILABLE = False
    FALLBACK_ERRORS = RIDGE_FALLBACK_ERRORS # Use only errors from ridge.py
    # Define a dummy batch function if unavailable
    def ridge_batch(*args, **kwargs):
        raise NotImplementedError("ridge_batch function is unavailable.")


# --- Import backend availability flags ---
try:
    # Import flags set in backend_selection based on actual module imports
    from .backend_selection import NUMBA_AVAILABLE, CUPY_AVAILABLE, MKL_AVAILABLE, GSL_CYTHON_AVAILABLE, PYTHON_AVAILABLE
except ImportError as e:
    logging.error(f"Failed to import backend availability flags from backend_selection: {e}")
    NUMBA_AVAILABLE, CUPY_AVAILABLE, MKL_AVAILABLE, GSL_CYTHON_AVAILABLE, PYTHON_AVAILABLE = False, False, False, False, True

logger = logging.getLogger(__name__)

# --- Utility for checking Y sparsity benefit ---
def _is_Y_sparse_beneficial_for_cpu(Y_np, threshold=0.1, min_elements=1e6):
    """Checks if Y matrix *might* benefit from sparse format for CPU memory."""
    if sps.issparse(Y_np): return True
    if not isinstance(Y_np, np.ndarray) or Y_np.size < min_elements or Y_np.size == 0: return False
    try:
        # Optimization: Check a sample if Y is huge
        sample_size = 1000000
        if Y_np.size > sample_size * 10: # Only sample if significantly larger
             y_flat = Y_np.ravel()
             np.random.shuffle(y_flat)
             sampled_y = y_flat[:sample_size]
             non_zeros = np.count_nonzero(sampled_y)
             density = non_zeros / sample_size
             logger.debug(f"Y density (sampled {sample_size}): nnz={non_zeros}, density={density:.4f}")
        else:
             non_zeros = np.count_nonzero(Y_np)
             density = non_zeros / Y_np.size
             logger.debug(f"Y density (full): size={Y_np.size}, nnz={non_zeros}, density={density:.4f}")

        is_beneficial = density < threshold
        logger.debug(f" -> beneficial={is_beneficial} (threshold={threshold})")
        return is_beneficial
    except MemoryError: return False # Cannot even count non-zeros
    except Exception as e: logger.warning(f"Error checking Y density ({e}), assuming dense."); return False


# --- Main Inference Function ---
def secact_inference(
    expr_data,
    sig_matrix="SecAct",
    lambda_val=5e5,
    n_rand=1000,
    method="auto",
    add_background=False,
    scale_method=None,
    epsilon_scale=1e-8,
    batch_size=None,
    batch_threshold=50000,
    verbose=0,
    alternative="two-sided" # Add alternative here
    ):
    """
    Run SecAct inference on expression data.

    Parameters
    ----------
    expr_data : pandas.DataFrame (genes x samples)
    sig_matrix : str or pandas.DataFrame (genes x features)
        "SecAct", "CytoSig", or DataFrame.
    lambda_val : float
    n_rand : int (0 for t-test)
    method : str ('auto', 'python', 'mkl', 'gsl_cython', 'numba', 'gpu')
    add_background : bool
    scale_method : {None, 'column', 'global'}
    epsilon_scale : float
    batch_size : int, optional
        Batch size for samples (Y columns). Auto-estimated if None and samples > threshold.
    batch_threshold : int
        Sample count above which batching is attempted (if n_rand>0 and supported backend).
    verbose : int
    alternative : str, optional
        Alternative hypothesis for t-test ('two-sided', 'less', 'greater').
        Default: "two-sided". Only used if n_rand=0.

    Returns
    -------
    dict
        Results: 'beta', 'se', 'zscore', 'pvalue' (DataFrames), metadata.
    """
    start_time = time.time()
    logger.info("Starting SecAct inference")
    log_params = locals() # Capture all local variables as parameters
    del log_params['expr_data'], log_params['sig_matrix'] # Don't log potentially large data
    logger.info(f"Parameters: {log_params}")

    # --- Input Validation ---
    if not isinstance(expr_data, pd.DataFrame): raise TypeError("expr_data must be a pandas DataFrame")
    if not isinstance(sig_matrix, (str, pd.DataFrame)): raise TypeError("sig_matrix must be a string or pandas DataFrame")
    if scale_method not in [None, 'column', 'global']: raise ValueError("scale_method must be None, 'column', or 'global'")
    if method is None: method = "auto"
    if alternative not in ['two-sided', 'less', 'greater']: raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")

    # --- Load Data & Filter ---
    if isinstance(sig_matrix, str):
        logger.info(f"Loading signature matrix: {sig_matrix}")
        sig_df_raw = load_signature_matrix(sig_matrix)
    else:
        sig_df_raw = sig_matrix.copy()

    expr_df_raw = expr_data.copy()
    try: # Ensure string indices for matching
        expr_df_raw.index = expr_df_raw.index.astype(str)
        sig_df_raw.index = sig_df_raw.index.astype(str)
    except Exception as e: logger.warning(f"Could not ensure string indices: {e}")

    common_genes = expr_df_raw.index.intersection(sig_df_raw.index)
    n_common = len(common_genes)
    logger.info(f"Found {n_common} common genes.")
    if n_common < 10: warnings.warn(f"Only {n_common} common genes found.", RuntimeWarning)
    if n_common == 0: raise ValueError("No common genes found.")

    Y_df = expr_df_raw.loc[common_genes].copy()
    X_df = sig_df_raw.loc[common_genes].copy()
    logger.info(f"Data shapes after filtering: Y={Y_df.shape}, X={X_df.shape}")
    del expr_df_raw, sig_df_raw; gc.collect()

    # --- Preprocessing ---
    if add_background:
        logger.info("Preprocessing: Adding background column to X...")
        if 'background' in X_df.columns: logger.warning("Overwriting existing 'background' column.")
        # Calculate background based on the *filtered* Y_df
        X_df['background'] = Y_df.mean(axis=1)
        logger.info(f"Shape of X after adding background: {X_df.shape}")

    feature_names = X_df.columns.tolist(); sample_names = Y_df.columns.tolist()
    n_samples = Y_df.shape[1]

    if scale_method == 'column':
        logger.info("Preprocessing: Performing internal COLUMN-WISE scaling...")
        X_df = scale_dataframe_columns(X_df, epsilon=epsilon_scale)
        Y_df = scale_dataframe_columns(Y_df, epsilon=epsilon_scale)
    elif scale_method == 'global':
        logger.info("Preprocessing: Performing internal GLOBAL scaling...")
        try:
            Y_vals, X_vals = Y_df.values, X_df.values
            Y_m, Y_s = Y_vals.mean(), Y_vals.std(); X_m, X_s = X_vals.mean(), X_vals.std()
            if Y_s < epsilon_scale: warnings.warn(f"Global std Y ({Y_s:.2e}) near zero.", RuntimeWarning)
            if X_s < epsilon_scale: warnings.warn(f"Global std X ({X_s:.2e}) near zero.", RuntimeWarning)
            # Use epsilon in division
            Y_np = (Y_vals - Y_m) / (Y_s + epsilon_scale); X_np = (X_vals - X_m) / (X_s + epsilon_scale)
            # Fill NaNs/Infs resulting from scaling (e.g., if std was near zero)
            np.nan_to_num(Y_np, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            np.nan_to_num(X_np, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            Y_df = pd.DataFrame(Y_np, index=Y_df.index, columns=Y_df.columns)
            X_df = pd.DataFrame(X_np, index=X_df.index, columns=X_df.columns)
            del Y_vals, X_vals, Y_np, X_np; gc.collect()
        except MemoryError: logger.error("MemoryError during global scaling.", exc_info=True); raise
    elif scale_method is None: logger.info("Preprocessing: No internal scaling.")

    # --- Prepare for Ridge ---
    logger.info("Converting final DataFrames to NumPy/Sparse arrays...")
    try: # X must be dense for T matrix calculation
        X_input = np.require(X_df.values, dtype=np.float64, requirements=['C', 'A'])
    except Exception as e: raise TypeError(f"Could not convert X to NumPy array: {e}") from e

    Y_input = None; is_Y_sparse_internal = False
    try:
        if _is_Y_sparse_beneficial_for_cpu(Y_df.values): # Check original Y_df values for sparsity
            logger.info("Converting Y to sparse CSR...")
            Y_input = sps.csr_matrix(Y_df.values, dtype=np.float64); is_Y_sparse_internal = True
        else:
            logger.info("Keeping Y as dense NumPy array.")
            Y_input = np.require(Y_df.values, dtype=np.float64, requirements=['C', 'A']); is_Y_sparse_internal = False
    except Exception as e: raise TypeError(f"Could not convert Y to NumPy/Sparse array: {e}") from e
    del X_df, Y_df; gc.collect()

    # --- Determine Final Method and Check Compatibility ---
    final_method = method.lower()
    logger.info(f"Passing method='{final_method}' to ridge function.")

    # --- Determine if batch processing should be used ---
    use_batches = False
    actual_batch_size = batch_size
    if BATCH_AVAILABLE and n_samples > batch_threshold and n_rand > 0:
        potentially_batched_methods = ['python', 'gpu']
        temp_method_selection = final_method
        if temp_method_selection == 'auto':
            # Simplified auto-select for batch check: prefer GPU if available, else Python
            temp_method_selection = 'gpu' if CUPY_AVAILABLE else 'python'

        if temp_method_selection in potentially_batched_methods:
            use_batches = True
            if actual_batch_size is None:
                 try:
                     import psutil; available_mem_gb = psutil.virtual_memory().available / (1024**3)
                 except ImportError: available_mem_gb = 4 # Default assumption if psutil fails
                 # Dynamic estimation (adjust target_mem_gb and limits as needed)
                 target_mem_gb = max(0.5, min(available_mem_gb / 4, 2.0))
                 estimated_bs = int((target_mem_gb * 1024**3) / (X_input.shape[0] * 8)) if X_input.shape[0] > 0 else 5000
                 actual_batch_size = max(100, min(estimated_bs, 20000))
                 logger.info(f"Large dataset ({n_samples} > {batch_threshold}), using dynamic batch_size={actual_batch_size} for '{temp_method_selection}' backend.")
            else:
                 if not isinstance(actual_batch_size, int) or actual_batch_size <= 0: raise ValueError("batch_size must be positive integer.")
                 logger.info(f"Large dataset ({n_samples} > {batch_threshold}), using specified batch_size={actual_batch_size} for '{temp_method_selection}' backend.")
        else:
            logger.warning(f"Large dataset ({n_samples} > {batch_threshold}) but selected method '{temp_method_selection}' does not support batching. Using standard ridge.")
    elif not BATCH_AVAILABLE and n_samples > batch_threshold and n_rand > 0:
         logger.warning(f"Large dataset ({n_samples} > {batch_threshold}) but batch module unavailable. Using standard ridge.")
    elif n_rand == 0 and n_samples > batch_threshold:
         logger.info(f"Large dataset ({n_samples} > {batch_threshold}) but t-test (n_rand=0) does not use batching. Using standard ridge.")


    # --- Run Ridge Regression ---
    ridge_results = { # Default empty results
        'beta': None, 'se': None, 'zscore': None, 'pvalue': None,
        'method_used': f"{final_method} (pre-execution)", 'execution_time': None,
        'peak_gpu_pool_mb': None, 'df': None
    }
    final_method_used_info = "unknown"

    try:
        if use_batches and BATCH_AVAILABLE:
            logger.info(f"Running efficient ridge_batch with method='{final_method}' and batch_size={actual_batch_size}")
            ridge_results = ridge_batch(
                X_input, Y_input, lambda_val=lambda_val, n_rand=n_rand,
                method=final_method, batch_size=actual_batch_size, verbose=verbose
            )
            final_method_used_info = ridge_results.get('method_used', f"{final_method}_batched (reported missing)")
        else:
            logger.info(f"Running standard ridge function with method='{final_method}'")
            # *** FIX: Pass the 'alternative' parameter correctly ***
            ridge_results = ridge(
                X_input, Y_input, lambda_=lambda_val, n_rand=n_rand,
                method=final_method, verbose=verbose, alternative=alternative
            )
            # *******************************************************
            final_method_used_info = ridge_results.get('method_used', f"{final_method} (reported missing)")

        essential_keys = ['beta', 'se', 'zscore', 'pvalue']
        if not all(k in ridge_results and ridge_results[k] is not None for k in essential_keys):
             missing = [k for k in essential_keys if k not in ridge_results or ridge_results[k] is None]
             raise RuntimeError(f"Ridge call using '{final_method_used_info}' did not return valid results for keys: {missing}.")

    except FALLBACK_ERRORS as ridge_err:
         logger.error(f"Ridge execution failed: {type(ridge_err).__name__} - {ridge_err}", exc_info=(verbose>1))
         raise RuntimeError(f"Ridge failed using method '{final_method_used_info}'.") from ridge_err
    except Exception as ridge_err:
         logger.error(f"Ridge execution failed unexpectedly with method '{final_method_used_info}': {ridge_err}", exc_info=True)
         raise # Re-raise unexpected error

    # --- Format Results ---
    logger.info("Formatting results into DataFrames...")
    try:
        beta = ridge_results['beta']; se = ridge_results['se']
        zscore = ridge_results['zscore']; pvalue = ridge_results['pvalue']
        beta_df = pd.DataFrame(beta, index=feature_names, columns=sample_names)
        se_df = pd.DataFrame(se, index=feature_names, columns=sample_names)
        zscore_df = pd.DataFrame(zscore, index=feature_names, columns=sample_names)
        pvalue_df = pd.DataFrame(pvalue, index=feature_names, columns=sample_names)
    except Exception as df_err:
         raise RuntimeError("Failed to format results into pandas DataFrames.") from df_err

    execution_time = time.time() - start_time
    logger.info(f"SecAct inference completed in {execution_time:.2f} seconds using method: {final_method_used_info}")

    final_results = {
        'beta': beta_df, 'se': se_df, 'zscore': zscore_df, 'pvalue': pvalue_df,
        'method': final_method_used_info, 'execution_time': execution_time,
        'batched': use_batches, 'batch_size': actual_batch_size if use_batches else None,
        'peak_gpu_pool_mb': ridge_results.get('peak_gpu_pool_mb'),
        'df': ridge_results.get('df') # Propagate df if available
    }
    # Add internal execution time if reported by ridge/batch
    internal_exec_time = ridge_results.get('execution_time') # ridge/batch report their own time
    if internal_exec_time is not None: final_results['internal_ridge_time'] = internal_exec_time

    return final_results


# --- secact_activity_inference (Wrapper with more preprocessing options) ---
def secact_activity_inference(
    input_profile,
    input_profile_control=None,
    is_differential=False,
    is_paired=False,
    is_single_sample_level=False,
    sig_matrix="SecAct",
    lambda_val=5e5,
    n_rand=1000,
    sig_filter=False,
    method="auto",
    add_background=False,
    scale_method=None,
    batch_size=None,
    batch_threshold=50000,
    verbose=0,
    alternative="two-sided" # Add alternative here too
):
    """
    Infer signaling activity, preparing target expression data (Y) first.

    Wraps `secact_inference`, handling differential calculations before inference.

    Parameters as described in `secact_inference` docstring, plus:
    input_profile : pd.DataFrame
        Expression profile (genes x samples).
    input_profile_control : pd.DataFrame, optional
        Control expression profile (genes x samples).
    is_differential : bool
        If True, input_profile is already the differential profile Y.
    is_paired : bool
        If True and control provided, perform paired subtraction.
    is_single_sample_level : bool
        If True, keep individual sample differential profiles; otherwise average.
    sig_filter : bool
        If True, filter signature matrix rows to match common genes with Y.
    alternative : str
        Alternative hypothesis for t-test.
    """
    logger.info("Starting secact_activity_inference wrapper...")
    if method is None: method = "auto"

    # --- Input Validation ---
    if not isinstance(input_profile, pd.DataFrame): raise TypeError("input_profile must be a pandas DataFrame.")
    if not isinstance(input_profile.index, pd.Index): raise TypeError("input_profile must have an index (genes).")

    # --- Prepare Differential Profile Y ---
    if is_differential:
        logger.info("Input profile is assumed to be differential. Skipping Y calculation.")
        Y_df = input_profile.copy()
        if Y_df.shape[1] == 1: Y_df.columns = ["Change"]
    else:
        logger.info("Calculating differential profile Y from input expression...")
        input_profile_proc = input_profile.copy()
        input_profile_proc.index = input_profile_proc.index.astype(str)

        if input_profile_control is None:
            logger.info("Calculating change from mean profile (no control provided).")
            row_means = input_profile_proc.mean(axis=1)
            Y_df = input_profile_proc.subtract(row_means.reindex(input_profile_proc.index), axis=0)
        else: # Control provided
            if not isinstance(input_profile_control, pd.DataFrame): raise TypeError("input_profile_control must be DataFrame.")
            if not isinstance(input_profile_control.index, pd.Index): raise TypeError("input_profile_control must have index.")
            input_control_proc = input_profile_control.copy()
            input_control_proc.index = input_control_proc.index.astype(str)

            common_genes_ctrl = input_profile_proc.index.intersection(input_control_proc.index)
            if len(common_genes_ctrl) == 0: raise ValueError("No common genes between input and control.")
            if len(common_genes_ctrl) < input_profile_proc.shape[0] or len(common_genes_ctrl) < input_control_proc.shape[0]:
                logger.warning(f"Input ({input_profile_proc.shape[0]}) and control ({input_control_proc.shape[0]}) using intersection of {len(common_genes_ctrl)} genes.")
            input_filt, control_filt = input_profile_proc.loc[common_genes_ctrl], input_control_proc.loc[common_genes_ctrl]

            if is_paired:
                logger.info("Calculating paired differences.")
                common_cols = sorted(list(set(input_filt.columns) & set(control_filt.columns)))
                if not common_cols: raise ValueError("No common samples for paired analysis.")
                logger.info(f"Found {len(common_cols)} paired samples.")
                Y_df = input_filt[common_cols] - control_filt[common_cols]
            else: # Unpaired
                logger.info("Calculating difference from mean of control samples.")
                control_means = control_filt.mean(axis=1)
                Y_df = input_filt.subtract(control_means.reindex(input_filt.index), axis=0)

        if not is_single_sample_level:
            logger.info("Averaging differential profiles across samples.")
            Y_df = pd.DataFrame({"Change": Y_df.mean(axis=1)})
        else: logger.info("Keeping single sample level differential profiles.")

        if Y_df.isnull().values.any():
             nan_count = Y_df.isnull().sum().sum()
             logger.warning(f"{nan_count} NaNs in calculated Y profile. Filling with 0.")
             Y_df = Y_df.fillna(0)
        logger.info(f"Calculated differential Y shape: {Y_df.shape}")

    # --- Load Signature Matrix & Optional Filtering ---
    X_arg = sig_matrix
    if sig_filter:
        logger.info("Signature filtering requested.")
        if isinstance(sig_matrix, str):
             try: X_sig_raw = load_signature_matrix(sig_matrix)
             except Exception as e: raise ValueError(f"Could not load sig matrix '{sig_matrix}' for filtering") from e
        elif isinstance(sig_matrix, pd.DataFrame): X_sig_raw = sig_matrix
        else: raise TypeError("sig_matrix must be str or DataFrame for filtering.")

        X_sig_raw.index = X_sig_raw.index.astype(str); Y_df.index = Y_df.index.astype(str)
        common_genes_sig_Y = X_sig_raw.index.intersection(Y_df.index)
        n_common_sig = len(common_genes_sig_Y)
        logger.info(f"Found {n_common_sig} genes common to signature and differential Y.")
        if n_common_sig == 0: raise ValueError("Signature has no genes in common with differential profile Y.")
        if n_common_sig < X_sig_raw.shape[0]:
             logger.info(f"Filtering signature rows from {X_sig_raw.shape[0]} to {n_common_sig}.")
             X_arg = X_sig_raw.loc[common_genes_sig_Y].copy() # Pass filtered DataFrame
        else: logger.info("All signature genes present in Y. No filtering needed."); X_arg = X_sig_raw
        if 'X_sig_raw' in locals(): del X_sig_raw; gc.collect()

    # --- Run Core Inference ---
    logger.info("Calling secact_inference with prepared Y and X...")
    # *** FIX: Pass 'alternative' down to secact_inference ***
    result = secact_inference(
        expr_data=Y_df,         # Pass prepared Y
        sig_matrix=X_arg,       # Pass potentially filtered X
        lambda_val=lambda_val, n_rand=n_rand, method=method,
        add_background=add_background, scale_method=scale_method,
        batch_size=batch_size, batch_threshold=batch_threshold, verbose=verbose,
        alternative=alternative # Pass it down
    )
    # ********************************************************

    logger.info("secact_activity_inference finished.")
    return result