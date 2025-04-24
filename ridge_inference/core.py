# ridge_inference/core.py

"""
Core functions for Ridge Inference package.

Contains Python, Numba, CuPy implementations.
"""

import os
import gc
import time
import logging
import warnings

import numpy as np
from scipy import linalg
from scipy import sparse as sps
from scipy.sparse import linalg as sps_linalg
from scipy import stats

logger = logging.getLogger(__name__)

# --- Numerical tolerance ---
EPS = 1e-12
# --- SEED constant for reproducibility ---
SEED = 42

# --- Numba Setup ---
try:
    import numba
    from numba import jit, prange, njit # njit added
    NUMBA_AVAILABLE = True
    logger.info(f"Numba {numba.__version__} available")
except ImportError:
    NUMBA_AVAILABLE = False
    logger.info("Numba not available.")
    def jit(signature_or_function=None, **kwargs):
        if callable(signature_or_function): return signature_or_function
        else: return lambda func: func
    njit = jit
    prange = range

# --- CuPy Setup ---
CUPY_AVAILABLE = False
cp = None
cusps = None
cpsp = None
# Define dummy exceptions globally first
class CuSparseError(Exception): pass
class LinAlgError(np.linalg.LinAlgError): pass
class OutOfMemoryError(MemoryError): pass

try:
    if os.environ.get('RIDGE_INFERENCE_DISABLE_GPU', '0') == '1':
        raise ImportError("GPU disabled by RIDGE_INFERENCE_DISABLE_GPU=1")
    import cupy as cp
    _ = cp.array([1]) # Check device access
    cp.cuda.Device().synchronize()
    CUPY_AVAILABLE = True
    import cupyx.scipy.sparse as cusps
    from cupyx.scipy import sparse as cpsp
    logger.info(f"CuPy {cp.__version__} available.")
    try: # Import specific CuPy errors
        from cupy.cuda import cusparse as cp_cusparse_module
        CuSparseError = cp_cusparse_module.CuSparseError
    except ImportError:
        try: from cupyx.scipy.sparse import CuSparseError as CuPySparseErrorClass; CuSparseError = CuPySparseErrorClass
        except ImportError: logger.warning("Could not import CuSparseError from CuPy.")
    LinAlgError = cp.linalg.LinAlgError
    OutOfMemoryError = cp.cuda.memory.OutOfMemoryError

except ImportError as e:
    logger.info(f"CuPy not available or disabled: {e}")
    CUPY_AVAILABLE = False; cp = None; cusps = None; cpsp = None
except Exception as e:
    logger.warning(f"CuPy initialization failed: {type(e).__name__} - {e}. GPU disabled.")
    CUPY_AVAILABLE = False; cp = None; cusps = None; cpsp = None


# ============================================
#   NumPy Backend Implementations
# ============================================
# (_calculate_T_numpy, _calculate_beta_numpy, _perform_permutation_numpy,
#  _perform_ttest_numpy, ridge_regression_numpy, ridge_sparse_iterative)
# Keep these exactly as they were in the previous version,
# ensuring _perform_ttest_numpy returns df.

def _calculate_T_numpy(X, lambda_val):
    """
    Calculates T = (X'X + lambda*I)^-1 * X' using NumPy.
    Assumes X is a dense NumPy array.
    """
    if sps.issparse(X):
        raise TypeError("_calculate_T_numpy requires dense X input.")
    if not isinstance(X, np.ndarray):
        X = np.asarray(X)

    n_genes, n_features = X.shape
    log_prefix = "NumPy Dense X:"
    logger.debug(f"{log_prefix} Calculating T matrix...")

    X_T = X.T
    logger.debug(f"{log_prefix} Calculating XtX = X.T @ X...")
    try:
        XtX = X_T @ X
    except Exception as e:
        raise ArithmeticError(f"{log_prefix} Failed calculating X'X: {e}") from e

    logger.debug(f"{log_prefix} Calculating A = XtX + lambda*I...")
    A = XtX + lambda_val * np.identity(n_features, dtype=XtX.dtype)
    del XtX

    rhs_T = X_T
    if not isinstance(rhs_T, np.ndarray): rhs_T = np.asarray(rhs_T)

    T = None
    A_pinv = None # Initialize pinv variable
    L = None # Initialize Cholesky factor
    logger.debug(f"{log_prefix} Solving for T matrix...")
    try:
        try:
            L = linalg.cholesky(A, lower=True)
            T = linalg.cho_solve((L, True), rhs_T)
            logger.debug(f"{log_prefix} Dense Cholesky solve successful.")
        except np.linalg.LinAlgError as e1_d:
            logger.warning(f"{log_prefix} Dense Cholesky failed ({e1_d}), trying standard solve...")
            try:
                T = linalg.solve(A, rhs_T, assume_a='pos')
                logger.debug(f"{log_prefix} Dense standard solve successful.")
            except np.linalg.LinAlgError as e2_d:
                logger.warning(f"{log_prefix} Dense standard solve failed ({e2_d}), trying pseudo-inverse...")
                try:
                    A_pinv = linalg.pinv(A)
                    T = A_pinv @ rhs_T
                    logger.debug(f"{log_prefix} Dense pseudo-inverse solve successful.")
                except np.linalg.LinAlgError as e3_d:
                    raise ArithmeticError(f"Dense matrix pseudo-inversion failed: {e3_d}") from e3_d
                except Exception as e_pinv:
                    raise ArithmeticError(f"Dense pseudo-inverse calculation failed: {e_pinv}") from e_pinv
        finally:
             if L is not None: del L

    except Exception as e:
         logger.error(f"{log_prefix} Unexpected error during T calculation: {e}", exc_info=True)
         raise ArithmeticError("Unexpected error during NumPy T calculation") from e
    finally:
         if 'A' in locals(): del A
         if 'rhs_T' in locals(): del rhs_T
         if A_pinv is not None: del A_pinv
         gc.collect()

    if T is None: raise ArithmeticError("Failed to solve for T matrix.")
    return T

def _calculate_beta_numpy(T, Y):
    """Calculates beta = T @ Y using NumPy, handling dense/sparse Y."""
    logger.debug("NumPy: Calculating beta = T @ Y...")
    if sps.issparse(Y):
        if not isinstance(T, np.ndarray): T = np.asarray(T)
        # Use CSR for efficient row slicing access in Y.T @ T.T
        Y_sp = Y.tocsr() if not isinstance(Y, sps.csr_matrix) else Y
        # Calculate (Y.T @ T.T).T which is equivalent to T @ Y
        beta = (Y_sp.T @ T.T).T
        logger.debug("NumPy: Calculated beta using sparse Y input.")
    else:
        if not isinstance(Y, np.ndarray): Y = np.asarray(Y)
        if not isinstance(T, np.ndarray): T = np.asarray(T)
        beta = T @ Y
        logger.debug("NumPy: Calculated beta using dense Y input.")

    if sps.issparse(beta):
        logger.warning("NumPy: Beta matrix calculation resulted in sparse, converting to dense.")
        beta = beta.toarray()
    elif not isinstance(beta, np.ndarray):
        beta = np.asarray(beta)
    return beta

def _perform_permutation_numpy(T, Y, beta, n_rand):
    """Performs permutation test using NumPy, given T and beta."""
    n_genes = Y.shape[0]
    n_features, n_outputs = beta.shape
    is_Y_sparse = sps.issparse(Y)
    logger.info(f"NumPy: Performing permutation test (nrand={n_rand}, Y is {'sparse' if is_Y_sparse else 'dense'})...")
    beta_rand_sum = np.zeros_like(beta)
    beta_rand_sq_sum = np.zeros_like(beta)
    pvalue_counts = np.zeros_like(beta)
    abs_beta = np.abs(beta)
    rng = np.random.default_rng(SEED)
    indices = np.arange(n_genes)

    if not isinstance(T, np.ndarray): T = np.asarray(T)

    Y_sp = None
    if is_Y_sparse:
        Y_sp = Y.tocsr() # Ensure CSR

    T_perm = None
    for i_perm in range(n_rand):
        perm_idx = rng.permutation(indices)
        T_perm = T[:, perm_idx]

        if is_Y_sparse:
             beta_rand = (Y_sp.T @ T_perm.T).T # Use T @ Y calculation here too
        else:
             # Ensure dense Y is numpy array
             if not isinstance(Y, np.ndarray): Y = np.asarray(Y)
             beta_rand = T_perm @ Y

        if sps.issparse(beta_rand): beta_rand = beta_rand.toarray()
        elif not isinstance(beta_rand, np.ndarray): beta_rand = np.asarray(beta_rand)

        beta_rand_sum += beta_rand
        beta_rand_sq_sum += beta_rand**2
        pvalue_counts += (np.abs(beta_rand) >= abs_beta - EPS) # Added EPS tolerance
        if (i_perm + 1) % 500 == 0 and logger.level <= logging.DEBUG:
             logger.debug(f"NumPy Permutation: {i_perm+1}/{n_rand}")

    logger.debug("NumPy Permutation: Finalizing statistics...")
    beta_rand_mean = beta_rand_sum / n_rand
    beta_rand_var = np.maximum((beta_rand_sq_sum / n_rand) - (beta_rand_mean**2), 0)
    se = np.sqrt(beta_rand_var)
    pvalue = (pvalue_counts + 1.0) / (n_rand + 1.0)
    pvalue = np.minimum(pvalue, 1.0)

    zscore = np.zeros_like(beta)
    valid_se_mask = se > EPS
    zscore[valid_se_mask] = (beta[valid_se_mask] - beta_rand_mean[valid_se_mask]) / se[valid_se_mask]
    zero_se_idx = ~valid_se_mask
    zscore[zero_se_idx] = np.where(np.abs(beta[zero_se_idx] - beta_rand_mean[zero_se_idx]) < EPS,
                                   0.0,
                                   np.inf * np.sign(beta[zero_se_idx] - beta_rand_mean[zero_se_idx]))

    num_zero_se = np.sum(zero_se_idx)
    if num_zero_se > 0: logger.warning(f"NumPy Permutation: {num_zero_se} SEs were near zero (set z=0 or +/-inf).")

    if T_perm is not None: del T_perm
    if Y_sp is not None: del Y_sp
    del beta_rand_sum, beta_rand_sq_sum, pvalue_counts, abs_beta, beta_rand_mean, beta_rand_var
    gc.collect()
    return se, zscore, pvalue

def _perform_ttest_numpy(X, Y, T, beta, lambda_val):
    """Performs t-test using NumPy, returns df."""
    n_genes, n_features = X.shape
    n_outputs = Y.shape[1]
    is_Y_sparse = sps.issparse(Y)

    if sps.issparse(X): raise TypeError("T-test requires dense X for NumPy backend.")
    X = np.require(X, dtype=np.float64, requirements=['C', 'A'])
    if not isinstance(T, np.ndarray): T = np.require(T, dtype=np.float64, requirements=['C', 'A'])
    if not isinstance(beta, np.ndarray): beta = np.require(beta, dtype=np.float64, requirements=['C', 'A'])

    logger.info("NumPy: Performing t-test (n_rand=0)...")
    se = np.zeros_like(beta)
    zscore = np.zeros_like(beta)
    pvalue = np.zeros_like(beta)
    df = np.nan
    XtX_inv = None; diag_XtX_inv = None; T_squared_row_sum = None

    logger.debug("NumPy T-test: Calculating predicted Y_hat = X @ beta...")
    Y_hat = X @ beta

    if abs(lambda_val) < EPS:  # OLS
        logger.debug("NumPy T-test: Calculating OLS stats...")
        df = float(max(1.0, n_genes - n_features)) # Ensure df >= 1
        try:
            XtX_local = X.T @ X
            XtX_inv = linalg.pinv(XtX_local)
            del XtX_local
        except np.linalg.LinAlgError as e: raise ArithmeticError(f"OLS pinv(X'X) failed: {e}") from e
        diag_XtX_inv = np.maximum(np.diag(XtX_inv), 0) # Ensure non-negative variance contribution
    else:  # Ridge
        logger.debug("NumPy T-test: Calculating Ridge stats...")
        H_trace = np.trace(X @ T)
        df = float(max(1.0, n_genes - H_trace)) # Ensure df >= 1
        logger.debug(f"NumPy T-test: Effective df = {df:.4f} (n={n_genes}, trace(H)={H_trace:.4f})")
        T_squared_row_sum = np.sum(T**2, axis=1)

    sigma2 = np.zeros(n_outputs)
    if is_Y_sparse:
        Y_csr = Y.tocsr()
        logger.debug("NumPy T-test: Calculating sigma^2 column-wise for sparse Y...")
        for j in range(n_outputs):
            y_j = Y_csr[:, j].toarray().ravel() # Dense column
            y_hat_j = Y_hat[:, j]
            sum_sq_resid = np.sum((y_j - y_hat_j)**2)
            sigma2[j] = max(sum_sq_resid / df, 0) if df > 0 else np.nan
    else:
        logger.debug("NumPy T-test: Calculating sigma^2 for dense Y...")
        Y_dense = np.require(Y, dtype=np.float64, requirements=['A'])
        residuals = Y_dense - Y_hat
        sum_sq_residuals = np.sum(residuals**2, axis=0)
        sigma2 = np.maximum(sum_sq_residuals / df, 0) if df > 0 else np.full(n_outputs, np.nan)
        del residuals, Y_dense

    logger.debug("NumPy T-test: Calculating standard errors (SE)...")
    if abs(lambda_val) < EPS:
        se = np.sqrt(np.maximum(np.outer(diag_XtX_inv, sigma2), 0))
    else:
        se = np.sqrt(np.maximum(np.outer(T_squared_row_sum, sigma2), 0))

    logger.debug("NumPy T-test: Calculating t-scores and p-values...")
    valid_se_mask = (se > EPS) & ~np.isnan(se)
    zscore[valid_se_mask] = beta[valid_se_mask] / se[valid_se_mask]
    zero_se_idx = ~valid_se_mask
    zscore[zero_se_idx] = np.where(np.abs(beta[zero_se_idx]) < EPS, # Check if beta is also zero
                                   0.0,
                                   np.inf * np.sign(beta[zero_se_idx]))

    num_zero_se = np.sum(zero_se_idx)
    if num_zero_se > 0: logger.warning(f"NumPy T-test: {num_zero_se} SEs were near zero or NaN (set t-score=0 or +/-inf).")

    if np.isnan(df):
        logger.error("NumPy T-test: Degrees of freedom is NaN. P-values set to NaN.")
        pvalue.fill(np.nan)
    elif df <= 0:
        logger.error(f"NumPy T-test: Degrees of freedom is non-positive ({df}). P-values set to 1.")
        pvalue.fill(1.0)
    else:
        pvalue = 2 * stats.t.sf(np.abs(zscore), df)
        pvalue = np.minimum(np.nan_to_num(pvalue, nan=1.0), 1.0)

    del Y_hat, sigma2
    if XtX_inv is not None: del XtX_inv
    if diag_XtX_inv is not None: del diag_XtX_inv
    if T_squared_row_sum is not None: del T_squared_row_sum
    gc.collect()
    return se, zscore, pvalue, df # Return df

def ridge_regression_numpy(X, Y, lambda_val=5e5, n_rand=1000):
    """Main NumPy backend function. Orchestrates T, beta, and test calculations."""
    logger.info("Starting NumPy/SciPy ridge regression calculation")
    start_time_np = time.time()
    is_X_sparse_orig = sps.issparse(X)

    if is_X_sparse_orig:
        if n_rand == 0:
            raise NotImplementedError("T-test (n_rand=0) is not supported for sparse X input.")
        logger.info("NumPy/SciPy backend: Detected sparse X, using iterative solver.")
        results_dict = ridge_sparse_iterative(X, Y, lambda_val, n_rand)
        results_dict['method_used'] = 'python_sparse_iterative'
    else:
        logger.info("NumPy/SciPy backend: Detected dense X, using direct solver.")
        X_dense = np.require(X, dtype=np.float64, requirements=['C', 'A'])

        T = _calculate_T_numpy(X_dense, lambda_val)
        beta = _calculate_beta_numpy(T, Y)
        df = np.nan # Default df for permutation test

        if n_rand > 0:
            logger.info("NumPy/SciPy backend: Performing permutation test for dense X.")
            se, zscore, pvalue = _perform_permutation_numpy(T, Y, beta, n_rand)
        else:
            logger.info("NumPy/SciPy backend: Performing t-test for dense X.")
            se, zscore, pvalue, df = _perform_ttest_numpy(X_dense, Y, T, beta, lambda_val)

        del T, X_dense; gc.collect()
        results_dict = {'beta': beta, 'se': se, 'zscore': zscore, 'pvalue': pvalue, 'method_used': 'python'}
        if n_rand == 0: results_dict['df'] = df # Add df if t-test

    np_execution_time = time.time() - start_time_np
    logger.info(f"NumPy/SciPy backend finished in {np_execution_time:.2f} seconds.")
    results_dict['execution_time'] = np_execution_time # Add execution time
    return results_dict

def ridge_sparse_iterative(X, Y, lambda_val, n_rand):
    """Ridge regression using iterative solver lsmr for sparse X."""
    if not sps.isspmatrix(X): raise TypeError("X must be sparse for ridge_sparse_iterative")
    if n_rand <= 0: raise ValueError("n_rand must be > 0 for ridge_sparse_iterative")
    if lambda_val < 0: lambda_val = 0.0

    start_time = time.time()
    n_genes, n_features = X.shape
    if not isinstance(X, (sps.csr_matrix, sps.csc_matrix)): X = X.tocsr()

    Y_is_sparse = sps.issparse(Y)
    if Y_is_sparse:
        n_samples = Y.shape[1]; Y_perm = Y.tocsr()
    else:
        Y_perm = np.require(Y, dtype=float, requirements=['C', 'A']); n_samples = Y.shape[1]

    beta = np.zeros((n_features, n_samples), dtype=float)
    damp = np.sqrt(lambda_val)

    logger.debug("Sparse Iterative: Calculating observed beta...")
    for j in range(n_samples):
        y_col = Y_perm[:, j].toarray().ravel() if Y_is_sparse else Y_perm[:, j]
        if Y_is_sparse and Y_perm[:,j].nnz == 0: y_col = np.zeros(n_genes)
        beta_j, istop, itn, *_ = sps_linalg.lsmr(X, y_col, damp=damp, show=False)
        beta[:, j] = beta_j
        if istop not in [0, 1, 2, 7]: logger.warning(f"lsmr convergence issue for observed column {j}: istop={istop}, itn={itn}")
    logger.debug("Sparse Iterative: Observed beta calculation complete.")

    logger.info(f"Sparse Iterative: Performing permutation test (nrand={n_rand})...")
    sum_b, sum_b2, count_ge = np.zeros_like(beta), np.zeros_like(beta), np.zeros_like(beta)
    abs_beta = np.abs(beta); rng = np.random.default_rng(SEED); indices = np.arange(n_genes)

    Y_shuffled = None
    for i in range(n_rand):
        perm_idx = rng.permutation(indices)
        Y_shuffled = Y_perm[perm_idx, :]
        for j in range(n_samples):
            ycol = Y_shuffled[:, j].toarray().ravel() if Y_is_sparse else Y_shuffled[:, j]
            if Y_is_sparse and Y_shuffled[:,j].nnz == 0: ycol = np.zeros(n_genes)
            beta_j_rand, *_ = sps_linalg.lsmr(X, ycol, damp=damp, show=False)
            sum_b[:, j]    += beta_j_rand
            sum_b2[:, j]   += beta_j_rand**2
            count_ge[:, j] += (np.abs(beta_j_rand) >= abs_beta[:, j] - EPS) # Add EPS tolerance
        if (i + 1) % 500 == 0 and logger.level <= logging.DEBUG: logger.debug(f"Sparse Permutation: {i+1}/{n_rand}")

    logger.debug("Sparse Iterative: Finalizing permutation statistics...")
    mean_b = sum_b / n_rand; var_b  = np.maximum((sum_b2 / n_rand) - mean_b**2, 0.0)
    se = np.sqrt(var_b); pvalue = np.minimum((count_ge + 1.0) / (n_rand + 1.0), 1.0)
    zscore = np.zeros_like(beta); mask = se > EPS
    zscore[mask] = (beta[mask] - mean_b[mask]) / se[mask]
    zero_se_idx = ~mask
    zscore[zero_se_idx] = np.where(np.abs(beta[zero_se_idx] - mean_b[zero_se_idx]) < EPS,
                                   0.0,
                                   np.inf * np.sign(beta[zero_se_idx] - mean_b[zero_se_idx]))

    zero_se_count = np.sum(zero_se_idx)
    if zero_se_count > 0: logger.warning(f"Sparse Iterative: {zero_se_count} coefficients with zero SE; z-score set to 0 or +/-inf")

    elap = time.time() - start_time
    logger.info(f"SciPy sparse iterative ridge completed in {elap:.2f}s")
    del Y_perm, sum_b, sum_b2, count_ge, abs_beta, mean_b, var_b, mask
    if Y_shuffled is not None: del Y_shuffled
    gc.collect()
    return {'beta': beta, 'se': se, 'zscore': zscore, 'pvalue': pvalue}


# ============================================
#  Numba Backend Implementations
# ============================================
# (_permutation_kernel_sparse_y [expects CSC], _permutation_kernel_dense_y,
#  ridge_regression_numba)
# Keep these exactly as they were in the previous version.

@njit(parallel=True, fastmath=True, cache=True)
def _permutation_kernel_sparse_y(T, Y_data, Y_indices, Y_indptr, beta_obs, n_rand):
    """Numba kernel for permutation test with sparse Y (CSC format expected)."""
    p, n = T.shape
    m = Y_indptr.shape[0] - 1
    se = np.empty_like(beta_obs, dtype=np.float64)
    z = np.empty_like(beta_obs, dtype=np.float64)
    pv = np.empty_like(beta_obs, dtype=np.float64)
    sum_b = np.zeros_like(beta_obs, dtype=np.float64)
    sum_b2 = np.zeros_like(beta_obs, dtype=np.float64)
    count_ge = np.zeros_like(beta_obs, dtype=np.float64)
    abs_obs = np.abs(beta_obs)
    idx_map = np.arange(n, dtype=np.int64)
    beta_rand_perm = np.empty_like(beta_obs)

    # Use np.random within njit context
    # Seed needs to be handled carefully in parallel contexts if strict reproducibility needed
    # np.random.seed(SEED) # Seeding inside kernel might affect parallel performance/correctness

    for perm in range(n_rand):
        # Shuffle in-place (safer for parallel if RNG state is tricky)
        np.random.shuffle(idx_map)

        # Parallelize the outer loop over columns of Y (samples)
        for j in prange(m): # Changed parallel loop
            start = Y_indptr[j]
            end = Y_indptr[j+1]
            # Inner loop over features (p) is typically smaller, better parallelize outer loop
            for r in range(p):
                acc = 0.0
                # This loop over non-zeros remains serial per (r,j) element
                for k in range(start, end):
                    gene_idx_in_Y = Y_indices[k] # Row index in CSC means gene index
                    y_val = Y_data[k]
                    # T-column permutation: Use permuted gene index to select column of T
                    permuted_gene_col = idx_map[gene_idx_in_Y]
                    # Bounds check isn't strictly necessary if idx_map is always 0..n-1
                    # if 0 <= permuted_gene_col < n:
                    acc += T[r, permuted_gene_col] * y_val
                beta_rand_perm[r, j] = acc

        # Accumulation needs to be done carefully if outer loop is parallel
        # For now, keep accumulation serial after parallel beta_rand calculation
        # If beta_rand calculation itself was parallelized, need atomic updates or reduction
        sum_b += beta_rand_perm
        sum_b2 += beta_rand_perm * beta_rand_perm
        count_ge += (np.abs(beta_rand_perm) >= abs_obs - EPS) # Add EPS tolerance

    inv_n_rand = 1.0 / n_rand
    mean_b = sum_b * inv_n_rand
    var_b  = sum_b2 * inv_n_rand - mean_b * mean_b

    # Parallelize the final calculation of se, z, pvalue
    for r in prange(p):
        for j in range(m):
            v = var_b[r, j]
            if v < 0.0 and v > -EPS: v = 0.0
            elif v < -EPS: v = np.nan # Indicate error state
            s = np.sqrt(v) # sqrt(nan) is nan
            se[r, j] = s
            if s > EPS: z[r, j] = (beta_obs[r, j] - mean_b[r, j]) / s
            elif not np.isnan(s): # SE is near zero
                 z[r,j] = 0.0 if abs(beta_obs[r,j] - mean_b[r,j]) < EPS else np.sign(beta_obs[r,j]-mean_b[r,j]) * np.inf
            else: # SE is NaN
                 z[r,j] = np.nan
            pval_raw = (count_ge[r, j] + 1.0) / (n_rand + 1.0)
            pv[r, j] = min(max(pval_raw, 0.0), 1.0)

    return se, z, pv

@njit(parallel=True, fastmath=True, cache=True)
def _permutation_kernel_dense_y(T, Y, beta_obs, n_rand):
    """Numba kernel for permutation test with dense Y."""
    p, n = T.shape
    _, m = Y.shape
    se = np.empty_like(beta_obs, dtype=np.float64)
    z = np.empty_like(beta_obs, dtype=np.float64)
    pv = np.empty_like(beta_obs, dtype=np.float64)
    sum_b = np.zeros_like(beta_obs, dtype=np.float64)
    sum_b2 = np.zeros_like(beta_obs, dtype=np.float64)
    count_ge = np.zeros_like(beta_obs, dtype=np.float64)
    abs_obs = np.abs(beta_obs)
    idx_map = np.arange(n, dtype=np.int64)
    beta_rand_perm = np.empty_like(beta_obs)

    # np.random.seed(SEED) # Avoid seeding inside parallel kernel

    for perm in range(n_rand):
        np.random.shuffle(idx_map)
        # Perform T[:, perm_idx] @ Y in parallel if possible
        # Slicing T creates a temporary copy which might be expensive
        # Explicit loop might be better for parallelization with prange
        T_perm = T[:, idx_map] # Potential large allocation if n is big
        beta_rand_perm = np.dot(T_perm, Y) # Matrix multiplication benefits from BLAS

        # Accumulation remains serial for simplicity, assumes beta_rand calc dominates
        sum_b += beta_rand_perm
        sum_b2 += beta_rand_perm * beta_rand_perm
        count_ge += (np.abs(beta_rand_perm) >= abs_obs - EPS) # Add EPS tolerance

    inv_n_rand = 1.0 / n_rand
    mean_b = sum_b * inv_n_rand
    var_b  = sum_b2 * inv_n_rand - mean_b * mean_b

    for r in prange(p): # Parallelize final stat calculation
        for j in range(m):
            v = var_b[r, j]
            if v < 0.0 and v > -EPS: v = 0.0
            elif v < -EPS: v = np.nan
            s = np.sqrt(v)
            se[r, j] = s
            if s > EPS: z[r, j] = (beta_obs[r, j] - mean_b[r, j]) / s
            elif not np.isnan(s): # SE near zero
                 z[r,j] = 0.0 if abs(beta_obs[r,j] - mean_b[r,j]) < EPS else np.sign(beta_obs[r,j]-mean_b[r,j]) * np.inf
            else: # SE is NaN
                 z[r,j] = np.nan
            pval_raw = (count_ge[r, j] + 1.0) / (n_rand + 1.0)
            pv[r, j] = min(max(pval_raw, 0.0), 1.0)

    return se, z, pv

def ridge_regression_numba(X, Y, lambda_val=5e5, n_rand=1000):
    """Hybrid: NumPy/SciPy for T matrix, Numba for permutation loop."""
    if not isinstance(X, np.ndarray) or sps.issparse(X):
        raise TypeError("Numba backend requires dense NumPy array for X.")
    if n_rand <= 0:
         raise NotImplementedError("Numba backend currently only supports permutation test (n_rand > 0).")

    is_Y_sparse = sps.issparse(Y)
    logger.info(f"Running Numba backend (NumPy T + Numba perm) with {'Sparse' if is_Y_sparse else 'Dense'} Y.")
    t0 = time.time()

    logger.debug("Numba backend: Calculating T matrix using NumPy...")
    Xc = np.require(X, dtype=np.float64, requirements=['C','A'])
    T = _calculate_T_numpy(Xc, lambda_val)

    logger.debug("Numba backend: Calculating observed beta...")
    beta_obs = _calculate_beta_numpy(T, Y)

    logger.debug("Numba backend: Preparing data for Numba kernel...")
    T_numba = np.require(T, dtype=np.float64, requirements=['C', 'A'])
    beta_obs_numba = np.require(beta_obs, dtype=np.float64, requirements=['C', 'A'])

    if is_Y_sparse:
        logger.info(f"Numba backend: Launching SPARSE Y permutation kernel (n_rand={n_rand})...")
        Yc_csc = Y.tocsc() # Kernel expects CSC
        Y_data    = np.require(Yc_csc.data, dtype=np.float64, requirements=['A'])
        Y_indices = np.require(Yc_csc.indices, dtype=np.int64, requirements=['A']) # Numba prefers int64 indices
        Y_indptr  = np.require(Yc_csc.indptr, dtype=np.int64, requirements=['A'])
        se, zscore, pvalue = _permutation_kernel_sparse_y(
            T_numba, Y_data, Y_indices, Y_indptr, beta_obs_numba, int(n_rand)
        )
        del Yc_csc, Y_data, Y_indices, Y_indptr
    else:
        logger.info(f"Numba backend: Launching DENSE Y permutation kernel (n_rand={n_rand})...")
        Y_numba = np.require(Y, dtype=np.float64, requirements=['C', 'A'])
        se, zscore, pvalue = _permutation_kernel_dense_y(
            T_numba, Y_numba, beta_obs_numba, int(n_rand)
        )
        del Y_numba

    logger.info("Numba backend: Permutation kernel finished.")
    del T, Xc, T_numba, beta_obs_numba; gc.collect()
    elapsed = time.time() - t0
    logger.info(f"Numba backend completed in {elapsed:.2f}s")
    return {'beta': beta_obs, 'se': se, 'zscore': zscore, 'pvalue': pvalue, 'method_used': 'numba', 'execution_time': elapsed}


# ============================================
#   MKL C Backend Placeholders
# ============================================
# These should not be called directly from core.py
def ridge_regression_mkl(*args, **kwargs):
    raise RuntimeError("Internal Error: MKL backend should be called via ridge_mkl.pyx wrapper.")

# ============================================
#   GSL C Backend Placeholders
# ============================================
# These should not be called directly from core.py
def ridge_regression_gsl(*args, **kwargs):
    raise RuntimeError("Internal Error: GSL backend should be called via ridge_gsl.pyx wrapper.")


# ==========================================
#   CuPy Backend Implementations
# ==========================================
# (_transfer_to_gpu, _calculate_T_cupy, _calculate_beta_cupy,
#  _perform_permutation_cupy, ridge_regression_cupy)
# Keep these exactly as they were in the previous version.

def _transfer_to_gpu(X, Y):
    """Transfer dense X and dense/sparse Y to GPU."""
    if not CUPY_AVAILABLE or cp is None: raise RuntimeError("CuPy is not available.")

    if sps.issparse(X):
         logger.warning("GPU backend received sparse X, converting to dense for transfer.")
         try: X_np = X.toarray(order='C').astype(np.float64)
         except MemoryError: raise MemoryError("GPU transfer: Out of memory converting sparse X to dense.")
         except Exception as e: raise TypeError(f"GPU transfer: Failed to convert sparse X to dense: {e}") from e
    else:
         X_np = np.require(X, dtype=np.float64, requirements=['C', 'A'])

    X_gpu = None
    try:
        X_gpu = cp.array(X_np); del X_np; gc.collect()
    except cp.cuda.memory.OutOfMemoryError as oom:
        raise RuntimeError("GPU out of memory while transferring X") from oom
    except Exception as e:
        raise RuntimeError(f"Error transferring X to GPU: {e}") from e

    is_Y_sparse_orig = sps.issparse(Y)
    Y_gpu = None
    Y_sp = None

    try:
        if is_Y_sparse_orig:
            Y_sp = Y.tocsr().astype(np.float64)
            logger.debug("GPU transfer: Creating CuPy CSR matrix...")
            Y_gpu = cpsp.csr_matrix((cp.asarray(Y_sp.data), cp.asarray(Y_sp.indices, dtype=cp.int32), cp.asarray(Y_sp.indptr, dtype=cp.int32)), shape=Y_sp.shape)
            del Y_sp; Y_sp = None # Clear host sparse matrix
        else:
            Y_np = np.require(Y, dtype=np.float64, requirements=['C', 'A'])
            logger.debug("GPU transfer: Transferring dense NumPy Y to GPU...")
            Y_gpu = cp.array(Y_np); del Y_np
    except cp.cuda.memory.OutOfMemoryError as oom:
        logger.error("GPU transfer: OOM transferring Y.", exc_info=True)
        raise RuntimeError("GPU out of memory transferring Y") from oom
    except Exception as e:
        logger.error(f"GPU transfer: Error preparing/transferring Y: {e}", exc_info=True)
        raise RuntimeError("Error transferring Y to GPU") from e
    finally: gc.collect()

    cp.cuda.Stream.null.synchronize()
    y_type_str = 'sparse CSR' if isinstance(Y_gpu, cpsp.csr_matrix) else 'dense'
    logger.debug(f"Transferred to GPU: X_gpu={X_gpu.shape}, Y_gpu={Y_gpu.shape} ({y_type_str})")
    return X_gpu, Y_gpu


def _calculate_T_cupy(X_gpu, lambda_val):
    """Calculates T = (X'X + lambda*I)^-1 @ X' using CuPy."""
    if not CUPY_AVAILABLE or cp is None: raise RuntimeError("CuPy is not available.")
    if not isinstance(X_gpu, cp.ndarray): raise TypeError("X_gpu must be CuPy ndarray.")

    n_genes, n_features = X_gpu.shape
    logger.debug(f"CuPy T calc: X_gpu shape=({n_genes}, {n_features})")
    T_gpu, A, X_T, A_pinv = None, None, None, None

    try:
        logger.debug("CuPy T calc: Computing X.T @ X...")
        X_T = X_gpu.T
        XtX = X_T @ X_gpu
        logger.debug(f"CuPy T calc: XtX shape={XtX.shape}")
        logger.debug(f"CuPy T calc: Adding lambda*I (lambda={lambda_val})...")
        lambda_f64 = float(lambda_val)
        A = XtX + lambda_f64 * cp.eye(n_features, dtype=cp.float64)
        del XtX; cp.cuda.Stream.null.synchronize()
        logger.debug("CuPy T calc: Attempting standard solve for T...")
        try:
            T_gpu = cp.linalg.solve(A, X_T)
            logger.debug("CuPy T calc: Standard solve successful.")
        except cp.linalg.LinAlgError as err_solve:
            logger.warning(f"CuPy T calc: Standard solve failed ({err_solve}), trying pseudo-inverse...")
            A_pinv = cp.linalg.pinv(A)
            T_gpu = A_pinv @ X_T
            logger.debug("CuPy T calc: Pseudo-inverse solve successful.")
        cp.cuda.Stream.null.synchronize()
    except cp.cuda.memory.OutOfMemoryError as e_oom:
        raise RuntimeError("GPU OOM during T calculation") from e_oom
    except Exception as e:
        raise RuntimeError(f"Error during GPU T calculation: {e}") from e
    finally:
        if A is not None: del A
        if X_T is not None: del X_T
        if A_pinv is not None: del A_pinv
        try:
            if cp: cp.get_default_memory_pool().free_all_blocks()
        except Exception: pass
        gc.collect()
    if T_gpu is None: raise RuntimeError("Failed to compute T on GPU")
    return T_gpu

def _calculate_beta_cupy(T_gpu, Y_gpu):
    """Calculates beta = T_gpu @ Y_gpu on the GPU."""
    if not CUPY_AVAILABLE or cp is None: raise RuntimeError("CuPy is not available.")
    if not isinstance(T_gpu, cp.ndarray): raise TypeError("T_gpu must be CuPy ndarray.")

    beta_gpu, beta_gpu_T = None, None
    try:
        is_Y_sparse = isinstance(Y_gpu, (cpsp.csr_matrix, cpsp.csc_matrix))
        logger.debug(f"GPU Beta calc: T_gpu={T_gpu.shape}, Y_gpu={Y_gpu.shape} ({'sparse' if is_Y_sparse else 'dense'})")
        if is_Y_sparse:
            logger.debug("GPU Beta calc: Using sparse Y.T @ T.T formulation...")
            beta_gpu_T = Y_gpu.T @ T_gpu.T
            beta_gpu = beta_gpu_T.T
        else:
            logger.debug("GPU Beta calc: Using dense T @ Y formulation...")
            if not isinstance(Y_gpu, cp.ndarray): Y_gpu = cp.array(Y_gpu, order='C')
            elif not Y_gpu.flags.c_contiguous: Y_gpu = cp.ascontiguousarray(Y_gpu)
            if Y_gpu.dtype != cp.float64: Y_gpu = Y_gpu.astype(cp.float64)
            beta_gpu = cp.dot(T_gpu, Y_gpu)
        cp.cuda.Stream.null.synchronize()
        logger.debug(f"GPU Beta calc: Final beta shape {beta_gpu.shape}")
        return beta_gpu
    except cp.cuda.memory.OutOfMemoryError as oom: raise RuntimeError("GPU OOM during beta calculation") from oom
    except CuSparseError as sparse_err: raise RuntimeError(f"cuSPARSE error during beta calculation: {sparse_err}") from sparse_err
    except Exception as err: raise RuntimeError(f"GPU beta calculation failed: {err}") from err
    finally:
        if beta_gpu_T is not None: del beta_gpu_T
        try:
            if cp: cp.get_default_memory_pool().free_all_blocks()
        except Exception: pass
        gc.collect()


def _perform_permutation_cupy(T_gpu, Y_gpu, beta_gpu, n_rand):
    """Performs permutation test using CuPy."""
    if not CUPY_AVAILABLE or cp is None: raise RuntimeError("CuPy not available.")
    if not isinstance(T_gpu, cp.ndarray): raise TypeError("T_gpu must be CuPy ndarray.")
    is_Y_sparse = isinstance(Y_gpu, (cpsp.csr_matrix, cpsp.csc_matrix))
    if not (isinstance(Y_gpu, cp.ndarray) or is_Y_sparse): raise TypeError("Y_gpu must be CuPy ndarray or sparse.")
    if not isinstance(beta_gpu, cp.ndarray): raise TypeError("beta_gpu must be dense CuPy ndarray.")

    p, n = T_gpu.shape
    _, m = beta_gpu.shape
    logger.info(f"CuPy Permutation: permuting T cols for {n_rand} iterations (Y is {'sparse' if is_Y_sparse else 'dense'})...")

    sum_b, sum_b2, count_ge = cp.zeros_like(beta_gpu), cp.zeros_like(beta_gpu), cp.zeros_like(beta_gpu)
    abs_beta = cp.abs(beta_gpu); rng = np.random.default_rng(SEED); indices = np.arange(n)
    T_perm, perm_gpu, beta_rand, beta_rand_T = None, None, None, None

    try:
        for i in range(n_rand):
            perm = rng.permutation(indices)
            perm_gpu = cp.asarray(perm, dtype=cp.intp)
            T_perm = T_gpu[:, perm_gpu]
            if is_Y_sparse:
                 beta_rand_T = Y_gpu.T @ T_perm.T; beta_rand = beta_rand_T.T
            else:
                 beta_rand = T_perm @ Y_gpu
            sum_b += beta_rand; sum_b2 += beta_rand ** 2
            count_ge += (cp.abs(beta_rand) >= abs_beta - EPS) # Add EPS tolerance
            if (i + 1) % 100 == 0: cp.get_default_memory_pool().free_all_blocks()
            if (i + 1) % 500 == 0 and logger.level <= logging.DEBUG: logger.debug(f"CuPy Permutation: {i+1}/{n_rand}")
        cp.cuda.Stream.null.synchronize()
        logger.info("CuPy Permutation loop completed.")
    except cp.cuda.memory.OutOfMemoryError as e: raise RuntimeError("GPU OOM during permutation loop") from e
    except Exception as e: raise RuntimeError(f"Error during GPU permutation loop: {e}") from e
    finally:
        if T_perm is not None: del T_perm
        if perm_gpu is not None: del perm_gpu
        if beta_rand is not None: del beta_rand
        if beta_rand_T is not None: del beta_rand_T

    logger.debug("CuPy Permutation: Finalizing statistics...")
    inv_n_rand = 1.0 / n_rand
    mean_b = sum_b * inv_n_rand; var_b = cp.maximum(sum_b2 * inv_n_rand - mean_b ** 2, 0.0)
    se_gpu = cp.sqrt(var_b); p_gpu = cp.minimum((count_ge + 1.0) / (n_rand + 1.0), 1.0)
    z_gpu = cp.zeros_like(beta_gpu); mask = se_gpu > EPS
    z_gpu[mask] = (beta_gpu[mask] - mean_b[mask]) / se_gpu[mask]
    zero_se_idx = ~mask
    z_gpu[zero_se_idx] = cp.where(cp.abs(beta_gpu[zero_se_idx] - mean_b[zero_se_idx]) < EPS,
                                   0.0,
                                   cp.inf * cp.sign(beta_gpu[zero_se_idx] - mean_b[zero_se_idx]))
    num_zero_se = cp.sum(zero_se_idx)
    if num_zero_se > 0: logger.warning(f"CuPy Permutation: {num_zero_se} SEs were near zero (set z=0 or +/-inf).")

    del sum_b, sum_b2, count_ge, abs_beta, mean_b, var_b, mask
    cp.get_default_memory_pool().free_all_blocks(); gc.collect()
    logger.debug("CuPy Permutation: Finalization and cleanup complete.")
    return se_gpu, z_gpu, p_gpu


def ridge_regression_cupy(X, Y, lambda_val=5e5, n_rand=1000):
    """Main CuPy backend function. Returns dict with results and GPU metrics."""
    if not CUPY_AVAILABLE or cp is None: raise RuntimeError("CuPy not available or GPU disabled.")
    if n_rand <= 0: raise NotImplementedError("CuPy backend requires n_rand > 0.")

    is_Y_sparse_orig = sps.issparse(Y)
    prefix = f"CuPy {'Sparse' if is_Y_sparse_orig else 'Dense'} Y:"
    logger.info(f"{prefix} Starting GPU ridge regression")
    start_evt, end_evt = cp.cuda.Event(), cp.cuda.Event(); start_evt.record()
    beta, se, zscores, pvalues = None, None, None, None
    peak_gpu_pool_mb = np.nan
    X_gpu, Y_gpu, T_gpu, beta_gpu, se_gpu, z_gpu, p_gpu = (None,) * 7

    try:
        X_gpu, Y_gpu = _transfer_to_gpu(X, Y)
        T_gpu    = _calculate_T_cupy(X_gpu, lambda_val)
        beta_gpu = _calculate_beta_cupy(T_gpu, Y_gpu)
        se_gpu, z_gpu, p_gpu = _perform_permutation_cupy(T_gpu, Y_gpu, beta_gpu, n_rand)
        try:
            pool = cp.get_default_memory_pool(); peak_bytes = getattr(pool, 'get_peak_bytes', pool.used_bytes)()
            peak_gpu_pool_mb = peak_bytes / (1024**2)
        except Exception as mem_err: logger.warning(f"{prefix} Could not measure GPU memory: {mem_err}")
        logger.debug(f"{prefix} Transferring results back to CPU...")
        beta, se, zscores, pvalues = cp.asnumpy(beta_gpu), cp.asnumpy(se_gpu), cp.asnumpy(z_gpu), cp.asnumpy(p_gpu)
    except Exception as err: logger.error(f"{prefix} Error in GPU workflow: {err}", exc_info=True); raise
    finally:
        end_evt.record(); end_evt.synchronize()
        gpu_time_s = cp.cuda.get_elapsed_time(start_evt, end_evt) / 1000.0 if start_evt.done and end_evt.done else np.nan
        logger.debug(f"{prefix} Cleaning up GPU memory...")
        gpu_vars_to_clean = [X_gpu, Y_gpu, T_gpu, beta_gpu, se_gpu, z_gpu, p_gpu]
        for obj in gpu_vars_to_clean:
             if obj is not None: del obj
        del gpu_vars_to_clean
        try:
            if cp: cp.get_default_memory_pool().free_all_blocks()
        except Exception as pool_e: logger.warning(f"{prefix} Could not free CuPy pool blocks: {pool_e}")
        gc.collect(); logger.debug(f"{prefix} GPU cleanup finished.")

    logger.info(f"{prefix} Completed in {gpu_time_s:.2f}s")
    return {
        'beta': beta, 'se': se, 'zscore': zscores, 'pvalue': pvalues, 'method_used': 'gpu',
        'execution_time': gpu_time_s, # Report GPU execution time
        'peak_gpu_pool_mb': peak_gpu_pool_mb if not np.isnan(peak_gpu_pool_mb) else None
    }