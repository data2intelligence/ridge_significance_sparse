"""
Logistic regression with significance testing and Firth correction.

This module provides functionality for logistic regression with
optional Firth correction to handle separation issues.
Primarily implemented using NumPy and SciPy for CPU execution.
"""

import numpy as np
from scipy import linalg
from scipy import stats
import time
import logging
import warnings

logger = logging.getLogger(__name__)

# Error codes (for informational purposes, not directly used as return values)
REG_FAIL = -1
CONVERGE_FAIL = -2

# Numerical tolerance
EPS = 1e-10 # Tolerance for checking near-zero values (e.g., SE, variance)
DEFAULT_TOL = 1e-6 # Default convergence tolerance for iterative solver

def sqrt_eps(x):
    """Safely computes square root, returning 0 for small negative inputs."""
    if x < 0 and x > -EPS:
        return 0.0
    elif x < -EPS:
         warnings.warn(f"Attempting sqrt of significant negative number ({x}). Result will be NaN.", RuntimeWarning)
         return np.nan
    else:
        return np.sqrt(x)

def logit(X, Y, tol=DEFAULT_TOL, max_delta=0.5, maxIter=100, correction=0, verbose=0):
    """
    Performs logistic regression using Iteratively Reweighted Least Squares (IRLS).
    Includes optional Firth correction for separation/small sample issues and
    calculates Wald test statistics (SE, Z-score, p-value).

    Parameters
    ----------
    X : numpy.ndarray
        Input matrix (design matrix) of shape (n_samples, n_features).
        Should include an intercept column if desired (not added automatically).
        Expected to be dense NumPy array.
    Y : numpy.ndarray
        Binary outcome vector (0 or 1) of shape (n_samples,).
        Expected to be dense NumPy array.
    tol : float
        Convergence tolerance for the change in the gradient norm.
        Default is 1e-6.
    max_delta : float
        Maximum relative step size allowed per iteration to control updates.
        Default is 0.5.
    maxIter : int
        Maximum number of IRLS iterations allowed. Default is 100.
    correction : int or bool
        If non-zero (True), applies Firth correction to the score function.
        Default is 0 (False).
    verbose : int
        Verbosity level (0=quiet, 1=info/warnings, 2=debug).

    Returns
    -------
    tuple
        (beta, stderr, zscore, pvalue) numpy arrays, each of shape (n_features,).
        Returns arrays of NaNs on failure (e.g., singular matrix, non-convergence).

    Raises
    ------
    ValueError
        If input dimensions are inconsistent or types are invalid.
    """
    start_time = time.time()

    # --- Input Validation ---
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise ValueError("X must be a 2D NumPy array.")
    if not isinstance(Y, np.ndarray) or Y.ndim != 1:
        raise ValueError("Y must be a 1D NumPy array.")
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"X and Y must have the same number of samples (rows), "
                         f"got X.shape={X.shape}, Y.shape={Y.shape}")
    if not np.all(np.isin(Y, [0, 1])):
        warnings.warn("Y contains values other than 0 or 1. Proceeding, but results may be invalid for logistic regression.", RuntimeWarning)

    # Ensure float64 for numerical stability and C compatibility if ever needed
    try:
        X = np.require(X, dtype=np.float64, requirements=['C', 'A'])
        Y = np.require(Y, dtype=np.float64, requirements=['C', 'A'])
    except Exception as e:
        raise TypeError("Failed to ensure X and Y are float64 NumPy arrays.") from e

    n_samples, n_features = X.shape
    log_prefix = "Logit:"
    if verbose >= 1:
         logger.info(f"{log_prefix} Starting logistic regression. "
                     f"Samples={n_samples}, Features={n_features}, Firth={bool(correction)}, MaxIter={maxIter}, Tol={tol}")

    # --- Initialization ---
    beta = np.zeros(n_features, dtype=np.float64)
    # Pre-allocate arrays for results
    stderr = np.full(n_features, np.nan, dtype=np.float64)
    zscore = np.full(n_features, np.nan, dtype=np.float64)
    pvalue = np.full(n_features, np.nan, dtype=np.float64)

    status_code = 0 # 0: Running, 1: Converged, CONVERGE_FAIL: Max iter exceeded, REG_FAIL: LinAlgError

    # --- Main IRLS Iteration Loop ---
    I_info = None # Fisher Information Matrix (or Hessian approximation)
    I_inv = None  # Inverse of I_info

    for iter_count in range(maxIter):
        # Calculate linear predictor: eta = X @ beta
        eta = X @ beta

        # Calculate probabilities: P = sigmoid(eta) = 1 / (1 + exp(-eta))
        # Avoid overflow in exp()
        # P = np.exp(-np.logaddexp(0, -eta)) # Numerically stable sigmoid
        # Simpler robust version: Clip eta to avoid extreme values in exp
        eta_clipped = np.clip(eta, -30, 30) # Adjust bounds if needed
        P = 1.0 / (1.0 + np.exp(-eta_clipped))
        # Clamp probabilities to avoid exact 0 or 1, which cause issues in W
        P = np.clip(P, EPS, 1.0 - EPS)

        # Calculate weights: W = P * (1 - P) (diagonal of weight matrix)
        W = P * (1.0 - P)

        # Calculate error (residuals): error = Y - P
        error = Y - P

        # Calculate Fisher Information Matrix: I = X' * diag(W) * X
        # Efficient calculation: sqrt(W)*X needed for Cholesky and Firth H
        W_sqrt = np.sqrt(W)
        X_w = X * W_sqrt[:, np.newaxis] # Element-wise multiplication broadcasts W_sqrt
        try:
            I_info = X_w.T @ X_w
        except Exception as e:
             logger.error(f"{log_prefix} Error calculating Information Matrix I = X'WX: {e}", exc_info=(verbose>1))
             status_code = REG_FAIL # Treat as matrix failure
             break # Exit loop

        # Add small diagonal jitter for numerical stability? Optional.
        # I_info += np.eye(n_features) * 1e-9

        # --- Cholesky decomposition for solving and potential inverse ---
        try:
            # Attempt Cholesky on I_info = X'WX
            L = linalg.cholesky(I_info, lower=True)
        except np.linalg.LinAlgError as e_chol:
            logger.warning(f"{log_prefix} Iter {iter_count+1}: Cholesky decomposition failed on X'WX ({e_chol}). Matrix might be singular or not positive definite. Trying pseudo-inverse.")
            # Fallback: Use pseudo-inverse for update, but stats might be less reliable
            try:
                I_inv = linalg.pinv(I_info)
                # Calculate gradient U = X' * error
                U = X.T @ error # Simple gradient without correction for pinv path
                delta = I_inv @ U # Calculate update step using pseudo-inverse
                status_code = 0 # Continue iterating, but mark stats as potentially unreliable?
                # Skip Firth correction if Cholesky failed? Yes, H depends on stable inv(I)
                logger.warning(f"{log_prefix} Using pseudo-inverse for update step. Firth correction skipped this iteration. Statistical results may be affected.")
                correction_this_iter = False # Disable correction for this step
            except np.linalg.LinAlgError as e_pinv:
                 logger.error(f"{log_prefix} Iter {iter_count+1}: Pseudo-inverse also failed ({e_pinv}). Cannot continue.")
                 status_code = REG_FAIL
                 break # Exit loop
            except Exception as e_pinv_other:
                 logger.error(f"{log_prefix} Iter {iter_count+1}: Unexpected error during pseudo-inverse fallback: {e_pinv_other}")
                 status_code = REG_FAIL
                 break
        except Exception as e_chol_other:
             logger.error(f"{log_prefix} Iter {iter_count+1}: Unexpected error during Cholesky: {e_chol_other}")
             status_code = REG_FAIL
             break
        else:
             # Cholesky succeeded, proceed with standard update and optional correction
             correction_this_iter = bool(correction) # Use requested correction status
             delta = None # Initialize delta

        # --- Apply Firth Correction (if enabled and Cholesky succeeded) ---
        score_corr = np.zeros(n_features, dtype=np.float64) # Firth bias term for score
        if correction_this_iter:
            try:
                # Calculate diagonal of Hat matrix H = W^(1/2) * X * (X'WX)^-1 * X' * W^(1/2)
                # H_diag_i = (X_w[i, :] @ I_inv) @ X_w[i, :].T
                # Use Cholesky solves for efficiency: Solve L*Z = X_w.T, then L.T*Q = Z -> Q = I_inv*X_w.T
                # Then H_diag[i] = W[i] * (X[i,:] @ Q[:,i]) ??? No, simpler way:
                # H_diag = sum(X_w * (X_w @ I_inv), axis=1) where I_inv = cho_solve(L)
                # Let I_inv_explicit = linalg.cho_solve((L, True), np.eye(n_features)) # Get full inverse
                # H_diag = np.sum(X_w @ I_inv_explicit * X_w, axis=1) # Efficient diagonal calculation

                # Alternative using solve: Solve I*Q = X_w.T for Q = I_inv*X_w.T
                Q = linalg.cho_solve((L, True), X_w.T) # Q is p x n
                # H_diag = np.sum(X_w * Q.T, axis=1) # Equivalent and avoids full inverse
                H_diag = np.einsum('ij,ji->i', X_w, Q) # More concise einsum for row-wise dot product

                # Firth bias term for score U: X.T @ (H_diag * (0.5 - P))
                firth_bias_per_sample = H_diag * (0.5 - P) # Vector n
                score_corr = X.T @ firth_bias_per_sample # Vector p
            except np.linalg.LinAlgError as e_firth:
                logger.warning(f"{log_prefix} Iter {iter_count+1}: LinAlgError during Firth correction calculation ({e_firth}). Skipping correction this iteration.")
                score_corr.fill(0.0) # Reset correction term
            except Exception as e_firth_other:
                 logger.warning(f"{log_prefix} Iter {iter_count+1}: Error during Firth correction ({e_firth_other}). Skipping correction this iteration.")
                 score_corr.fill(0.0)

        # --- Calculate Gradient (Score) vector U ---
        # Standard gradient: U = X' * error
        # Corrected gradient: U_corr = X' * error + score_corr (Firth bias)
        U = X.T @ error + score_corr # score_corr is zero if correction is off or failed

        # --- Check Convergence ---
        # Use norm of the *potentially corrected* gradient U
        gradient_norm = np.linalg.norm(U)
        if gradient_norm < tol:
            if verbose >= 1: logger.info(f"{log_prefix} Converged at iteration {iter_count+1} (gradient norm = {gradient_norm:.2e})")
            status_code = 1 # Converged
            # Calculate final I_inv needed for stats *after* convergence
            try:
                if I_inv is None: # If we didn't use pseudo-inverse path
                     I_inv = linalg.cho_solve((L, True), np.eye(n_features, dtype=np.float64))
            except np.linalg.LinAlgError as e_final_inv:
                 logger.error(f"{log_prefix} Failed to compute final inverse of I for stats after convergence: {e_final_inv}")
                 status_code = REG_FAIL # Mark as failure if stats cannot be computed
            break # Exit loop

        # --- Calculate Update Step (Delta) ---
        if delta is None: # If not calculated via pseudo-inverse fallback
             try:
                 # Solve I_info * delta = U using Cholesky factors: L*L.T*delta = U
                 # Step 1: Solve L*y = U for y
                 y = linalg.solve_triangular(L, U, lower=True, check_finite=False)
                 # Step 2: Solve L.T*delta = y for delta
                 delta = linalg.solve_triangular(L.T, y, lower=False, check_finite=False)
             except np.linalg.LinAlgError as e_solve:
                  logger.warning(f"{log_prefix} Iter {iter_count+1}: Solve failed during update step calculation ({e_solve}). Trying pseudo-inverse.")
                  # Attempt pseudo-inverse for update only if solve fails after successful Cholesky
                  try:
                      if I_inv is None: # Calculate full inverse if not already done
                           I_inv = linalg.cho_solve((L, True), np.eye(n_features, dtype=np.float64))
                      delta = I_inv @ U
                  except Exception as e_pinv_update:
                       logger.error(f"{log_prefix} Iter {iter_count+1}: Pseudo-inverse also failed during update ({e_pinv_update}). Cannot continue.")
                       status_code = REG_FAIL
                       break # Exit loop

        # --- Step Size Control ---
        norm_delta = np.linalg.norm(delta)
        norm_beta = np.linalg.norm(beta)

        # Calculate scale factor to limit step size relative to current beta magnitude
        # Avoid division by zero if norm_beta is very small initially
        max_step_abs = max_delta * (norm_beta + 1.0) # Max allowed absolute change based on relative limit

        if norm_delta > max_step_abs:
            scale_factor = max_step_abs / norm_delta
            if verbose >= 2: logger.debug(f"{log_prefix} Iter {iter_count+1}: Reducing step size (factor={scale_factor:.3f}). Delta norm={norm_delta:.2e}, Beta norm={norm_beta:.2e}")
            delta *= scale_factor

        # --- Update Beta ---
        beta += delta

        if verbose >= 2: logger.debug(f"{log_prefix} Iter {iter_count+1}: Gradient norm={gradient_norm:.2e}, Delta norm={np.linalg.norm(delta):.2e}")

    # --- End of Iteration Loop ---

    # --- Check Final Status ---
    if status_code == 0: # Means loop finished without converging
        logger.warning(f"{log_prefix} Failed to converge within {maxIter} iterations. Last gradient norm = {gradient_norm:.2e}")
        status_code = CONVERGE_FAIL
        # Try to compute I_inv for stats based on last iteration? Risky.
        try:
             if I_inv is None and 'L' in locals() and L is not None: # If Cholesky was computed in last iter
                  I_inv = linalg.cho_solve((L, True), np.eye(n_features, dtype=np.float64))
             elif I_inv is None and I_info is not None: # Fallback to pinv if Cholesky failed last iter
                  I_inv = linalg.pinv(I_info)
        except Exception:
             logger.error(f"{log_prefix} Could not compute final I_inv for stats after non-convergence.")
             I_inv = None # Ensure I_inv is None if stats calculation is impossible

    elif status_code == REG_FAIL:
        logger.error(f"{log_prefix} Logistic regression failed due to matrix singularity or numerical errors.")
        # beta already zeroed or contains last unstable value. stderr etc remain NaN.
        return beta, stderr, zscore, pvalue # Return NaNs

    # --- Calculate Final Statistics (if converged or non-convergence but I_inv available) ---
    if I_inv is not None:
        try:
            # Standard errors are sqrt of diagonal elements of the inverse Fisher Info matrix
            diag_I_inv = np.diag(I_inv)
            # Check for negative variances which indicate issues
            if np.any(diag_I_inv < -EPS):
                 neg_indices = np.where(diag_I_inv < -EPS)[0]
                 logger.warning(f"{log_prefix} Negative diagonal elements found in Inverse Fisher Information at indices {neg_indices}. SE/Z/Pvalue will be NaN for these features.")
            # Calculate SE using safe sqrt
            stderr = np.array([sqrt_eps(v) for v in diag_I_inv])

            # Calculate Z-scores (Wald statistic)
            valid_se_mask = (stderr > EPS) & ~np.isnan(stderr)
            zscore[valid_se_mask] = beta[valid_se_mask] / stderr[valid_se_mask]
            zscore[~valid_se_mask] = 0.0 # Set Z to 0 where SE is invalid/zero

            # Calculate two-sided p-values from standard normal distribution N(0,1)
            pvalue = 2.0 * stats.norm.sf(np.abs(zscore)) # sf is 1 - cdf
            pvalue = np.clip(pvalue, 0.0, 1.0) # Ensure p-values are in [0, 1]

            if status_code == CONVERGE_FAIL:
                 logger.warning(f"{log_prefix} Calculated statistics after non-convergence. Results might be unreliable.")

        except Exception as e_stats:
            logger.error(f"{log_prefix} Error calculating final statistics: {e_stats}", exc_info=(verbose>1))
            # Leave stderr, zscore, pvalue as NaNs if stats calculation fails
    else:
        logger.error(f"{log_prefix} Could not calculate final statistics because inverse Fisher Information matrix is unavailable.")
        # Results remain NaNs


    execution_time = time.time() - start_time
    if verbose >= 1:
        final_status_msg = "Converged" if status_code == 1 else "Failed (Max Iterations)" if status_code == CONVERGE_FAIL else "Failed (Singular Matrix/Error)"
        logger.info(f"{log_prefix} Logistic regression finished in {execution_time:.2f} seconds. Status: {final_status_msg}")

    # Return the results (beta might be non-zero even on failure, but stats will be NaN)
    return beta, stderr, zscore, pvalue