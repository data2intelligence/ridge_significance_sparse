#!/usr/bin/env python
"""
Example demonstrating sparse matrix handling utilities and ridge regression
with artificially sparsified real infection data. Includes correlation checks
between results from dense and sparse inputs using the package's default lambda.
"""

import os
import numpy as np
import pandas as pd
# Import matplotlib only if needed later
# import matplotlib.pyplot as plt
from pathlib import Path
import sys
import time
import logging
import warnings
from scipy import sparse as sps # Use alias
from scipy.stats import pearsonr # Import pearsonr
import argparse # For command line arguments
import gc # For garbage collection

# Set OpenMP threads early if needed (C backend uses this)
os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "4")

# Ensure ridge_inference is importable
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent
sys.path.append(str(project_root))

try:
    from ridge_inference import ridge, load_signature_matrix
    # Try to import DEFAULT_LAMBDA from the correct module (assuming ridge.py)
    try:
        from ridge_inference.ridge import DEFAULT_LAMBDA
    except ImportError:
        # Fallback if it's defined elsewhere or renamed
        logger.warning("Could not import DEFAULT_LAMBDA from ridge_inference.ridge, using 5e5.")
        DEFAULT_LAMBDA = 5e5

    # Import utils needed for sparse checks/conversion if desired
    from ridge_inference.utils import convert_to_sparse
    # Import the renamed function
    from ridge_inference.inference import _is_Y_sparse_beneficial_for_cpu
    from ridge_inference.core import CUPY_AVAILABLE # Check if GPU is available
    # *** FIX: Import is_backend_available ***
    from ridge_inference.backend_selection import is_backend_available
except ImportError as e:
    print(f"Error importing ridge_inference: {e}")
    print("Ensure the package is installed or the project root is in PYTHONPATH.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
DATA_DIR = project_root / "data" / "sample_data"
EXPR_PATH = DATA_DIR / "infection_GSE147507.gz"
SIG_MATRIX_NAME = "CytoSig" # Signature matrix (likely dense)
OUTPUT_DIR = script_dir / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Sparsification parameters for expression data
SPARSITY_THRESHOLD_PERCENTILE = 50 # Set values below this percentile (of non-zeros) to 0

# Regression parameters
LAMBDA_VAL = DEFAULT_LAMBDA # Use the imported default lambda
N_RAND = 1000 # Fewer permutations for example speed
VERBOSE_LEVEL = 1

# --- Correlation Thresholds for Dense vs Sparse Comparison ---
BETA_CORR_THRESHOLD_DS = 0.99
SE_CORR_THRESHOLD_DS = 0.90   # SE might be more affected by sparsification
Z_CORR_THRESHOLD_DS = 0.85
PVAL_CORR_THRESHOLD_DS = 0.85

# Define EPS constant used in helper function
EPS = 1e-8


# --- Helper Functions ---
def difference(x, y, max_mode=True):
    """Calculates max or mean absolute difference between two numpy arrays."""
    if x is None or y is None or not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray): return np.inf
    if x.shape != y.shape:
         if x.size == y.size: logger.warning(f"Difference check: Shape mismatch but same size ({x.shape} vs {y.shape}). Comparing flattened.")
         else: return np.inf
    try:
        diff = np.abs(x.flatten().astype(np.float64) - y.flatten().astype(np.float64))
        return np.nanmax(diff) if max_mode else np.nanmean(diff)
    except Exception: return np.inf

def calculate_and_report_correlation(mat1, mat2, name1="Mat1", name2="Mat2", threshold=0.99):
    """Calculates and reports Pearson correlation against a threshold."""
    passed = False; corr = None; message = f"Correlation check between {name1} and {name2}"
    if mat1 is None or mat2 is None: message += " - SKIPPED (Missing Data)"; logger.warning(message); print(message); return passed, corr, message
    if not isinstance(mat1, np.ndarray) or not isinstance(mat2, np.ndarray): message += " - SKIPPED (Not NumPy arrays)"; logger.warning(message); print(message); return passed, corr, message
    shape_mismatch = False
    if mat1.shape != mat2.shape:
         logger.warning(f"  Shape mismatch: {name1} {mat1.shape}, {name2} {mat2.shape}."); shape_mismatch = True
         if mat1.size != mat2.size: message += " - FAILED (Element Count Mismatch)"; logger.error(message); print(message); return passed, corr, message
         logger.warning("  Attempting correlation on flattened arrays despite shape mismatch.")
    try:
        flat1 = mat1.flatten(); flat2 = mat2.flatten(); valid_mask = np.isfinite(flat1) & np.isfinite(flat2); num_valid = np.sum(valid_mask)
        if num_valid < 2: message += f": R=NaN (Insufficient valid data points: {num_valid}) - FAILED"; logger.error(message); print(message); return passed, corr, message
        flat1_valid = flat1[valid_mask]; flat2_valid = flat2[valid_mask]; std1 = np.std(flat1_valid); std2 = np.std(flat2_valid)
        # Use the globally defined EPS constant here
        if std1 < EPS or std2 < EPS:
             if np.allclose(flat1_valid, flat2_valid, atol=EPS*10, rtol=EPS*10): corr = 1.0; passed = True; message += f": R=1.0 (Identical Const) - PASSED"; logger.info(message)
             else: corr = np.nan; message += " - FAILED (Different/Non-Identical Const)"; logger.warning(message)
             print(message); return passed, corr, message
        corr, p_val = pearsonr(flat1_valid, flat2_valid)
        if np.isnan(corr): message += f": R=NaN - FAILED (Invalid Corr Result)"; logger.error(message); print(message); return passed, corr, message
        message += f": R={corr:+.6f}"
        if corr >= threshold: passed = True; message += f" (>= {threshold:.4f}) - PASSED"; logger.info(message)
        else: message += f" (< {threshold:.4f}) - FAILED THRESHOLD"; logger.warning(message)
        print(message)
    except ValueError as e: logger.error(f" Correlation ValueError ({name1} vs {name2}): {e}"); message += f" - FAILED (ValueError)"; print(message)
    except Exception as e: logger.error(f" Correlation Exception ({name1} vs {name2}): {type(e).__name__} - {e}", exc_info=logger.level <= logging.DEBUG); message += f" - FAILED (Exception)"; print(message)
    if shape_mismatch: message += " [Shape Mismatch Warning]"
    return passed, corr, message
# --- END OF HELPER FUNCTIONS ---


# --- Main Script Logic ---
if __name__ == "__main__":
    # --- Load Data ---
    logger.info(f"Loading expression data from: {EXPR_PATH}")
    if not EXPR_PATH.exists(): logger.error(f"File not found: {EXPR_PATH}"); sys.exit(1)
    expr_df = pd.read_csv(EXPR_PATH, sep='\t', index_col=0, compression='gzip')
    logger.info(f"Original expression data shape: {expr_df.shape} (Genes x Samples)")

    logger.info(f"Loading signature matrix: {SIG_MATRIX_NAME}")
    try:
        sig_df = load_signature_matrix(SIG_MATRIX_NAME)
        logger.info(f"Signature matrix shape: {sig_df.shape} (Genes x Proteins)")
    except Exception as e:
         logger.error(f"Failed to load signature matrix '{SIG_MATRIX_NAME}': {e}")
         sys.exit(1)

    # --- Preprocessing (Filter common genes) ---
    logger.info("Preprocessing: Filtering to common genes...")
    # Ensure indices are string type for reliable intersection
    expr_df.index = expr_df.index.astype(str)
    sig_df.index = sig_df.index.astype(str)
    common_genes = sorted(list(set(expr_df.index) & set(sig_df.index)))
    if not common_genes: logger.error("No common genes found."); sys.exit(1)
    logger.info(f"Found {len(common_genes)} common genes.")

    expr_filtered = expr_df.loc[common_genes].copy()
    sig_filtered = sig_df.loc[common_genes].copy()

    # Store indices/columns before converting to numpy/sparse
    protein_names = sig_filtered.columns.tolist()
    sample_names = expr_filtered.columns.tolist()

    # Dense matrices
    X_mat_dense = sig_filtered.values # genes x proteins (features)
    Y_mat_dense = expr_filtered.values # genes x samples (outputs)
    logger.info(f"Dense matrix shapes - X (Sig): {X_mat_dense.shape}, Y (Expr): {Y_mat_dense.shape}")

    # --- Introduce Sparsity to Expression Matrix (Y) ---
    logger.info(f"\nIntroducing sparsity to expression matrix (Y)...")
    non_zero_vals = Y_mat_dense[Y_mat_dense != 0]
    if len(non_zero_vals) > 0:
        threshold_val = np.percentile(np.abs(non_zero_vals), SPARSITY_THRESHOLD_PERCENTILE) # Use absolute value percentile
        logger.info(f"Setting values with absolute value < {threshold_val:.4f} (approx {SPARSITY_THRESHOLD_PERCENTILE}th percentile of non-zeros) to zero.")
        Y_mat_artificially_sparse_np = Y_mat_dense.copy()
        Y_mat_artificially_sparse_np[np.abs(Y_mat_artificially_sparse_np) < threshold_val] = 0
    else:
        logger.warning("Expression matrix Y seems to be all zeros. Skipping sparsification.")
        threshold_val = 0
        Y_mat_artificially_sparse_np = Y_mat_dense.copy()

    original_density = np.count_nonzero(Y_mat_dense) / Y_mat_dense.size
    sparse_density = np.count_nonzero(Y_mat_artificially_sparse_np) / Y_mat_artificially_sparse_np.size
    logger.info(f"Original Y density: {original_density:.4f}")
    logger.info(f"Artificially sparsified Y density: {sparse_density:.4f}")

    # --- Check Sparse Benefit & Convert ---
    sparse_beneficial = _is_Y_sparse_beneficial_for_cpu(Y_mat_artificially_sparse_np)
    logger.info(f"Is sparse beneficial (internal check)? {sparse_beneficial}")

    logger.info("Converting artificially sparse Y matrix to CSR format...")
    Y_input_sparse = sps.csr_matrix(Y_mat_artificially_sparse_np) # Convert Y to CSR
    X_input_dense = X_mat_dense # Keep X dense

    logger.info(f"Input types for sparse run: X={type(X_input_dense)}, Y={type(Y_input_sparse)}")
    logger.info(f"Input shapes for sparse run: X={X_input_dense.shape}, Y={Y_input_sparse.shape}")

    # --- Run Ridge Regression with Dense Y (for comparison) ---
    logger.info(f"\n--- Running Ridge Regression with ORIGINAL DENSE Y (lambda={LAMBDA_VAL:.1e}) ---")
    results_dense = None # Initialize results dict
    start_time_dense = time.time()
    try:
        results_dense = ridge(
            X_mat_dense, Y_mat_dense, # Use original dense arrays
            lambda_=LAMBDA_VAL, n_rand=N_RAND, method="auto", verbose=VERBOSE_LEVEL
        )
        duration = time.time() - start_time_dense
        logger.info(f"Ridge regression (Dense Y) finished in {duration:.2f} seconds. Method used: {results_dense.get('method_used','N/A')}")
        logger.info("--- Ridge Results (Dense Y) ---")
        beta_dense_df = pd.DataFrame(results_dense['beta'], index=protein_names, columns=sample_names)
        print("Estimated beta (head):")
        print(beta_dense_df.head(3))
    except Exception as e:
        logger.error(f"Ridge regression with dense Y failed: {e}", exc_info=True)

    # --- Run Ridge Regression with Sparse Y ---
    logger.info(f"\n--- Running Ridge Regression with ARTIFICIALLY SPARSE Y (lambda={LAMBDA_VAL:.1e}) ---")
    results_sparse = None # Initialize results dict
    # Choose a backend that supports sparse Y
    # Preference: MKL > GPU > Python > Numba (GSL densifies)
    # *** USE is_backend_available (now imported) ***
    if is_backend_available('mkl'): sparse_method = 'mkl'
    elif is_backend_available('gpu'): sparse_method = 'gpu'
    elif is_backend_available('python'): sparse_method = 'python'
    elif is_backend_available('numba'): sparse_method = 'numba'
    else:
        logger.error("No backend supporting sparse Y input is available!")
        sys.exit(1)

    logger.info(f"Using method='{sparse_method}' as it supports sparse input.")
    start_time_sparse = time.time()
    try:
        results_sparse = ridge(
            X_input_dense, Y_input_sparse, # Use dense X, sparse Y
            lambda_=LAMBDA_VAL, n_rand=N_RAND, method=sparse_method, verbose=VERBOSE_LEVEL
        )
        duration = time.time() - start_time_sparse
        logger.info(f"Ridge regression (Sparse Y) finished in {duration:.2f} seconds. Method used: {results_sparse.get('method_used','N/A')}")
        logger.info("\n--- Ridge Results (Sparse Y) ---")
        beta_sparse_df = pd.DataFrame(results_sparse['beta'], index=protein_names, columns=sample_names)
        print("Estimated beta (head):")
        print(beta_sparse_df.head(3))

        # --- Compare Dense vs Sparse Results using CORRELATION ---
        logger.info("\n--- Comparing Dense Y vs Sparse Y Results (Correlation Check) ---")
        comparison_overall_passed = True # Assume true initially

        # Extract numpy arrays only if runs were successful
        beta_dense_np = results_dense.get('beta') if results_dense else None
        se_dense_np = results_dense.get('se') if results_dense else None
        zscore_dense_np = results_dense.get('zscore') if results_dense else None
        pvalue_dense_np = results_dense.get('pvalue') if results_dense else None

        beta_sparse_np = results_sparse.get('beta') if results_sparse else None
        se_sparse_np = results_sparse.get('se') if results_sparse else None
        zscore_sparse_np = results_sparse.get('zscore') if results_sparse else None
        pvalue_sparse_np = results_sparse.get('pvalue') if results_sparse else None

        if beta_dense_np is not None and beta_sparse_np is not None: # Check if both runs succeeded
            passed_beta, _, _ = calculate_and_report_correlation(beta_dense_np, beta_sparse_np, "Dense Beta", "Sparse Beta", BETA_CORR_THRESHOLD_DS)
            passed_se, _, _ = calculate_and_report_correlation(se_dense_np, se_sparse_np, "Dense SE", "Sparse SE", SE_CORR_THRESHOLD_DS)
            passed_z, _, _ = calculate_and_report_correlation(zscore_dense_np, zscore_sparse_np, "Dense Z", "Sparse Z", Z_CORR_THRESHOLD_DS)
            passed_p, _, _ = calculate_and_report_correlation(pvalue_dense_np, pvalue_sparse_np, "Dense Pval", "Sparse Pval", PVAL_CORR_THRESHOLD_DS)
            if not (passed_beta and passed_se and passed_z and passed_p): comparison_overall_passed = False
        else:
            logger.warning("Skipping correlation comparison - one or both ridge runs failed.")
            comparison_overall_passed = False

        # --- Max Difference Comparison (Keep for info) ---
        if beta_dense_np is not None and beta_sparse_np is not None:
            diff = difference(beta_dense_np, beta_sparse_np)
            logger.info(f"\nComparison: Max absolute difference between dense Y beta and sparse Y beta: {diff:.4f} (for info)")
            # --- Scatter Plot ---
            # Uncomment below if matplotlib is installed
            # try:
            #     import matplotlib.pyplot as plt
            #     plt.figure(figsize=(8, 8))
            #     plt.scatter(beta_dense_np.flatten(), beta_sparse_np.flatten(), alpha=0.1, s=5)
            #     lim_min = min(np.nanmin(beta_dense_np), np.nanmin(beta_sparse_np)); lim_max = max(np.nanmax(beta_dense_np), np.nanmax(beta_sparse_np))
            #     padding = (lim_max - lim_min) * 0.05; lim_min -= padding; lim_max += padding
            #     plt.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', alpha=0.7)
            #     plt.xlabel("Beta (Dense Y)"); plt.ylabel("Beta (Sparse Y)"); plt.title("Beta Comparison: Dense vs Artificially Sparse Y")
            #     plt.xlim(lim_min, lim_max); plt.ylim(lim_min, lim_max); plt.grid(True, linestyle=':'); plt.tight_layout()
            #     plot_path = OUTPUT_DIR / "sparse_vs_dense_beta_comparison.png"
            #     plt.savefig(plot_path); logger.info(f"Beta comparison plot saved to {plot_path}"); plt.close()
            # except ImportError: logger.warning("matplotlib not found, skipping plot generation.")
            # except Exception as plot_err: logger.warning(f"Could not plot beta differences: {plot_err}")
        # --- End of Comparison ---

    except NotImplementedError as e: logger.error(f"Sparse input seems not implemented in the selected backend '{sparse_method}': {e}"); comparison_overall_passed = False
    except Exception as e: logger.error(f"Ridge regression with sparse Y failed unexpectedly: {e}", exc_info=True); comparison_overall_passed = False

    # --- Final Summary ---
    logger.info("\n--- Final Summary of Dense vs Sparse Correlation Checks ---")
    if results_dense is None or results_sparse is None: logger.error("\nOne or both ridge runs FAILED. Cannot determine pass/fail status.")
    elif comparison_overall_passed: logger.info("All correlation checks between dense Y and sparse Y results passed minimum thresholds.")
    else: logger.warning("\nOne or more correlation checks FAILED between dense Y and sparse Y results (check logs above).")

    logger.info("\nSparse matrix example finished.")

# --- END OF FILE sparse_matrix_example.py ---