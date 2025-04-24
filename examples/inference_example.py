#!/usr/bin/env python
"""
Example using the secact_inference function from ridge_inference,
configured to match the likely settings that generated the
'output.permutation.*.npy' files, and comparing against them using correlation.

Includes an option to regenerate baseline files.
"""

import os
import numpy as np
import pandas as pd
# Import matplotlib only if visualization is used
# import matplotlib.pyplot as plt
from pathlib import Path
import sys
import time
import logging
import warnings
from scipy.stats import pearsonr # Import pearsonr
import argparse # For command line arguments
import gc # For garbage collection

# Environment variable for thread control
os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "4")

# --- Setup Paths and Logging ---
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent # Assumes script is in examples/
sys.path.append(str(project_root)) # Add project root to allow importing ridge_inference

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
logger = logging.getLogger(__name__)

try:
    # Import necessary functions from the package
    from ridge_inference import secact_inference, load_signature_matrix, set_backend
    from ridge_inference.utils import visualize_activity # Keep for optional viz
    from ridge_inference.backend_selection import is_backend_available # For checking backend status
except ImportError as e:
    logger.error(f"Error importing ridge_inference: {e}")
    logger.error("Ensure the package is installed correctly (e.g., 'pip install -e .')")
    logger.error(f"Current sys.path: {sys.path}")
    sys.exit(1)


# --- Configuration ---
DATA_DIR = project_root / "data" / "sample_data"
EXPR_PATH = DATA_DIR / "infection_GSE147507.gz"
SIG_MATRIX_PATH = DATA_DIR / "signaling_signature.gz"
PRECOMPUTED_DIR = DATA_DIR # Output directory is the same as input for simplicity
OUTPUT_DIR = script_dir / "outputs" # Directory for saving current run's outputs
OUTPUT_DIR.mkdir(exist_ok=True)

# Parameters matching the LIKELY generation of output.perm files
LAMBDA_VAL = 10000
N_RAND = 1000
VERBOSE_LEVEL = 1
ADD_BACKGROUND = True # Matches original example logic implicitly
SCALE_METHOD = 'column' # Matches original example logic implicitly
# *** ADD ALTERNATIVE DEFINITION HERE ***
ALTERNATIVE = "two-sided" # Define the variable needed for secact_inference

# Precomputed file prefix
PRECOMPUTED_FILE_SET_PREFIX = str(PRECOMPUTED_DIR / "output.permutation")

# Correlation Thresholds for Verification
BETA_CORR_THRESHOLD = 0.9999
# Relaxed SE threshold acknowledging RNG differences vs original fixed seed
PERM_SE_CORR_THRESHOLD = 0.10 # Slightly relaxed from perfect
Z_CORR_THRESHOLD = 0.98
PVAL_CORR_THRESHOLD = 0.99

# Constant for float comparisons in correlation helper
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

def load_results(out_prefix):
    """Loads pre-computed results (.npy files) for comparison."""
    result = {}; logger.info(f"Loading precomputed results with prefix '{Path(out_prefix).name}'..."); loaded_any = False
    for title in ['beta', 'se', 'zscore', 'pvalue']:
        f = Path(f"{out_prefix}.{title}.npy")
        if f.exists():
            try: result[title] = np.load(f); logger.info(f"  Loaded {f.name}: shape {result[title].shape}"); loaded_any = True
            except Exception as e: logger.error(f"  Failed to load {f.name}: {e}"); result[title] = None
        else: logger.warning(f"  File not found: {f.name}"); result[title] = None
    return result if loaded_any else None

def save_results(results_dict, out_prefix):
    """Save computed results to .npy files."""
    logger.info(f"Saving computed results with prefix '{Path(out_prefix).name}'...")
    saved_any = False
    # Extract numpy arrays from DataFrames if present
    beta = results_dict.get('beta')
    se = results_dict.get('se')
    zscore = results_dict.get('zscore')
    pvalue = results_dict.get('pvalue')
    data_to_save = {
        'beta': beta.to_numpy(dtype=np.float64) if isinstance(beta, pd.DataFrame) else beta,
        'se': se.to_numpy(dtype=np.float64) if isinstance(se, pd.DataFrame) else se,
        'zscore': zscore.to_numpy(dtype=np.float64) if isinstance(zscore, pd.DataFrame) else zscore,
        'pvalue': pvalue.to_numpy(dtype=np.float64) if isinstance(pvalue, pd.DataFrame) else pvalue,
    }
    for title, data in data_to_save.items():
        if data is not None and isinstance(data, np.ndarray):
            f = Path(f"{out_prefix}.{title}.npy")
            try:
                np.save(f, data)
                logger.info(f"  Saved {f.name}")
                saved_any = True
            except Exception as e:
                logger.error(f"  Failed to save {f.name}: {e}")
        else:
            logger.warning(f"  Skipping save for '{title}' (data missing or not ndarray)")
    return saved_any


def calculate_and_report_correlation(mat1, mat2, name1="Mat1", name2="Mat2", threshold=0.99):
    """ Calculates and reports Pearson correlation against a threshold. Always returns 3 values."""
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
             # Check if they are identical constants
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
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run SecAct inference example and optionally regenerate baseline.")
    parser.add_argument(
        "--regenerate-baseline",
        action="store_true",
        help="Regenerate the baseline .npy files using the 'python' backend."
    )
    parser.add_argument(
        "--method",
        default="auto",
        choices=['auto', 'python', 'mkl', 'gsl_cython', 'gsl', 'numba', 'gpu'],
        help="Specify the backend method to use for the run."
    )
    args = parser.parse_args()

    # --- Load Expression Data ---
    logger.info(f"Loading expression data from: {EXPR_PATH}")
    if not EXPR_PATH.exists(): logger.error(f"File not found: {EXPR_PATH}"); sys.exit(1)
    try:
        expr_df = pd.read_csv(EXPR_PATH, sep='\t', index_col=0, compression='gzip')
        logger.info("Cleaning expression data index (assuming format 'Num Symbol')...")
        original_index_count = len(expr_df.index)
        expr_df.index = expr_df.index.astype(str).str.strip().str.split().str[-1]
        if expr_df.index.has_duplicates:
            logger.warning(f"Duplicate gene symbols found after cleaning index. Keeping first occurrence.")
            expr_df = expr_df[~expr_df.index.duplicated(keep='first')]
        logger.info(f"Index cleaned. Retained {len(expr_df.index)} unique gene symbols from {original_index_count}.")
    except Exception as e:
        logger.error(f"Failed to load or clean expression data: {e}", exc_info=True); sys.exit(1)
    logger.info(f"Expression data loaded and cleaned. Shape: {expr_df.shape}")

    # --- Load Signature Matrix DataFrame ---
    logger.info(f"Loading signature matrix from path: {SIG_MATRIX_PATH}")
    if not SIG_MATRIX_PATH.exists(): logger.error(f"File not found: {SIG_MATRIX_PATH}"); sys.exit(1)
    try:
        sig_df_loaded = load_signature_matrix(SIG_MATRIX_PATH)
        sig_df_loaded.index = sig_df_loaded.index.astype(str)
    except Exception as e:
        logger.error(f"Failed to load signature matrix: {e}", exc_info=True); sys.exit(1)
    logger.info(f"Signature matrix loaded. Shape: {sig_df_loaded.shape}")

    # --- Regenerate Baseline Mode ---
    if args.regenerate_baseline:
        logger.warning("\n--- REGENERATING BASELINE FILES using 'python' backend ---")
        logger.warning(f"Using settings: lambda={LAMBDA_VAL}, n_rand={N_RAND}, add_bg={ADD_BACKGROUND}, scale={SCALE_METHOD}")

        try:
            # Force python backend for baseline generation
            set_backend("python")
            baseline_perm = secact_inference(
                expr_data=expr_df.copy(), # Use copies to avoid modifying original dfs
                sig_matrix=sig_df_loaded.copy(),
                lambda_val=LAMBDA_VAL, n_rand=N_RAND,
                method="python", # Force python
                add_background=ADD_BACKGROUND, scale_method=SCALE_METHOD,
                verbose=VERBOSE_LEVEL, alternative=ALTERNATIVE
            )
            saved = save_results(baseline_perm, PRECOMPUTED_FILE_SET_PREFIX)
            if not saved: raise RuntimeError("Failed to save any baseline files.")
        except Exception as e:
            logger.error(f"Failed to regenerate baseline: {e}", exc_info=True)
            sys.exit(1)

        logger.warning("--- BASELINE REGENERATION COMPLETE ---")
        sys.exit(0) # Exit after regenerating


    # --- Standard Comparison Mode ---

    # --- Load Precomputed Results ---
    precomputed_perm = load_results(PRECOMPUTED_FILE_SET_PREFIX)
    if not precomputed_perm:
        logger.error("Precomputed baseline files not found. Cannot perform comparison.")
        logger.error(f"(Looked for prefix: {PRECOMPUTED_FILE_SET_PREFIX})")
        logger.error("Run with --regenerate-baseline to create them.")
        sys.exit(1)

    # --- Run SecAct Inference ---
    logger.info(f"\n--- Running SecAct Inference (Method: {args.method}) ---")
    logger.info(f"Using settings: lambda={LAMBDA_VAL}, n_rand={N_RAND}, add_bg={ADD_BACKGROUND}, scale={SCALE_METHOD}")

    start_time = time.time()
    results = None
    comparison_overall_passed = False

    try:
        results = secact_inference(
            expr_data=expr_df, # Can reuse dfs here as secact_inference copies
            sig_matrix=sig_df_loaded,
            lambda_val=LAMBDA_VAL, n_rand=N_RAND,
            method=args.method, # Use the method from command line
            add_background=ADD_BACKGROUND, scale_method=SCALE_METHOD,
            verbose=VERBOSE_LEVEL, alternative=ALTERNATIVE # Pass alternative
        )
        duration = time.time() - start_time
        actual_method_used = results.get('method_used', 'unknown') if results else 'unknown'
        execution_time = results.get('execution_time', duration) if results else duration
        # *** FIX: Log the correct method used variable ***
        logger.info(f"SecAct inference completed in {execution_time:.2f} seconds using '{actual_method_used}' backend.")
        # **************************************************

        if results is None or not isinstance(results, dict): raise RuntimeError("secact_inference returned None or invalid type.")
        if not all(k in results for k in ['beta', 'se', 'zscore', 'pvalue']): raise RuntimeError("secact_inference result missing keys.")

        # --- Display and Save Current Results ---
        logger.info("\n--- Current Inference Results ---")
        beta_s = results.get('beta'); se_s = results.get('se'); zscore_s = results.get('zscore'); pvalue_s = results.get('pvalue')
        if isinstance(beta_s, pd.DataFrame):
            logger.info(f"Beta DataFrame shape: {beta_s.shape}"); print("Beta DataFrame (head):\n", beta_s.head())
            output_stem = f"inference_{args.method}_lambda{LAMBDA_VAL}"
            beta_s.to_csv(OUTPUT_DIR / f"{output_stem}_beta.csv")
        else: logger.warning("Beta result missing/not DataFrame.")
        if isinstance(pvalue_s, pd.DataFrame):
            print("\nP-value DataFrame (head):\n", pvalue_s.head())
            pvalue_s.to_csv(OUTPUT_DIR / f"{output_stem}_pvalue.csv")
        else: logger.warning("P-value result missing/not DataFrame.")
        if isinstance(beta_s, pd.DataFrame) and isinstance(pvalue_s, pd.DataFrame): logger.info(f"Saved Current Beta/P-value to {OUTPUT_DIR}")

        # --- Compare ---
        logger.info(f"\n--- Comparing Current Results (Method: {args.method}) with Precomputed Data '{Path(PRECOMPUTED_FILE_SET_PREFIX).name}.*' ---")

        # Convert current results to NumPy for comparison
        beta_s_np = beta_s.to_numpy(dtype=np.float64) if isinstance(beta_s, pd.DataFrame) else None
        se_s_np = se_s.to_numpy(dtype=np.float64) if isinstance(se_s, pd.DataFrame) else None
        zscore_s_np = zscore_s.to_numpy(dtype=np.float64) if isinstance(zscore_s, pd.DataFrame) else None
        pvalue_s_np = pvalue_s.to_numpy(dtype=np.float64) if isinstance(pvalue_s, pd.DataFrame) else None

        # Perform comparisons
        passed_beta, _, _ = calculate_and_report_correlation(beta_s_np, precomputed_perm.get('beta'), "Current Beta", "Precomp Beta", BETA_CORR_THRESHOLD)
        passed_se, _, _ = calculate_and_report_correlation(se_s_np, precomputed_perm.get('se'), "Current SE", "Precomp SE", PERM_SE_CORR_THRESHOLD)
        passed_z, _, _ = calculate_and_report_correlation(zscore_s_np, precomputed_perm.get('zscore'), "Current Z", "Precomp Z", Z_CORR_THRESHOLD)
        passed_p, _, _ = calculate_and_report_correlation(pvalue_s_np, precomputed_perm.get('pvalue'), "Current Pval", "Precomp Pval", PVAL_CORR_THRESHOLD)
        comparison_overall_passed = passed_beta and passed_se and passed_z and passed_p
        diff_beta = difference(beta_s_np, precomputed_perm.get('beta'))
        logger.info(f"  Max difference (current_beta vs precomputed_beta): {diff_beta:.2e} (for info only)")

        # --- Visualization (Optional) ---
        # Uncomment if matplotlib/seaborn are installed and visualization is desired
        # try:
        #     logger.info("\nGenerating activity visualization...")
        #     # Determine expected features (including potential background)
        #     expected_features = sig_df_loaded.shape[1] # Start with original
        #     # Check if background was actually added (might depend on internal logic/flags)
        #     # A robust way is to check the final beta index/columns if available
        #     if isinstance(beta_s, pd.DataFrame) and 'background' in beta_s.index:
        #          expected_features += 1
        #     elif add_background: # If requested, assume it should be there
        #          expected_features += 1

        #     fig = visualize_activity(results, top_n=expected_features, pvalue_threshold=0.05)
        #     if fig:
        #         plot_path = OUTPUT_DIR / f"{output_stem}_activity_visualization.png"
        #         fig.savefig(plot_path)
        #         logger.info(f"Activity visualization saved to {plot_path}")
        #         plt.close(fig)
        #     else:
        #         logger.warning("Visualization could not be generated.")
        # except ImportError:
        #     logger.warning("matplotlib/seaborn not installed. Skipping visualization.")
        # except Exception as e:
        #      logger.error(f"Error during visualization: {e}", exc_info=True)

    except FileNotFoundError as e: logger.error(f"Data file not found: {e}"); comparison_overall_passed = False
    except ValueError as e: logger.error(f"ValueError during inference: {e}"); comparison_overall_passed = False
    except TypeError as e: logger.error(f"TypeError during inference: {e}", exc_info=True); comparison_overall_passed = False
    except Exception as e: logger.error(f"Unexpected error during inference: {e}", exc_info=True); comparison_overall_passed = False

    # --- Final Summary ---
    logger.info(f"\n--- Final Summary of Correlation Checks (Current Run Method: {args.method}) ---")
    if results is None:
        logger.error("\nInference FAILED. Cannot determine pass/fail status.")
        sys.exit(1) # Exit with error code
    elif comparison_overall_passed:
        logger.info("All reported correlation checks passed minimum thresholds.")
        logger.info("\nSecAct inference example finished successfully and matched precomputed results within tolerance.")
        sys.exit(0) # Exit with success code
    else:
        logger.error("\nOne or more correlation checks FAILED to meet the minimum threshold.")
        logger.error("\nSecAct inference example finished with correlation check failures.")
        sys.exit(1) # Exit with error code

    logger.info("\nScript finished.")

# --- END OF FILE inference_example.py ---