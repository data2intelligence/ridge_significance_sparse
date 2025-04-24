# tests/simple_test.py

import os
import numpy as np
import logging
import sys
import traceback
from scipy import sparse as sps
import time

# --- Configuration ---
OMP_THREADS = "1" # Keep minimal for testing
MKL_THREADS = "1" # Keep minimal for testing
SEED = 42
N_GENES = 50       # Minimal dimensions
N_FEATURES = 5
N_SAMPLES = 10     # Slightly more samples
LAMBDA_VAL = 1.0   # Minimal lambda
N_RAND = 5         # Minimal permutations
SPARSITY = 0.1     # Sparsity level for sparse matrices

# Set threads early
os.environ["OMP_NUM_THREADS"] = OMP_THREADS
os.environ["MKL_NUM_THREADS"] = MKL_THREADS

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)-8s - %(name)-15s - %(message)s'
)
logger = logging.getLogger("RidgeTest")

# Try importing the main package components
try:
    from ridge_inference import ridge, set_backend, list_available_backends
    from ridge_inference.backend_selection import (
        MKL_AVAILABLE, GSL_CYTHON_AVAILABLE, NUMBA_AVAILABLE, CUPY_AVAILABLE, PYTHON_AVAILABLE
    )
    # Check CuPy separately for GPU test skipping
    try:
        import cupy
        GPU_ENABLED_TEST = CUPY_AVAILABLE # Use flag from package
    except ImportError:
        GPU_ENABLED_TEST = False

    # Check MKL separately for MKL test skipping
    MKL_ENABLED_TEST = MKL_AVAILABLE

    # Check GSL Cython separately for skipping
    GSL_ENABLED_TEST = GSL_CYTHON_AVAILABLE

    # Check Numba separately for skipping
    NUMBA_ENABLED_TEST = NUMBA_AVAILABLE

    PACKAGE_IMPORT_SUCCESS = True
    logger.info("Successfully imported ridge_inference package components.")
    logger.info(f"Backend Availability: MKL={MKL_ENABLED_TEST}, GSL={GSL_ENABLED_TEST}, Numba={NUMBA_ENABLED_TEST}, GPU={GPU_ENABLED_TEST}, Python={PYTHON_AVAILABLE}")

except ImportError as e:
    logger.error(f"FATAL: Failed to import ridge_inference package: {e}")
    PACKAGE_IMPORT_SUCCESS = False
    # Define flags as False if import failed
    MKL_ENABLED_TEST = GSL_ENABLED_TEST = NUMBA_ENABLED_TEST = GPU_ENABLED_TEST = PYTHON_AVAILABLE = False

# --- Generate Test Data ---
def generate_data(sparse_x=False, sparse_y=False):
    logger.debug(f"Generating data: sparse_x={sparse_x}, sparse_y={sparse_y}")
    np.random.seed(SEED)
    X_dense = np.random.randn(N_GENES, N_FEATURES).astype(np.float64)
    Y_dense = np.random.randn(N_GENES, N_SAMPLES).astype(np.float64)

    X = sps.random(N_GENES, N_FEATURES, density=SPARSITY, format='csr', dtype=np.float64, random_state=SEED) if sparse_x else X_dense
    Y = sps.random(N_GENES, N_SAMPLES, density=SPARSITY, format='csr', dtype=np.float64, random_state=SEED+1) if sparse_y else Y_dense # Use diff seed for Y

    # Ensure correct dtypes and contiguity for dense inputs needed by some backends
    if not sparse_x:
        X = np.require(X, dtype=np.float64, requirements=['C', 'A'])
    if not sparse_y:
        Y = np.require(Y, dtype=np.float64, requirements=['C', 'A'])

    logger.debug(f"Data generated: X type={type(X).__name__}, Y type={type(Y).__name__}")
    return X, Y

# --- Test Runner Function ---
def run_single_test(backend_name, X, Y, n_rand=N_RAND):
    """Runs ridge regression for a specific backend and input data."""
    logger.info(f"--- Testing backend: '{backend_name}' (n_rand={n_rand}) ---")
    is_X_sparse = sps.issparse(X)
    is_Y_sparse = sps.issparse(Y)
    logger.info(f"Input types: X={'Sparse' if is_X_sparse else 'Dense'}, Y={'Sparse' if is_Y_sparse else 'Dense'}")

    start_time = time.time()
    try:
        # Use the main ridge function, explicitly setting the method
        # Verbosity level 1 for more insight during tests
        result = ridge(X=X, Y=Y, lambda_=LAMBDA_VAL, n_rand=n_rand, method=backend_name, verbose=1)
        duration = time.time() - start_time

        # Basic checks on results
        if not isinstance(result, dict):
            raise TypeError(f"Expected dict result, got {type(result)}")
        for key in ['beta', 'se', 'zscore', 'pvalue']:
            if key not in result or not isinstance(result[key], np.ndarray):
                raise ValueError(f"Result missing or invalid type for key: '{key}'")
            if result[key].shape != (N_FEATURES, N_SAMPLES):
                 # Allow p=0 case for t-test if X is rank deficient? Check result shapes
                 # For now, strict check:
                 raise ValueError(f"Unexpected shape for key '{key}': {result[key].shape}, expected {(N_FEATURES, N_SAMPLES)}")
            if np.isnan(result[key]).all():
                logger.warning(f"Result key '{key}' contains all NaNs.")
            # Allow NaNs for now, deeper tests would compare values

        method_used = result.get('method_used', 'unknown')
        logger.info(f"Backend '{backend_name}' SUCCEEDED in {duration:.3f}s. Method reported: '{method_used}'")
        return True, method_used

    except ImportError:
        duration = time.time() - start_time
        logger.warning(f"Backend '{backend_name}' SKIPPED (ImportError - likely not installed/available). Duration: {duration:.3f}s")
        return False, "skipped_import_error"
    except NotImplementedError as nie:
        duration = time.time() - start_time
        logger.warning(f"Backend '{backend_name}' SKIPPED (NotImplementedError: {nie}). Duration: {duration:.3f}s")
        return False, "skipped_not_implemented"
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Backend '{backend_name}' FAILED in {duration:.3f}s: {type(e).__name__} - {e}")
        if logger.level <= logging.DEBUG: # Only print traceback if verbose
             traceback.print_exc()
        return False, f"failed_{type(e).__name__}"

# --- Test Scenarios ---
test_results = {}

def test_dense_dense():
    """Tests backends compatible with Dense X and Dense Y."""
    logger.info("\n===== Scenario: Dense X / Dense Y =====")
    X, Y = generate_data(sparse_x=False, sparse_y=False)
    backends_to_test = []
    if PYTHON_AVAILABLE: backends_to_test.append('python')
    if GPU_ENABLED_TEST: backends_to_test.append('gpu')
    if GSL_ENABLED_TEST: backends_to_test.append('gsl_cython')
    if MKL_ENABLED_TEST: backends_to_test.append('mkl')      
    if NUMBA_ENABLED_TEST: backends_to_test.append('numba')

    scenario_results = {}
    for backend in backends_to_test:
        success, detail = run_single_test(backend, X.copy(), Y.copy(), n_rand=N_RAND)
        scenario_results[backend] = {'success': success, 'detail': detail}
        # Also test t-test (n_rand=0) for python backend
        if backend == 'python':
             logger.info("--- Testing backend: 'python' (t-test, n_rand=0) ---")
             success_tt, detail_tt = run_single_test('python', X.copy(), Y.copy(), n_rand=0)
             scenario_results['python_ttest'] = {'success': success_tt, 'detail': detail_tt}

    test_results["Dense_Dense"] = scenario_results

def test_dense_sparse():
    """Tests backends compatible with Dense X and Sparse Y."""
    logger.info("\n===== Scenario: Dense X / Sparse Y =====")
    X, Y = generate_data(sparse_x=False, sparse_y=True)
    backends_to_test = []
    if PYTHON_AVAILABLE: backends_to_test.append('python')
    if GPU_ENABLED_TEST: backends_to_test.append('gpu')
    if MKL_ENABLED_TEST: backends_to_test.append('mkl') 
    if NUMBA_ENABLED_TEST: backends_to_test.append('numba') 

    scenario_results = {}
    for backend in backends_to_test:
        success, detail = run_single_test(backend, X.copy(), Y.copy(), n_rand=N_RAND)
        scenario_results[backend] = {'success': success, 'detail': detail}
        # T-test only for python backend (should handle sparse Y)
        if backend == 'python':
             logger.info("--- Testing backend: 'python' (t-test, n_rand=0) ---")
             success_tt, detail_tt = run_single_test('python', X.copy(), Y.copy(), n_rand=0)
             scenario_results['python_ttest'] = {'success': success_tt, 'detail': detail_tt}

    test_results["Dense_Sparse"] = scenario_results

def test_sparse_sparse():
    """Tests backends compatible with Sparse X and Sparse Y."""
    logger.info("\n===== Scenario: Sparse X / Sparse Y =====")
    X, Y = generate_data(sparse_x=True, sparse_y=True)
    # Currently, only the python backend's iterative solver handles sparse X
    backends_to_test = []
    if PYTHON_AVAILABLE: backends_to_test.append('python')

    scenario_results = {}
    for backend in backends_to_test:
        success, detail = run_single_test(backend, X.copy(), Y.copy(), n_rand=N_RAND)
        scenario_results[backend] = {'success': success, 'detail': detail}
        # T-test (n_rand=0) is not supported for sparse X with iterative solver
        if backend == 'python':
            scenario_results['python_ttest'] = {'success': False, 'detail': 'skipped_not_supported'}


    test_results["Sparse_Sparse"] = scenario_results


# --- Main Execution ---
if __name__ == "__main__":
    if not PACKAGE_IMPORT_SUCCESS:
        logger.error("Exiting due to package import failure.")
        sys.exit(1)

    overall_success = True

    test_dense_dense()
    test_dense_sparse()
    test_sparse_sparse()

    # --- Print Final Summary ---
    print("\n" + "="*25 + " FINAL TEST SUMMARY " + "="*25)
    logger.info("Final Test Summary:")
    for scenario, results in test_results.items():
        print(f"\n--- {scenario.replace('_', ' / ')} ---")
        logger.info(f"--- {scenario.replace('_', ' / ')} ---")
        all_passed_scenario = True
        for backend, outcome in results.items():
            status_symbol = "✅" if outcome['success'] else ("⚠️" if "skipped" in outcome['detail'] else "❌")
            status_text = "PASSED" if outcome['success'] else ("SKIPPED" if "skipped" in outcome['detail'] else "FAILED")
            log_message = f"{backend:<15}: {status_symbol} {status_text} ({outcome['detail']})"
            print(log_message)
            logger.info(log_message)
            if not outcome['success'] and "skipped" not in outcome['detail']:
                all_passed_scenario = False
                overall_success = False
        if not all_passed_scenario:
             logger.warning(f"One or more tests FAILED in scenario: {scenario}")


    print("="*70)
    logger.info("="*70)

    if overall_success:
        logger.info("All essential backend tests passed or were skipped due to availability.")
        print("\nOverall Result: SUCCESS")
        sys.exit(0)
    else:
        logger.error("One or more essential backend tests FAILED.")
        print("\nOverall Result: FAILED")
        sys.exit(1)