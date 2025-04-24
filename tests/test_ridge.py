#!/usr/bin/env python
"""
Test suite for the ridge regression functionality in the ridge_inference package.
"""

import unittest
import numpy as np
import pandas as pd
import os
import sys
import logging
import time
from scipy import sparse as sps # Import sparse module
import gc # Import garbage collector

# Ensure parent directory is in the path to import ridge_inference
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import necessary components from the package
try:
    from ridge_inference import ridge, list_available_backends
    # Import specific backend functions for direct testing and availability flags
    from ridge_inference.backend_selection import (
        MKL_AVAILABLE, GSL_CYTHON_AVAILABLE, NUMBA_AVAILABLE, CUPY_AVAILABLE, PYTHON_AVAILABLE,
        is_backend_available
    )
    from ridge_inference.core import (
        ridge_regression_numpy,
        ridge_regression_numba,
        ridge_regression_cupy
    )
    # Try to import Cython/MKL/GSL implementations specifically
    try:
        # Import the CORRECT GSL function from the CORRECT module
        from ridge_inference.ridge_gsl import ridge_regression_gsl
    except ImportError:
        # Function is already defined as raising ImportError by backend_selection if unavailable
        # No need for a separate import here, rely on the GSL_CYTHON_AVAILABLE flag later
        from ridge_inference.backend_selection import ridge_regression_gsl # Keep dummy import

    try:
        from ridge_inference.ridge_mkl import ridge_regression_mkl
    except ImportError:
        # Function is already defined as raising ImportError by backend_selection if unavailable
        from ridge_inference.backend_selection import ridge_regression_mkl # Keep dummy import

except ImportError as e:
    print(f"FATAL: Failed to import ridge_inference components: {e}")
    print(f"Current sys.path: {sys.path}")
    # Exit if core components can't be imported, tests cannot run
    sys.exit(f"Required ridge_inference components not found. Installation might be incomplete or sys.path incorrect.")


# Set up logging for tests
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)-8s - %(name)s - %(message)s')
logger = logging.getLogger("TestRidge")


class TestRidgeRegression(unittest.TestCase):
    """Test class for ridge regression functionality."""

    @classmethod
    def setUpClass(cls):
        """Set up test data once for all tests."""
        np.random.seed(42)
        cls.n_genes = 100
        cls.n_samples = 20
        cls.n_features = 10 # Renamed from n_proteins for clarity

        # Dense data
        cls.X_dense = np.random.randn(cls.n_genes, cls.n_features).astype(np.float64) * 2.0 + 1.0
        cls.Y_dense = np.random.randn(cls.n_genes, cls.n_samples).astype(np.float64) * 1.5 + 0.5

        # Ensure C-contiguity for backends that require it
        cls.X_dense = np.require(cls.X_dense, requirements=['C', 'A'])
        cls.Y_dense = np.require(cls.Y_dense, requirements=['C', 'A'])

        # Sparse Y data (CSR format)
        cls.Y_sparse = sps.random(cls.n_genes, cls.n_samples, density=0.1, format='csr', random_state=42, dtype=np.float64)
        cls.Y_sparse.data = np.random.randn(len(cls.Y_sparse.data)) * 1.5 + 0.5 # Assign realistic values

        # Sparse X data (CSR format) - only used for python backend test
        cls.X_sparse = sps.random(cls.n_genes, cls.n_features, density=0.15, format='csr', random_state=43, dtype=np.float64)
        cls.X_sparse.data = np.random.randn(len(cls.X_sparse.data)) * 2.0 + 1.0

        cls.lambda_val = 1000.0 # Use float
        cls.n_rand = 50 # Reduced for faster testing, increase for more robust checks

        logger.info("-" * 40)
        logger.info("Test Environment Configuration:")
        # Use the imported availability flags
        logger.info(f" Python backend available: {PYTHON_AVAILABLE}")
        logger.info(f" MKL backend available:    {MKL_AVAILABLE}")
        logger.info(f" GSL Cython available:   {GSL_CYTHON_AVAILABLE}")
        logger.info(f" Numba available:        {NUMBA_AVAILABLE}")
        logger.info(f" CuPy (GPU) available:   {CUPY_AVAILABLE}")
        logger.info("-" * 40)

        # Cache for potentially slow test results
        cls.cache = {}
        # Pre-run and cache results for available backends to speed up consistency checks
        cls._pre_run_backends()

    @classmethod
    def _pre_run_backends(cls):
        """Run each available backend once to cache results."""
        logger.info("Pre-running backends to cache results...")
        # Dense X / Dense Y scenarios
        if PYTHON_AVAILABLE: cls._run_and_cache('numpy_dense', ridge_regression_numpy, cls.X_dense, cls.Y_dense, cls.lambda_val, cls.n_rand)
        if MKL_AVAILABLE: cls._run_and_cache('mkl_dense', ridge_regression_mkl, cls.X_dense, cls.Y_dense, cls.lambda_val, cls.n_rand)
        # *** USE CORRECT FUNCTION NAME HERE ***
        if GSL_CYTHON_AVAILABLE: cls._run_and_cache('gsl_cython_dense', ridge_regression_gsl, cls.X_dense, cls.Y_dense, cls.lambda_val, cls.n_rand)
        if NUMBA_AVAILABLE: cls._run_and_cache('numba_dense', ridge_regression_numba, cls.X_dense, cls.Y_dense, cls.lambda_val, cls.n_rand)
        if CUPY_AVAILABLE: cls._run_and_cache('cupy_dense', ridge_regression_cupy, cls.X_dense, cls.Y_dense, cls.lambda_val, cls.n_rand)
        # Dense X / Sparse Y scenarios
        if PYTHON_AVAILABLE: cls._run_and_cache('numpy_sparse', ridge_regression_numpy, cls.X_dense, cls.Y_sparse, cls.lambda_val, cls.n_rand)
        if MKL_AVAILABLE: cls._run_and_cache('mkl_sparse', ridge_regression_mkl, cls.X_dense, cls.Y_sparse, cls.lambda_val, cls.n_rand)
        if NUMBA_AVAILABLE: cls._run_and_cache('numba_sparse', ridge_regression_numba, cls.X_dense, cls.Y_sparse, cls.lambda_val, cls.n_rand)
        if CUPY_AVAILABLE: cls._run_and_cache('cupy_sparse', ridge_regression_cupy, cls.X_dense, cls.Y_sparse, cls.lambda_val, cls.n_rand)
        # Sparse X / Sparse Y scenario (only Python)
        if PYTHON_AVAILABLE: cls._run_and_cache('python_sparse_X', ridge_regression_numpy, cls.X_sparse, cls.Y_sparse, cls.lambda_val, cls.n_rand)
        logger.info("Backend pre-run complete.")

    @classmethod
    def _run_and_cache(cls, method_name, func, *args, **kwargs):
        """Helper to run a backend function and cache its results."""
        if method_name in cls.cache:
            return cls.cache[method_name] # Return cached result

        logger.info(f"Running {method_name} implementation for caching...")
        start_time = time.time()
        result = None
        try:
            # Check backend availability *before* calling (relevant for MKL/GSL)
            if method_name.startswith('mkl') and not MKL_AVAILABLE:
                raise ImportError("MKL backend not available")
            if method_name.startswith('gsl_cython') and not GSL_CYTHON_AVAILABLE:
                 raise ImportError("GSL backend not available")
            if method_name.startswith('numba') and not NUMBA_AVAILABLE:
                 raise ImportError("Numba backend not available")
            if method_name.startswith('cupy') and not CUPY_AVAILABLE:
                 raise ImportError("CuPy backend not available")

            result = func(*args, **kwargs)
            # Perform basic validation before caching
            if isinstance(result, dict) and all(k in result for k in ['beta', 'se', 'zscore', 'pvalue']):
                 # Check shapes immediately
                 expected_shape = (cls.n_features, cls.n_samples)
                 if result['beta'].shape != expected_shape:
                     raise ValueError(f"{method_name} returned incorrect beta shape: {result['beta'].shape}, expected {expected_shape}")
                 cls.cache[method_name] = result
                 logger.info(f"{method_name} completed successfully in {time.time() - start_time:.3f} seconds.")
            else:
                 logger.error(f"{method_name} returned unexpected result structure: {type(result)}")
                 cls.cache[method_name] = None # Cache failure explicitly
        except (ImportError, NotImplementedError) as skip_err:
             logger.warning(f"{method_name} skipped: {skip_err}")
             cls.cache[method_name] = "skipped" # Mark as skipped
        except Exception as e:
            logger.error(f"Error running {method_name}: {e}", exc_info=True)
            cls.cache[method_name] = None # Cache failure explicitly

        gc.collect() # Suggest garbage collection after each run
        return cls.cache[method_name]

    def _assert_result_dict(self, result, method_name):
        """Assert basic structure and shape of result dict."""
        if result == "skipped":
            self.skipTest(f"Backend {method_name} was skipped (likely unavailable or not applicable).")
            return # Skip further checks

        self.assertIsNotNone(result, f"{method_name} result was None (indicates failure during execution)")
        self.assertIsInstance(result, dict, f"{method_name} result should be a dictionary")
        keys = ['beta', 'se', 'zscore', 'pvalue']
        for key in keys:
            self.assertIn(key, result, f"{method_name} result missing key '{key}'")
            self.assertIsInstance(result[key], np.ndarray, f"{method_name} result['{key}'] not ndarray")
            self.assertEqual(result[key].shape, (self.n_features, self.n_samples),
                             f"{method_name} result['{key}'] shape mismatch: got {result[key].shape} expected {(self.n_features, self.n_samples)}")
            # Check for excessive NaNs (allow some for potentially ill-conditioned test data)
            nan_fraction = np.isnan(result[key]).sum() / result[key].size
            self.assertLess(nan_fraction, 0.5, f"{method_name} result['{key}'] has too many NaNs ({nan_fraction:.1%})")

    # --- Individual Backend Tests ---

    def test_01_ridge_regression_numpy_dense(self):
        """Test NumPy backend with dense Y."""
        result = self.cache.get('numpy_dense')
        self._assert_result_dict(result, "numpy_dense")

    def test_02_ridge_regression_numpy_sparse(self):
        """Test NumPy backend with sparse Y."""
        result = self.cache.get('numpy_sparse')
        self._assert_result_dict(result, "numpy_sparse")

    def test_03_ridge_regression_numpy_sparse_X(self):
        """Test NumPy backend with sparse X (iterative)."""
        result = self.cache.get('python_sparse_X')
        self._assert_result_dict(result, "python_sparse_X")

    @unittest.skipIf(not MKL_AVAILABLE, "MKL backend not available")
    def test_10_ridge_regression_mkl_dense(self):
        """Test MKL backend with dense Y."""
        result = self.cache.get('mkl_dense')
        self._assert_result_dict(result, "mkl_dense")

    @unittest.skipIf(not MKL_AVAILABLE, "MKL backend not available")
    def test_11_ridge_regression_mkl_sparse(self):
        """Test MKL backend with sparse Y."""
        result = self.cache.get('mkl_sparse')
        self._assert_result_dict(result, "mkl_sparse")

    @unittest.skipIf(not GSL_CYTHON_AVAILABLE, "GSL Cython backend not available")
    def test_20_ridge_regression_gsl_cython_dense(self):
        """Test GSL Cython backend (requires dense Y)."""
        result = self.cache.get('gsl_cython_dense')
        self._assert_result_dict(result, "gsl_cython_dense")

    @unittest.skipIf(not NUMBA_AVAILABLE, "Numba backend not available")
    def test_30_ridge_regression_numba_dense(self):
        """Test Numba backend with dense Y."""
        result = self.cache.get('numba_dense')
        self._assert_result_dict(result, "numba_dense")

    @unittest.skipIf(not NUMBA_AVAILABLE, "Numba backend not available")
    def test_31_ridge_regression_numba_sparse(self):
        """Test Numba backend with sparse Y."""
        result = self.cache.get('numba_sparse')
        self._assert_result_dict(result, "numba_sparse")

    @unittest.skipIf(not CUPY_AVAILABLE, "CuPy (GPU) backend not available")
    def test_40_ridge_regression_cupy_dense(self):
        """Test GPU backend with dense Y."""
        result = self.cache.get('cupy_dense')
        self._assert_result_dict(result, "cupy_dense")

    @unittest.skipIf(not CUPY_AVAILABLE, "CuPy (GPU) backend not available")
    def test_41_ridge_regression_cupy_sparse(self):
        """Test GPU backend with sparse Y."""
        result = self.cache.get('cupy_sparse')
        self._assert_result_dict(result, "cupy_sparse")


    # --- Consistency Tests ---
    def test_50_consistency_dense_Y(self):
        """Compare beta results between available backends using Dense Y."""
        logger.info("Comparing backend beta results for Dense Y scenario...")
        ref_result = self.cache.get('numpy_dense')
        if ref_result is None or ref_result == "skipped":
            self.skipTest("Reference NumPy (Dense Y) backend failed or was skipped.")

        # Backends to compare against the NumPy reference
        methods_to_compare = ['mkl_dense', 'gsl_cython_dense', 'numba_dense', 'cupy_dense']

        for method_name in methods_to_compare:
            current_result = self.cache.get(method_name)
            if current_result is None or current_result == "skipped":
                logger.warning(f"Skipping comparison for {method_name} (failed or skipped during pre-run).")
                continue # Skip this comparison

            logger.info(f"Comparing numpy_dense vs {method_name} [BETA only]")
            # Use a reasonably tight tolerance for beta comparison
            beta_rtol=1e-5
            beta_atol=1e-6
            try:
                np.testing.assert_allclose(ref_result['beta'], current_result['beta'], rtol=beta_rtol, atol=beta_atol,
                                           err_msg=f"Beta values differ significantly (numpy_dense vs {method_name})")
                # Compare other results too (optional, might need looser tolerance for stats)
                # se_rtol=1e-4; se_atol=1e-5
                # np.testing.assert_allclose(ref_result['se'], current_result['se'], rtol=se_rtol, atol=se_atol,
                #                           err_msg=f"SE values differ significantly (numpy_dense vs {method_name})")
            except AssertionError as ae:
                # Provide more info on beta failure
                diff = np.abs(ref_result['beta'] - current_result['beta'])
                logger.error(f"Max diff for beta ({method_name}): {np.max(diff)}")
                self.fail(f"Consistency check failed for {method_name} (Dense Y - Beta): {ae}")

        logger.info("Beta consistency checks passed for Dense Y scenario.")


    def test_51_consistency_sparse_Y(self):
        """Compare beta results between available backends using Sparse Y."""
        logger.info("Comparing backend beta results for Dense X / Sparse Y scenario...")
        ref_result = self.cache.get('numpy_sparse')
        if ref_result is None or ref_result == "skipped":
            self.skipTest("Reference NumPy (Sparse Y) backend failed or was skipped.")

        # Backends supporting Dense X / Sparse Y
        methods_to_compare = ['mkl_sparse', 'numba_sparse', 'cupy_sparse']

        for method_name in methods_to_compare:
            current_result = self.cache.get(method_name)
            if current_result is None or current_result == "skipped":
                logger.warning(f"Skipping comparison for {method_name} (failed or skipped during pre-run).")
                continue # Skip this comparison

            logger.info(f"Comparing numpy_sparse vs {method_name} [BETA only]")
            beta_rtol=1e-5
            beta_atol=1e-6
            try:
                np.testing.assert_allclose(ref_result['beta'], current_result['beta'], rtol=beta_rtol, atol=beta_atol,
                                           err_msg=f"Beta values differ significantly (numpy_sparse vs {method_name})")
                # Compare other results too (optional)
                # se_rtol=1e-4; se_atol=1e-5
                # np.testing.assert_allclose(ref_result['se'], current_result['se'], rtol=se_rtol, atol=se_atol,
                #                           err_msg=f"SE values differ significantly (numpy_sparse vs {method_name})")
            except AssertionError as ae:
                 diff = np.abs(ref_result['beta'] - current_result['beta'])
                 logger.error(f"Max diff for beta ({method_name}): {np.max(diff)}")
                 self.fail(f"Consistency check failed for {method_name} (Sparse Y - Beta): {ae}")

        logger.info("Beta consistency checks passed for Sparse Y scenario.")

    # --- Main API Tests ---
    def test_60_ridge_main_api_dispatch(self):
        """Test the main ridge() dispatch API with various methods."""
        logger.info("Testing main ridge API dispatch functionality")

        # List of available methods to test via the main API
        methods_to_test_api = ["auto", "python"]
        if MKL_AVAILABLE: methods_to_test_api.append("mkl")
        if GSL_CYTHON_AVAILABLE: methods_to_test_api.append("gsl_cython")
        if NUMBA_AVAILABLE: methods_to_test_api.append("numba")
        if CUPY_AVAILABLE: methods_to_test_api.append("gpu")

        for method in methods_to_test_api:
            logger.info(f"Testing ridge API with method='{method}' (Dense X / Dense Y)")
            try:
                # Use smaller n_rand for faster API tests
                result_dict = ridge(
                    self.X_dense, self.Y_dense, lambda_=self.lambda_val, n_rand=10,
                    method=method, verbose=0 # Keep verbose low for API test clarity
                )
                self._assert_result_dict(result_dict, f"API_{method}_dense")
                logger.info(f"API call method='{method}' (Dense Y) successful. Used backend: {result_dict.get('method_used', 'N/A')}")
            except Exception as e:
                self.fail(f"API call method='{method}' (Dense Y) failed unexpectedly: {e}")

            # Test sparse Y with compatible methods called via API
            # GSL needs special handling as ridge() densifies input for it
            if method == "gsl_cython":
                 logger.info(f"Testing ridge API with method='{method}' (Dense X / Sparse Y - expecting densification)")
                 if not GSL_CYTHON_AVAILABLE: self.skipTest("GSL Cython not available.")
                 try:
                     result_dict_sp_gsl = ridge(
                          self.X_dense, self.Y_sparse, lambda_=self.lambda_val, n_rand=10,
                          method=method, verbose=0
                     )
                     self._assert_result_dict(result_dict_sp_gsl, f"API_{method}_sparseY_densified")
                     # Check if the intended method was ultimately used (or reported)
                     self.assertTrue(method in result_dict_sp_gsl.get('method_used', '').lower(),
                                     f"Expected '{method}' in method_used, got '{result_dict_sp_gsl.get('method_used', '')}'")
                     logger.info(f"API call method='{method}' (Sparse Y) successfully executed after internal densification.")
                 except Exception as e:
                      self.fail(f"API call method='{method}' (Sparse Y) failed unexpectedly during densification/execution: {e}")

            # Test other backends that handle sparse Y directly
            elif method in ["auto", "python", "gpu", "mkl", "numba"]:
                # Skip if backend not actually available (can happen if MKL/GSL build failed but test runs)
                if not is_backend_available(method) and method != "auto" and method != "python":
                    logger.warning(f"Skipping sparse Y API test for method='{method}' as it is not available.")
                    continue
                logger.info(f"Testing ridge API with method='{method}' (Dense X / Sparse Y)")
                try:
                     result_dict_sp = ridge(
                          self.X_dense, self.Y_sparse, lambda_=self.lambda_val, n_rand=10,
                          method=method, verbose=0
                     )
                     self._assert_result_dict(result_dict_sp, f"API_{method}_sparseY")
                     logger.info(f"API call method='{method}' (Sparse Y) successful. Used backend: {result_dict_sp.get('method_used', 'N/A')}")
                except Exception as e:
                     self.fail(f"API call method='{method}' (Sparse Y) failed unexpectedly: {e}")


    def test_61_ridge_main_api_sparse_X(self):
        """Test the main ridge API dispatch with sparse X (should use python)."""
        logger.info("Testing main ridge API dispatch (Sparse X / Sparse Y)")
        try:
            result_dict = ridge(
                self.X_sparse, self.Y_sparse, lambda_=self.lambda_val, n_rand=10,
                method="auto", verbose=0
            )
            self._assert_result_dict(result_dict, "API_auto_sparseX")
            self.assertIn("python_sparse_iterative", result_dict.get('method_used', ''),
                          "Method used for sparse X (auto) was not python_sparse_iterative")
            logger.info(f"API call (Sparse X, auto) successful. Used backend: {result_dict.get('method_used', 'N/A')}")
        except Exception as e:
            self.fail(f"API call (Sparse X, auto) failed unexpectedly: {e}")

        # Test explicit python request
        try:
            result_dict_py = ridge(
                self.X_sparse, self.Y_sparse, lambda_=self.lambda_val, n_rand=10,
                method="python", verbose=0
            )
            self._assert_result_dict(result_dict_py, "API_python_sparseX")
            self.assertIn("python_sparse_iterative", result_dict_py.get('method_used', ''),
                          "Method used for sparse X (python) was not python_sparse_iterative")
            logger.info(f"API call method='python' (Sparse X) successful.")
        except Exception as e:
            self.fail(f"API call method='python' (Sparse X) failed unexpectedly: {e}")


    def test_70_t_test_mode_api(self):
        """Test t-test mode (n_rand=0) via main API."""
        logger.info("Testing t-test mode (n_rand=0) via ridge API")
        # T-test requires Python or GSL backend and dense X
        tested_backend = False
        if GSL_CYTHON_AVAILABLE:
             logger.info("Testing t-test with GSL backend...")
             try:
                 result_dict = ridge(self.X_dense, self.Y_dense, lambda_=self.lambda_val, n_rand=0, method="gsl_cython", verbose=0)
                 self._assert_result_dict(result_dict, "API_ttest_gsl")
                 self.assertEqual(result_dict.get('method_used', ''), 'gsl_cython', "Method used for t-test was not gsl_cython")
                 self.assertIn('df', result_dict, "DF key missing from GSL t-test result")
                 self.assertFalse(np.isnan(result_dict['df']), "DF is NaN in GSL t-test result")
                 pvalue = result_dict['pvalue']
                 self.assertFalse(np.isnan(pvalue).any(), "T-test P-values contain NaNs (GSL)")
                 self.assertTrue(np.all((pvalue >= 0) & (pvalue <= 1)), "T-test P-values out of [0, 1] range (GSL)")
                 logger.info(f"T-test mode successful via ridge API using GSL.")
                 tested_backend = True
             except Exception as e:
                 self.fail(f"T-test mode with GSL failed unexpectedly: {e}")

        # Always test with Python if available (fallback or primary)
        if PYTHON_AVAILABLE:
            logger.info("Testing t-test with Python backend...")
            try:
                result_dict = ridge(self.X_dense, self.Y_dense, lambda_=self.lambda_val, n_rand=0, method="python", verbose=0)
                self._assert_result_dict(result_dict, "API_ttest_python")
                self.assertEqual(result_dict.get('method_used', ''), 'python', "Method used for t-test was not python")
                self.assertIn('df', result_dict, "DF key missing from Python t-test result") # Numpy version calculates DF too
                self.assertFalse(np.isnan(result_dict['df']), "DF is NaN in Python t-test result")
                pvalue = result_dict['pvalue']
                self.assertFalse(np.isnan(pvalue).any(), "T-test P-values contain NaNs (Python)")
                self.assertTrue(np.all((pvalue >= 0) & (pvalue <= 1)), "T-test P-values out of [0, 1] range (Python)")
                logger.info(f"T-test mode successful via ridge API using Python.")
                tested_backend = True
            except Exception as e:
                self.fail(f"T-test mode with Python failed unexpectedly: {e}")

        if not tested_backend:
             self.skipTest("Neither GSL nor Python backends available for t-test.")


    def test_71_t_test_mode_api_auto(self):
        """Test t-test mode (n_rand=0) via main API using auto (should select gsl or python)."""
        logger.info("Testing t-test mode (n_rand=0) via ridge API (method=auto)")
        # Skip if neither suitable backend is available
        if not GSL_CYTHON_AVAILABLE and not PYTHON_AVAILABLE:
            self.skipTest("Neither GSL nor Python backend available for t-test auto-selection.")
            return

        try:
            result_dict = ridge(
                self.X_dense, self.Y_dense, lambda_=self.lambda_val, n_rand=0,
                method="auto", # Auto should select GSL if available, else Python
                verbose=0
            )
            self._assert_result_dict(result_dict, "API_ttest_auto")
            expected_backend = 'gsl_cython' if GSL_CYTHON_AVAILABLE else 'python'
            self.assertEqual(result_dict.get('method_used', ''), expected_backend, f"Method used for t-test (auto) was not '{expected_backend}'")
            self.assertIn('df', result_dict, "DF key missing from auto t-test result")
            self.assertFalse(np.isnan(result_dict['df']), "DF is NaN in auto t-test result")
            pvalue = result_dict['pvalue']
            self.assertFalse(np.isnan(pvalue).any(), "T-test P-values contain NaNs (auto)")
            self.assertTrue(np.all((pvalue >= 0) & (pvalue <= 1)), "T-test P-values out of [0, 1] range (auto)")
            logger.info(f"T-test mode (auto) successful via ridge API using method '{result_dict.get('method_used', 'N/A')}'.")
        except Exception as e:
            self.fail(f"T-test mode (auto) failed unexpectedly: {e}")


    def test_80_with_pandas_inputs_api(self):
        """Test main API with pandas DataFrame inputs."""
        logger.info("Testing ridge API with pandas DataFrame inputs")
        gene_names = [f"G_{i}" for i in range(self.n_genes)]
        sample_names = [f"S_{i}" for i in range(self.n_samples)]
        feature_names = [f"F_{i}" for i in range(self.n_features)] # Use consistent naming

        X_df = pd.DataFrame(self.X_dense, index=gene_names, columns=feature_names)
        Y_df = pd.DataFrame(self.Y_dense, index=gene_names, columns=sample_names)

        try:
            result_dict = ridge(
                X_df, Y_df, # Pass DataFrames directly
                lambda_=self.lambda_val,
                n_rand=10, # Use small n_rand for speed
                method="auto", # Use auto to ensure it works with any available backend
                verbose=0
            )
            self._assert_result_dict(result_dict, "API_pandas")
            logger.info(f"DataFrame inputs test successful via ridge API using '{result_dict.get('method_used', 'N/A')}'.")
        except Exception as e:
            self.fail(f"DataFrame inputs test failed via ridge API: {e}")

    def test_81_invalid_inputs(self):
        """Test ridge API with invalid inputs."""
        logger.info("Testing ridge API with invalid inputs")
        # Shape mismatch
        with self.assertRaises(ValueError, msg="Should fail on X/Y shape mismatch"):
            ridge(self.X_dense[:50,:], self.Y_dense, lambda_=1)
        # Negative n_rand (should clamp to 0 and run t-test or raise if incompatible)
        # The main ridge function might clamp n_rand >= 0 now, let's test that behaviour.
        # If it clamps, it might raise NotImplementedError if the selected backend doesn't do t-tests.
        logger.info("Testing ridge API with negative n_rand (expect clamp or error)")
        try:
            ridge(self.X_dense, self.Y_dense, lambda_=1, n_rand=-10, method="auto")
            # If it gets here, it clamped n_rand=0 and ran t-test. Check if that's the case
            # This depends on t-test availability (GSL/Python)
        except ValueError as ve:
             # Allow ValueError if underlying backend raises it for negative n_rand
             # if it wasn't caught/clamped earlier.
             self.assertTrue("n_rand" in str(ve).lower(), "ValueError for n_rand<0 missing relevant message.")
             logger.info("Caught expected ValueError for negative n_rand.")
        except NotImplementedError as nie:
             # This could happen if n_rand is clamped to 0, but the selected backend (e.g., MKL)
             # doesn't support t-tests.
             self.assertTrue("t-test" in str(nie).lower(), "NotImplementedError for t-test missing relevant message.")
             logger.info("Caught expected NotImplementedError for t-test on backend that doesn't support it.")
        except Exception as e:
             self.fail(f"Unexpected error type {type(e).__name__} for negative n_rand: {e}")


        # Invalid alternative (only checked if n_rand=0)
        with self.assertRaises(ValueError, msg="Should fail on invalid alternative"):
             ridge(self.X_dense, self.Y_dense, lambda_=1, n_rand=0, alternative="sideways")

        # Sparse X with t-test
        with self.assertRaises(NotImplementedError, msg="Should fail on sparse X with t-test"):
            ridge(self.X_sparse, self.Y_sparse, lambda_=1, n_rand=0)

        # Sparse X with non-python backend request (should override to python)
        # Test each potentially available non-python backend
        for method in ["mkl", "gsl_cython", "numba", "gpu"]:
            if is_backend_available(method): # Use function from backend_selection
                 logger.info(f"Testing invalid input: Sparse X with method='{method}' (expect override to python)")
                 result = ridge(self.X_sparse, self.Y_sparse, lambda_=1, n_rand=10, method=method, verbose=0)
                 self.assertIn("python_sparse_iterative", result.get('method_used',''),
                               f"{method.upper()} request with sparse X did not report using python_sparse_iterative.")


if __name__ == '__main__':
    # Configure test runner
    runner = unittest.TextTestRunner(verbosity=2) # Increase verbosity of test runner output
    unittest.main(testRunner=runner)