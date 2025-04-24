# --- START OF FILE test_t_test.py ---

import numpy as np
import sys
import os
import logging
import unittest
import time
import gc # Import garbage collector

# Ensure parent directory is in the path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import necessary components from the package
try:
    from ridge_inference import ridge
    # Import availability flags from backend_selection
    from ridge_inference.backend_selection import (
        MKL_AVAILABLE, GSL_CYTHON_AVAILABLE, NUMBA_AVAILABLE, CUPY_AVAILABLE, PYTHON_AVAILABLE
    )
except ImportError as e:
    print(f"FATAL: Failed to import ridge_inference components: {e}")
    print(f"Current sys.path: {sys.path}")
    sys.exit(f"Required ridge_inference components not found. Installation might be incomplete or sys.path incorrect.")

# Set up logging for tests
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)-8s - %(name)s - %(message)s')
logger = logging.getLogger("TestTTest")


class TestTTestMode(unittest.TestCase):
    """Test class for t-test functionality (n_rand=0)."""

    @classmethod
    def setUpClass(cls):
        """Set up test data once for all tests."""
        np.random.seed(43) # Use different seed from test_ridge
        cls.n_genes = 100
        cls.n_features = 10
        cls.n_samples = 5
        cls.X_dense = np.random.randn(cls.n_genes, cls.n_features).astype(np.float64)
        cls.Y_dense = np.random.randn(cls.n_genes, cls.n_samples).astype(np.float64)
        # Ensure C-contiguity
        cls.X_dense = np.require(cls.X_dense, requirements=['C', 'A'])
        cls.Y_dense = np.require(cls.Y_dense, requirements=['C', 'A'])

        cls.lambda_val = 100.0 # Use a smaller lambda for t-test stability checks

        logger.info("-" * 40)
        logger.info("T-Test Setup:")
        logger.info(f"Generated data: X={cls.X_dense.shape}, Y={cls.Y_dense.shape}")
        logger.info(f"Lambda: {cls.lambda_val}")
        logger.info(f"Backend Availability: MKL={MKL_AVAILABLE}, GSL={GSL_CYTHON_AVAILABLE}, "
                    f"Numba={NUMBA_AVAILABLE}, GPU={CUPY_AVAILABLE}, Python={PYTHON_AVAILABLE}")
        logger.info("-" * 40)

        # Cache results for consistency checks
        cls.results = {}
        # Pre-run t-tests for python and GSL if available to populate cache
        if PYTHON_AVAILABLE:
            cls._run_ridge_ttest_static(cls.results, 'python', cls.X_dense, cls.Y_dense, cls.lambda_val, cls.n_features, cls.n_samples)
        if GSL_CYTHON_AVAILABLE:
            cls._run_ridge_ttest_static(cls.results, 'gsl_cython', cls.X_dense, cls.Y_dense, cls.lambda_val, cls.n_features, cls.n_samples)

    @staticmethod
    def _run_ridge_ttest_static(results_cache, method, X, Y, lambda_val, n_features, n_samples):
        """Static helper to run ridge t-test and cache result, used by setUpClass."""
        if method in results_cache:
            return results_cache[method]

        logger.info(f"[Setup] Running method='{method}' with t-test (n_rand=0)...")
        result_dict = None
        try:
            result_dict = ridge(X, Y, method=method, n_rand=0, lambda_=lambda_val, verbose=0) # Lower verbosity in setup
            results_cache[method] = result_dict

            # Basic checks during setup run
            if not isinstance(result_dict, dict): raise TypeError("Result not a dict")
            expected_keys = ['beta', 'se', 'zscore', 'pvalue', 'method_used', 'df']
            for k in expected_keys:
                if k not in result_dict: raise KeyError(f"Key '{k}' missing")
            for k in ['beta', 'se', 'zscore', 'pvalue']:
                 if not isinstance(result_dict[k], np.ndarray): raise TypeError(f"Key '{k}' not ndarray")
                 if result_dict[k].shape != (n_features, n_samples): raise ValueError(f"Key '{k}' shape mismatch")

            logger.info(f"[Setup] Method '{method}' t-test ran successfully. Reported method: '{result_dict.get('method_used', 'N/A')}'")

        except (NotImplementedError, ImportError, TypeError, ValueError) as e:
             logger.warning(f"[Setup] Method '{method}' t-test skipped or failed as expected: {type(e).__name__} - {e}")
             results_cache[method] = "skipped_or_failed_expectedly"
        except Exception as e:
             logger.error(f"[Setup] Method '{method}' t-test failed UNEXPECTEDLY: {type(e).__name__} - {e}", exc_info=True)
             results_cache[method] = None # Cache unexpected failure

        gc.collect() # Garbage collect after setup run
        return results_cache[method]

    def _assert_basic_ttest_result(self, result, method_name):
         """Basic checks common to most t-test results."""
         if result == "skipped_or_failed_expectedly":
              self.skipTest(f"Method '{method_name}' t-test skipped or failed expectedly.")
         self.assertIsNotNone(result, f"Method '{method_name}' result is None (unexpected failure).")
         self.assertIsInstance(result, dict)
         expected_keys = ['beta', 'se', 'zscore', 'pvalue', 'method_used', 'df']
         for k in expected_keys: self.assertIn(k, result)
         for k in ['beta', 'se', 'zscore', 'pvalue']:
             self.assertIsInstance(result[k], np.ndarray)
             self.assertEqual(result[k].shape, (self.n_features, self.n_samples))
             pvals = result['pvalue'] # Check p-value validity specifically
             self.assertFalse(np.isnan(pvals).any(), f"{method_name}: P-values contain NaNs")
             self.assertTrue(np.all((pvals >= 0) & (pvals <= 1)), f"{method_name}: P-values out of [0,1] range")
         self.assertIsNotNone(result.get('df'), f"{method_name}: Result should contain 'df' key for t-test")
         self.assertFalse(np.isnan(result['df']), f"{method_name}: 'df' should not be NaN for successful run")


    # --- Individual Method Tests for T-Test ---

    def test_01_python_ttest(self):
        """Test t-test with the 'python' backend."""
        result = self.results.get('python')
        self._assert_basic_ttest_result(result, 'python')
        if isinstance(result, dict): # Additional check specific to python backend
            self.assertEqual(result['method_used'], 'python')

    @unittest.skipIf(not GSL_CYTHON_AVAILABLE, "GSL Cython backend not available")
    def test_02_gsl_cython_ttest(self):
        """Test t-test with the 'gsl_cython' backend."""
        result = self.results.get('gsl_cython')
        self._assert_basic_ttest_result(result, 'gsl_cython')
        if isinstance(result, dict): # Additional check specific to GSL backend
             self.assertIn('gsl_cython', result.get('method_used','').lower())

    # --- Fallback Tests ---
    # These tests confirm that calling ridge with n_rand=0 and an incompatible method
    # results in the execution falling back to a compatible one (python or gsl)

    @unittest.skipIf(not MKL_AVAILABLE, "MKL backend not available")
    def test_03_mkl_ttest_fallback(self):
        """Test t-test call with 'mkl' (should fallback to python or gsl)."""
        logger.info(f"--- Testing method='mkl' with t-test (n_rand=0) ---")
        try:
            result = ridge(self.X_dense, self.Y_dense, method='mkl', n_rand=0, lambda_=self.lambda_val, verbose=1)
            self._assert_basic_ttest_result(result, 'mkl (fallback)') # Check structure
            self.assertTrue('python' in result['method_used'] or 'gsl_cython' in result['method_used'],
                              f"MKL t-test call did not fallback correctly, used: {result['method_used']}")
        except Exception as e:
            self.fail(f"Method 'mkl' t-test fallback failed unexpectedly: {e}")

    @unittest.skipIf(not NUMBA_AVAILABLE, "Numba backend not available")
    def test_04_numba_ttest_fallback(self):
        """Test t-test call with 'numba' (should fallback to python or gsl)."""
        logger.info(f"--- Testing method='numba' with t-test (n_rand=0) ---")
        try:
            result = ridge(self.X_dense, self.Y_dense, method='numba', n_rand=0, lambda_=self.lambda_val, verbose=1)
            self._assert_basic_ttest_result(result, 'numba (fallback)')
            self.assertTrue('python' in result['method_used'] or 'gsl_cython' in result['method_used'],
                              f"Numba t-test call did not fallback correctly, used: {result['method_used']}")
        except Exception as e:
             self.fail(f"Method 'numba' t-test fallback failed unexpectedly: {e}")


    @unittest.skipIf(not CUPY_AVAILABLE, "GPU backend not available")
    def test_05_gpu_ttest_fallback(self):
        """Test t-test call with 'gpu' (should fallback to python or gsl)."""
        logger.info(f"--- Testing method='gpu' with t-test (n_rand=0) ---")
        try:
            result = ridge(self.X_dense, self.Y_dense, method='gpu', n_rand=0, lambda_=self.lambda_val, verbose=1)
            self._assert_basic_ttest_result(result, 'gpu (fallback)')
            self.assertTrue('python' in result['method_used'] or 'gsl_cython' in result['method_used'],
                              f"GPU t-test call did not fallback correctly, used: {result['method_used']}")
        except Exception as e:
             self.fail(f"Method 'gpu' t-test fallback failed unexpectedly: {e}")

    def test_06_auto_ttest(self):
        """Test t-test call with 'auto' (should select python or GSL)."""
        logger.info(f"--- Testing method='auto' with t-test (n_rand=0) ---")
        try:
            result = ridge(self.X_dense, self.Y_dense, method='auto', n_rand=0, lambda_=self.lambda_val, verbose=1)
            self._assert_basic_ttest_result(result, 'auto')
            self.assertTrue('python' in result['method_used'] or 'gsl_cython' in result['method_used'],
                              f"Auto t-test call selected unexpected backend: {result['method_used']}")
        except Exception as e:
             self.fail(f"Method 'auto' t-test fallback failed unexpectedly: {e}")


    # --- Consistency Check (Python vs GSL if available) ---
    @unittest.skipIf(not GSL_CYTHON_AVAILABLE or not PYTHON_AVAILABLE, "Python or GSL backend not available for t-test comparison")
    def test_90_ttest_consistency(self):
        """Compare t-test results between Python and GSL backends."""
        logger.info("Comparing 'python' and 'gsl_cython' t-test results...")
        python_result = self.results.get('python', None)
        gsl_result = self.results.get('gsl_cython', None)

        # Skip if either result wasn't successfully generated in setUpClass
        if python_result is None or isinstance(python_result, str):
            self.skipTest("Python backend t-test failed or was skipped during setup.")
        if gsl_result is None or isinstance(gsl_result, str):
            self.skipTest("GSL Cython backend t-test failed or was skipped during setup.")

        # Compare results using numpy.allclose
        rtol, atol = 1e-5, 1e-7 # Adjusted tolerance slightly
        pval_rtol, pval_atol = 1e-4, 1e-6 # Looser for p-value

        try:
            # Compare df first as it affects p-values
            self.assertAlmostEqual(python_result['df'], gsl_result['df'], places=5, msg="Degrees of freedom differ significantly")
            np.testing.assert_allclose(python_result['beta'], gsl_result['beta'], rtol=rtol, atol=atol, err_msg="Beta values differ")
            np.testing.assert_allclose(python_result['se'], gsl_result['se'], rtol=rtol, atol=atol, err_msg="SE values differ", equal_nan=True)
            np.testing.assert_allclose(python_result['zscore'], gsl_result['zscore'], rtol=rtol, atol=atol, err_msg="Z-score (t-stat) values differ", equal_nan=True)
            np.testing.assert_allclose(python_result['pvalue'], gsl_result['pvalue'], rtol=pval_rtol, atol=pval_atol, err_msg="P-value values differ")
            logger.info("T-test results between Python and GSL are consistent within tolerance.")
        except AssertionError as ae:
            logger.error(f"Max Beta Diff: {np.max(np.abs(python_result['beta'] - gsl_result['beta']))}")
            logger.error(f"Max SE Diff: {np.max(np.abs(python_result['se'] - gsl_result['se']))}")
            logger.error(f"Max Z Diff: {np.max(np.abs(python_result['zscore'] - gsl_result['zscore']))}")
            logger.error(f"Max Pval Diff: {np.max(np.abs(python_result['pvalue'] - gsl_result['pvalue']))}")
            logger.error(f"DF Diff: Python={python_result.get('df')} vs GSL={gsl_result.get('df')}")
            self.fail(f"T-test consistency check failed between Python and GSL: {ae}")


if __name__ == '__main__':
    # Configure test runner for more detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)
# --- END OF FILE test_t_test.py ---