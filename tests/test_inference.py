#!/usr/bin/env python
"""
Test suite for the SecAct inference functionality.
"""

import unittest
import numpy as np
import pandas as pd
import os
import sys
import logging
import time
import tempfile
import shutil # For removing temp dir robustly
import warnings
from scipy import sparse as sps

# Ensure parent directory is in path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import necessary components from the package
try:
    from ridge_inference import secact_inference, secact_activity_inference
    from ridge_inference.loaders import load_signature_matrix
    # Import utilities needed for testing
    from ridge_inference.utils import convert_to_sparse # Keep this for direct testing
    from ridge_inference.inference import _is_Y_sparse_beneficial_for_cpu # Test internal helper

    # Check if batching is available (for specific tests)
    try:
        from ridge_inference.batch import ridge_batch
        BATCH_AVAILABLE = True
    except ImportError:
        BATCH_AVAILABLE = False
        # Define dummy to prevent NameError if tests are run without batch module
        def ridge_batch(*args, **kwargs):
            raise ImportError("Batch module not found")

    # Import availability flags from backend_selection
    from ridge_inference.backend_selection import (
        MKL_AVAILABLE, GSL_CYTHON_AVAILABLE, NUMBA_AVAILABLE, CUPY_AVAILABLE, PYTHON_AVAILABLE
    )

except ImportError as e:
    print(f"FATAL: Failed to import ridge_inference components: {e}")
    print(f"Current sys.path: {sys.path}")
    # Exit if core components can't be imported, tests cannot run
    sys.exit(f"Required ridge_inference components not found. Installation might be incomplete or sys.path incorrect.")

# Set up logging for tests
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)-8s - %(name)s - %(message)s')
logger = logging.getLogger("TestInference")


class TestSecActInference(unittest.TestCase):
    """Test class for SecAct inference functionality."""

    @classmethod
    def setUpClass(cls):
        """Set up test data once for all tests."""
        np.random.seed(42)
        cls.n_genes = 100
        cls.n_samples = 10
        cls.n_features = 5 # Renamed from n_proteins

        # Dense Data
        cls.X_dense = np.random.randn(cls.n_genes, cls.n_features).astype(np.float64) * 1.5
        cls.Y_dense = np.random.randn(cls.n_genes, cls.n_samples).astype(np.float64) * 1.0

        cls.gene_names = [f"GENE_{i}" for i in range(cls.n_genes)]
        cls.sample_names = [f"SAMPLE_{i}" for i in range(cls.n_samples)]
        cls.feature_names = [f"FEAT_{i}" for i in range(cls.n_features)] # Renamed from protein_names

        cls.expr_df = pd.DataFrame(cls.Y_dense, index=cls.gene_names, columns=cls.sample_names)
        cls.sig_df = pd.DataFrame(cls.X_dense, index=cls.gene_names, columns=cls.feature_names)

        # Sparse Y Data (used for sparse input tests)
        cls.Y_sparse = sps.random(cls.n_genes, cls.n_samples, density=0.1, format='csr', random_state=42, dtype=np.float64)
        cls.Y_sparse.data = np.random.randn(len(cls.Y_sparse.data)) * 1.5 # Assign realistic values
        cls.expr_sparse_df = pd.DataFrame.sparse.from_spmatrix(cls.Y_sparse, index=cls.gene_names, columns=cls.sample_names)


        # Temp file for signature loading test
        cls.temp_dir = tempfile.mkdtemp()
        cls.temp_sig_path = os.path.join(cls.temp_dir, "test_signature.tsv")
        cls.sig_df.to_csv(cls.temp_sig_path, sep="\t")

        cls.lambda_val = 1000.0
        cls.n_rand = 10 # Reduced n_rand for faster testing

        logger.info("-" * 40)
        logger.info("Test setup complete:")
        logger.info(f"Shapes: Expression={cls.expr_df.shape}, Signature={cls.sig_df.shape}")
        logger.info(f"Temp signature file: {cls.temp_sig_path}")
        logger.info(f"Batch processing available: {BATCH_AVAILABLE}")
        # Use imported availability flags
        logger.info(f"Backend Availability: MKL={MKL_AVAILABLE}, GSL={GSL_CYTHON_AVAILABLE}, Numba={NUMBA_AVAILABLE}, GPU={CUPY_AVAILABLE}, Python={PYTHON_AVAILABLE}")
        logger.info("-" * 40)

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary directory and files."""
        try:
            if os.path.exists(cls.temp_dir):
                shutil.rmtree(cls.temp_dir)
                logger.info(f"Removed temp directory: {cls.temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to remove temp directory {cls.temp_dir}: {e}")

    def test_01_load_signature_matrix(self):
        """Test loading signature matrix from file and DataFrame."""
        logger.info("Testing load_signature_matrix...")
        # Test loading from DataFrame
        df_loaded = load_signature_matrix(self.sig_df)
        pd.testing.assert_frame_equal(df_loaded, self.sig_df)

        # Test loading from file path
        file_loaded = load_signature_matrix(self.temp_sig_path)
        # Compare loaded DataFrame from file to the original DataFrame
        pd.testing.assert_frame_equal(file_loaded, self.sig_df, check_exact=False, atol=1e-6) # Allow minor float diffs

        # Test loading built-in (if they exist, otherwise skip)
        try:
            secact_loaded = load_signature_matrix("SecAct")
            self.assertIsInstance(secact_loaded, pd.DataFrame)
            self.assertFalse(secact_loaded.empty)
            logger.info("Loaded built-in 'SecAct' signature.")
        except FileNotFoundError:
            logger.warning("Skipping test for built-in 'SecAct' matrix (file not found).")
        except Exception as e:
             self.fail(f"Failed to load built-in 'SecAct' matrix: {e}")

        logger.info("load_signature_matrix tests passed.")

    def _assert_result_structure(self, result, expected_feature_names, expected_sample_names):
        """Helper to assert common structure of inference results."""
        self.assertIsInstance(result, dict, "Result should be a dictionary")
        expected_keys = ['beta', 'se', 'zscore', 'pvalue', 'execution_time', 'method']
        # Optional batch keys
        if 'batched' in result: expected_keys.extend(['batched', 'batch_size'])
        # Optional GPU keys
        if 'peak_gpu_pool_mb' in result: expected_keys.append('peak_gpu_pool_mb')

        for key in expected_keys:
            self.assertIn(key, result, f"Result should contain '{key}'")

        expected_shape = (len(expected_feature_names), len(expected_sample_names))

        for key in ['beta', 'se', 'zscore', 'pvalue']:
             self.assertIn(key, result, f"Result should contain '{key}'")
             self.assertIsInstance(result[key], pd.DataFrame, f"{key} should be a DataFrame")
             self.assertEqual(result[key].shape, expected_shape, f"{key} shape incorrect - expected {expected_shape}, got {result[key].shape}")
             # Check index and columns match expected names
             self.assertListEqual(list(result[key].index), expected_feature_names, f"{key} index incorrect")
             self.assertListEqual(list(result[key].columns), expected_sample_names, f"{key} columns incorrect")
             # Check for excessive NaNs
             nan_fraction = result[key].isnull().sum().sum() / result[key].size
             self.assertLess(nan_fraction, 0.5, f"{key} has too many NaNs ({nan_fraction:.1%})")

    def test_10_secact_inference_basic(self):
        """Test basic secact_inference functionality with dense inputs."""
        logger.info("Testing basic secact_inference (dense inputs)...")
        result = secact_inference(
            self.expr_df, sig_matrix=self.sig_df,
            lambda_val=self.lambda_val, n_rand=self.n_rand,
            method="python", verbose=0 # Use reliable backend, low verbosity
        )
        self._assert_result_structure(result, self.feature_names, self.sample_names)
        self.assertEqual(result['method'], 'python')
        logger.info("Basic secact_inference test passed.")

    def test_11_secact_inference_with_file_path(self):
        """Test secact_inference loading signature matrix from file path."""
        logger.info("Testing secact_inference with signature file path...")
        result = secact_inference(
            self.expr_df, sig_matrix=self.temp_sig_path,
            lambda_val=self.lambda_val, n_rand=self.n_rand,
            method="python", verbose=0
        )
        self._assert_result_structure(result, self.feature_names, self.sample_names)
        logger.info("secact_inference with file path test passed.")

    def test_12_secact_inference_add_background(self):
        """Test secact_inference with add_background=True."""
        logger.info("Testing secact_inference with add_background=True...")
        result = secact_inference(
            self.expr_df, sig_matrix=self.sig_df,
            lambda_val=self.lambda_val, n_rand=self.n_rand,
            method="python", add_background=True, verbose=0
        )
        expected_features = self.feature_names + ['background']
        self._assert_result_structure(result, expected_features, self.sample_names)
        self.assertIn('background', result['beta'].index)
        logger.info("secact_inference add_background test passed.")

    def test_13_secact_inference_scaling(self):
        """Test secact_inference with scaling options."""
        logger.info("Testing secact_inference with scaling...")
        for scale_method in ['column', 'global']:
            with self.subTest(scale_method=scale_method):
                logger.info(f"Testing scale_method='{scale_method}'")
                result = secact_inference(
                    self.expr_df, sig_matrix=self.sig_df,
                    lambda_val=self.lambda_val, n_rand=5, # Faster for scaling test
                    method="python", scale_method=scale_method, verbose=0
                )
                self._assert_result_structure(result, self.feature_names, self.sample_names)
        logger.info("secact_inference scaling tests passed (execution check).")

    def test_15_secact_inference_methods(self):
        """Test secact_inference with different available methods (dense Y)."""
        logger.info("Testing secact_inference with different methods (Dense Y)...")
        methods_to_test = ["python", "auto"]
        if MKL_AVAILABLE: methods_to_test.append("mkl")
        if GSL_CYTHON_AVAILABLE: methods_to_test.append("gsl_cython")
        if NUMBA_AVAILABLE: methods_to_test.append("numba") # Numba handles dense Y now
        if CUPY_AVAILABLE: methods_to_test.append("gpu")

        # Reduce n_rand further for faster multi-method testing
        n_rand_fast = 5

        for method in methods_to_test:
             with self.subTest(method=method):
                logger.info(f"Testing method: {method}")
                try:
                    result = secact_inference(
                        self.expr_df, sig_matrix=self.sig_df,
                        lambda_val=self.lambda_val, n_rand=n_rand_fast,
                        method=method, verbose=0,
                        batch_threshold=1000 # Ensure non-batching for this test
                    )
                    self._assert_result_structure(result, self.feature_names, self.sample_names)
                    logger.info(f"Method '{method}' test passed. Reported method: {result['method']}")
                except (ImportError, NotImplementedError) as skip_err:
                     logger.warning(f"Method '{method}' skipped: {skip_err}")
                     self.skipTest(f"Backend {method} skipped: {skip_err}")
                except Exception as e:
                     self.fail(f"Method '{method}' failed unexpectedly: {e}")

    def test_16_secact_inference_sparse_Y(self):
        """Test secact_inference with sparse Y input and compatible methods."""
        logger.info("Testing secact_inference with sparse Y input...")
        # Methods expected to handle sparse Y directly or via internal conversion
        methods_to_test = ["python", "auto"]
        if MKL_AVAILABLE: methods_to_test.append("mkl")
        if NUMBA_AVAILABLE: methods_to_test.append("numba") # Numba now handles sparse Y
        if CUPY_AVAILABLE: methods_to_test.append("gpu")
        # GSL would require densification, handled by ridge() but test direct call here

        n_rand_fast = 5
        for method in methods_to_test:
             with self.subTest(method=method):
                logger.info(f"Testing method: {method} with sparse Y")
                try:
                    result = secact_inference(
                        self.expr_sparse_df, # Pass sparse DataFrame
                        sig_matrix=self.sig_df,
                        lambda_val=self.lambda_val, n_rand=n_rand_fast,
                        method=method, verbose=0,
                        batch_threshold=1000
                    )
                    self._assert_result_structure(result, self.feature_names, self.sample_names)
                    logger.info(f"Method '{method}' (Sparse Y) test passed. Reported method: {result['method']}")
                except (ImportError, NotImplementedError) as skip_err:
                     logger.warning(f"Method '{method}' (Sparse Y) skipped: {skip_err}")
                     self.skipTest(f"Backend {method} skipped: {skip_err}")
                except Exception as e:
                     self.fail(f"Method '{method}' (Sparse Y) failed unexpectedly: {e}")


    @unittest.skipIf(not BATCH_AVAILABLE, "Batch processing function not available")
    def test_20_secact_inference_batching(self):
        """Test secact_inference with batching enabled."""
        logger.info("Testing secact_inference with batching...")
        small_batch_size = 3
        # Threshold lower than n_samples to trigger batching
        small_batch_threshold = self.n_samples - 1

        # Test with a backend known to support batching (python or gpu)
        batch_method = "python"
        if CUPY_AVAILABLE: batch_method = "gpu" # Prefer GPU if available

        logger.info(f"Using method '{batch_method}' for batching test.")
        result = secact_inference(
            self.expr_df, sig_matrix=self.sig_df,
            lambda_val=self.lambda_val, n_rand=self.n_rand,
            method=batch_method,
            batch_size=small_batch_size,
            batch_threshold=small_batch_threshold,
            verbose=0
        )

        self._assert_result_structure(result, self.feature_names, self.sample_names)
        self.assertTrue(result.get('batched', False), "Result should indicate batching was used")
        self.assertEqual(result.get('batch_size'), small_batch_size, "Result should contain correct batch size")
        self.assertIn("_batched", result.get('method', ''), "Method name should reflect batching")
        logger.info("Batch processing test passed.")


    def test_30_secact_activity_inference_differential(self):
        """Test secact_activity_inference with is_differential=True."""
        logger.info("Testing activity inference wrapper (is_differential=True)...")
        # Prepare a mock differential profile (e.g., logFC)
        diff_expr = self.expr_df - self.expr_df.mean(axis=1).mean() # Simple mock diff
        result = secact_activity_inference(
            diff_expr, is_differential=True, sig_matrix=self.sig_df,
            lambda_val=self.lambda_val, n_rand=self.n_rand, method="python", verbose=0
        )
        self._assert_result_structure(result, self.feature_names, self.sample_names)
        logger.info("Activity inference (is_differential=True) test passed.")

    def test_31_secact_activity_inference_control(self):
        """Test secact_activity_inference comparing to control."""
        logger.info("Testing activity inference wrapper (compare to control)...")
        control_df = pd.DataFrame(
            np.random.randn(self.n_genes, self.n_samples) * 0.8, # Slightly different control data
            index=self.gene_names, columns=self.sample_names
        )

        # Test unpaired, averaged
        logger.info("... comparing to control (unpaired, averaged)")
        result_unpaired_avg = secact_activity_inference(
            self.expr_df, input_profile_control=control_df,
            is_paired=False, is_single_sample_level=False, sig_matrix=self.sig_df,
            lambda_val=self.lambda_val, n_rand=self.n_rand, method="python", verbose=0
        )
        self._assert_result_structure(result_unpaired_avg, self.feature_names, ["Change"])

        # Test paired, single sample level
        logger.info("... comparing to control (paired, single sample)")
        result_paired_single = secact_activity_inference(
            self.expr_df, input_profile_control=control_df,
            is_paired=True, is_single_sample_level=True, sig_matrix=self.sig_df,
            lambda_val=self.lambda_val, n_rand=self.n_rand, method="python", verbose=0
        )
        self._assert_result_structure(result_paired_single, self.feature_names, self.sample_names)
        logger.info("Activity inference (compare to control) tests passed.")

    def test_40_common_genes_handling(self):
        """Test handling of mismatched gene sets between expression and signature."""
        logger.info("Testing common genes handling...")
        common_genes_subset = self.gene_names[:80] # Use 80 common genes
        extra_expr_genes = [f"EXTRA_EXPR_{i}" for i in range(20)]
        extra_sig_genes = [f"EXTRA_SIG_{i}" for i in range(10)]

        expr_mod_index = common_genes_subset + extra_expr_genes
        sig_mod_index = common_genes_subset + extra_sig_genes

        expr_mod_df = pd.DataFrame(
            np.random.randn(100, self.n_samples),
            index=expr_mod_index, columns=self.sample_names
        )
        sig_mod_df = pd.DataFrame(
            np.random.randn(90, self.n_features), # 80 common + 10 extra
            index=sig_mod_index, columns=self.feature_names
        )

        # Expect successful run using only the 80 common genes
        result = secact_inference(
            expr_mod_df, sig_matrix=sig_mod_df,
            lambda_val=self.lambda_val, n_rand=10, method="python", verbose=0
        )
        # Result structure should still be based on original feature/sample names
        self._assert_result_structure(result, self.feature_names, self.sample_names)
        logger.info("Common genes handling test passed.")

    def test_50_utility_sparse_check(self):
        """Test the _is_Y_sparse_beneficial_for_cpu utility."""
        logger.info("Testing _is_Y_sparse_beneficial_for_cpu utility...")
        # Dense, small -> False
        dense_small = np.random.rand(100, 10)
        self.assertFalse(_is_Y_sparse_beneficial_for_cpu(dense_small, min_elements=2000))
        # Dense, large -> False
        dense_large = np.random.rand(1000, 1000)
        self.assertFalse(_is_Y_sparse_beneficial_for_cpu(dense_large, threshold=0.5, min_elements=1e5))
        # Sparse, large -> True
        sparse_large = sps.random(1000, 1000, density=0.05, format='csr')
        self.assertTrue(_is_Y_sparse_beneficial_for_cpu(sparse_large, threshold=0.1, min_elements=1e5))
        # Dense but sparse content, large -> True
        dense_sparse_content = np.zeros((1000, 1000))
        dense_sparse_content[0,0] = 1
        self.assertTrue(_is_Y_sparse_beneficial_for_cpu(dense_sparse_content, threshold=0.1, min_elements=1e5))
        logger.info("_is_Y_sparse_beneficial_for_cpu tests passed.")


if __name__ == '__main__':
    # Configure test runner
    runner = unittest.TextTestRunner(verbosity=2) # Increase verbosity of test runner output
    unittest.main(testRunner=runner)
# --- END OF FILE test_inference.py ---