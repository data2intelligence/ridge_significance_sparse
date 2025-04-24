=====
Core
=====

.. module:: ridge_inference.core

The ``core`` module provides the fundamental computational functions for ridge regression.

This module contains the core implementations of ridge regression algorithms in various 
backends (NumPy, Numba, CuPy/GPU). These implementations are used by the higher-level
modules like ``ridge`` and ``batch``.

Backend Functions
---------------

NumPy Backend
~~~~~~~~~~~

The NumPy backend provides a pure Python implementation of ridge regression:

.. function:: _calculate_T_numpy(X, lambda_val)

   Calculates T = (X'X + lambda*I)^-1 * X' using NumPy.

   :param X: Input matrix of shape (n_genes, n_features)
   :param lambda_val: Ridge regularization parameter
   :return: T matrix of shape (n_features, n_genes)

.. function:: _calculate_beta_numpy(T, Y)

   Calculates beta = T @ Y using NumPy.

   :param T: T matrix from _calculate_T_numpy
   :param Y: Output matrix of shape (n_genes, n_samples)
   :return: Beta matrix of shape (n_features, n_samples)

.. function:: _perform_permutation_numpy(T, Y, beta, n_rand)

   Performs permutation test using NumPy.

   :param T: T matrix from _calculate_T_numpy
   :param Y: Output matrix of shape (n_genes, n_samples)
   :param beta: Beta matrix from _calculate_beta_numpy
   :param n_rand: Number of permutations
   :return: tuple of (se, zscore, pvalue) matrices

.. function:: _perform_ttest_numpy(X, Y, T, beta, lambda_val)

   Performs t-test using NumPy.

   :param X: Input matrix of shape (n_genes, n_features)
   :param Y: Output matrix of shape (n_genes, n_samples)
   :param T: T matrix from _calculate_T_numpy
   :param beta: Beta matrix from _calculate_beta_numpy
   :param lambda_val: Ridge regularization parameter
   :return: tuple of (se, zscore, pvalue) matrices

.. function:: ridge_regression_numpy(X, Y, lambda_val=5e5, n_rand=1000)

   Main NumPy backend function.

   :param X: Input matrix of shape (n_genes, n_features)
   :param Y: Output matrix of shape (n_genes, n_samples)
   :param lambda_val: Ridge regularization parameter
   :param n_rand: Number of permutations
   :return: dict with beta, se, zscore, pvalue matrices

Numba Backend
~~~~~~~~~~

The Numba backend provides a JIT-compiled implementation for improved performance:

.. function:: _calculate_T_numba(X, lambda_val)

   Calculates T = (X'X + lambda*I)^-1 * X' using Numba.

   :param X: Input matrix of shape (n_genes, n_features)
   :param lambda_val: Ridge regularization parameter
   :return: T matrix of shape (n_features, n_genes)

.. function:: _calculate_beta_numba(T, Y)

   Calculates beta = T @ Y using Numba.

   :param T: T matrix from _calculate_T_numba
   :param Y: Output matrix of shape (n_genes, n_samples)
   :return: Beta matrix of shape (n_features, n_samples)

.. function:: _perform_permutation_numba(T, Y, beta, n_rand)

   Performs permutation test using Numba.

   :param T: T matrix from _calculate_T_numba
   :param Y: Output matrix of shape (n_genes, n_samples)
   :param beta: Beta matrix from _calculate_beta_numba
   :param n_rand: Number of permutations
   :return: tuple of (se, zscore, pvalue) matrices

.. function:: ridge_regression_numba(X, Y, lambda_val=5e5, n_rand=1000)

   Main Numba backend function.

   :param X: Input matrix of shape (n_genes, n_features)
   :param Y: Output matrix of shape (n_genes, n_samples)
   :param lambda_val: Ridge regularization parameter
   :param n_rand: Number of permutations
   :return: dict with beta, se, zscore, pvalue matrices

CuPy/GPU Backend
~~~~~~~~~~~~~~

The CuPy/GPU backend provides GPU-accelerated implementation:

.. function:: _transfer_to_gpu(X, Y)

   Transfers X and Y matrices to GPU.

   :param X: Input matrix of shape (n_genes, n_features)
   :param Y: Output matrix of shape (n_genes, n_samples)
   :return: tuple of (X_gpu, Y_gpu) matrices on GPU

.. function:: _calculate_T_cupy(X_gpu, lambda_val)

   Calculates T = (X'X + lambda*I)^-1 * X' using CuPy.

   :param X_gpu: Input matrix on GPU
   :param lambda_val: Ridge regularization parameter
   :return: T matrix on GPU

.. function:: _calculate_beta_cupy(T_gpu, Y_gpu)

   Calculates beta = T @ Y using CuPy.

   :param T_gpu: T matrix on GPU
   :param Y_gpu: Output matrix on GPU
   :return: Beta matrix on GPU

.. function:: _perform_permutation_cupy(T_gpu, Y_gpu, beta_gpu, n_rand)

   Performs permutation test using CuPy.

   :param T_gpu: T matrix on GPU
   :param Y_gpu: Output matrix on GPU
   :param beta_gpu: Beta matrix on GPU
   :param n_rand: Number of permutations
   :return: tuple of (se, zscore, pvalue) matrices on GPU

.. function:: ridge_regression_cupy(X, Y, lambda_val=5e5, n_rand=1000)

   Main CuPy/GPU backend function.

   :param X: Input matrix of shape (n_genes, n_features)
   :param Y: Output matrix of shape (n_genes, n_samples)
   :param lambda_val: Ridge regularization parameter
   :param n_rand: Number of permutations
   :return: dict with beta, se, zscore, pvalue matrices and peak GPU memory usage

Implementation Details
--------------------

The module implements each step of the ridge regression algorithm:

1. **T Matrix Calculation**: (X'X + λI)^-1 * X'
   - Handles dense and sparse matrices
   - Uses Cholesky decomposition when possible
   - Provides fallback to direct inversion

2. **Beta Calculation**: T @ Y
   - Optimized for both dense and sparse Y

3. **Significance Testing**:
   - Permutation testing (n_rand > 0)
   - T-test (n_rand = 0, NumPy only)

4. **GPU Memory Management**:
   - Explicit cleanup after operations
   - Pool management for reduced fragmentation

The implementations carefully handle numerical stability, memory usage, and error conditions.

Backend Availability Flags
------------------------

The module provides availability flags for each backend:

- ``NUMBA_AVAILABLE``: Whether Numba is available
- ``CUPY_AVAILABLE``: Whether CuPy/GPU is available
- ``C_AVAILABLE``: Whether the C backend is available (via c_bindings)

These flags are used by higher-level modules for backend selection.

See Also
--------
:func:`ridge_inference.ridge.ridge`: High-level ridge function
:func:`ridge_inference.batch.ridge_batch`: Batch processing function
