=====
Ridge
=====

.. module:: ridge_inference.ridge

The ``ridge`` module provides the main interface for ridge regression with significance testing.

This module implements ridge regression with both permutation-based and t-test-based significance testing.
It supports multiple backend implementations (NumPy, Numba, CuPy/GPU, C) for optimal performance
across different hardware configurations.

Main Function
------------

.. autofunction:: ridge

Function Details
---------------

The ``ridge`` function is the primary entry point for ridge regression:

- It accepts dense or sparse input matrices
- It automatically selects the most appropriate backend based on inputs and available hardware
- It provides fallback mechanisms if the primary backend fails
- It supports both permutation testing (n_rand > 0) and t-test (n_rand = 0)

Backend Selection
----------------

The ``method`` parameter determines which backend is used:

- ``'auto'``: Automatically select the best backend based on inputs and available hardware
- ``'c'``: Use the C/OpenMP implementation for maximum CPU parallelization
- ``'python'``: Use the pure Python/NumPy implementation
- ``'numba'``: Use the Numba-accelerated implementation
- ``'gpu'``: Use the CuPy/GPU-accelerated implementation

When ``method='auto'``, the backend is selected as follows:

1. If Y is sparse, use 'gpu' if available, otherwise 'python'
2. If Y is dense, try in order: 'gpu', 'c', 'numba', 'python'

Return Value
-----------

The ``ridge`` function returns a dictionary with the following keys:

- ``'beta'``: Coefficient matrix (shape: n_features × n_samples)
- ``'se'``: Standard error matrix (shape: n_features × n_samples)
- ``'zscore'``: Z-score matrix (shape: n_features × n_samples)
- ``'pvalue'``: P-value matrix (shape: n_features × n_samples)
- ``'execution_time'``: Total execution time in seconds
- ``'method_used'``: The backend method that was actually used

Examples
--------

Basic usage::

    from ridge_inference import ridge
    import numpy as np
    
    # Create sample data
    X = np.random.randn(1000, 10)  # 1000 observations, 10 features
    Y = np.random.randn(1000, 5)   # 5 samples
    
    # Run ridge regression with permutation testing
    result = ridge(X, Y, lambda_=1000, n_rand=100)
    
    # Access results
    beta = result['beta']        # Coefficient matrix
    pvalue = result['pvalue']    # P-values

Using t-test instead of permutation testing::

    # Run ridge regression with t-test
    result = ridge(X, Y, lambda_=1000, n_rand=0)

Specifying a particular backend::

    # Run ridge regression with GPU backend
    result = ridge(X, Y, lambda_=1000, n_rand=100, method='gpu')

See Also
--------
:func:`ridge_inference.batch.ridge_batch`: Batch processing version for large datasets
