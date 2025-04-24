=====
Batch
=====

.. module:: ridge_inference.batch

The ``batch`` module provides memory-efficient batch processing for ridge regression on large datasets.

This module implements a batched version of ridge regression that processes columns of the Y matrix
in smaller chunks to reduce memory consumption. This is particularly useful for large datasets
where the full computation would exceed available memory.

Main Function
------------

.. autofunction:: ridge_batch

Function Details
---------------

The ``ridge_batch`` function is designed for memory-efficient ridge regression:

- It calculates the T matrix (``(X'X + λI)^-1 X'``) once and reuses it across batches
- It processes columns of Y in smaller batches to reduce memory usage
- It supports all backend implementations (Python, Numba, GPU, C)
- It automatically handles GPU memory management for the GPU backend
- It provides progress reporting with the ``verbose`` parameter

The function is particularly beneficial when:

- The number of samples (columns in Y) is large
- The number of features (columns in X) is moderate
- The permutation count (n_rand) is high

Backend Selection
----------------

The ``method`` parameter works similarly to the main ``ridge`` function:

- ``'auto'``: Automatically select the best backend
- ``'c'``: Use the C/OpenMP implementation
- ``'python'``: Use the pure Python/NumPy implementation
- ``'numba'``: Use the Numba-accelerated implementation
- ``'gpu'``: Use the CuPy/GPU-accelerated implementation

For ``method='auto'``, the selection logic prioritizes:

1. GPU backend for sparse Y or large sample count (>1000)
2. C backend for dense matrices
3. Numba backend for dense matrices
4. Python backend as fallback

Return Value
-----------

The ``ridge_batch`` function returns a tuple:

- ``beta``: Coefficient matrix (numpy.ndarray)
- ``se``: Standard error matrix (numpy.ndarray)
- ``zscore``: Z-score matrix (numpy.ndarray)
- ``pvalue``: P-value matrix (numpy.ndarray)

All matrices have shape (n_features × n_samples).

Examples
--------

Basic usage::

    from ridge_inference.batch import ridge_batch
    import numpy as np
    
    # Create sample data with many samples
    X = np.random.randn(1000, 20)       # 1000 observations, 20 features
    Y = np.random.randn(1000, 10000)    # 10,000 samples
    
    # Run ridge regression with batch processing
    beta, se, zscore, pvalue = ridge_batch(
        X, Y, 
        lambda_val=1000, 
        n_rand=100,
        batch_size=500,
        verbose=1
    )

Using with sparse matrices::

    from scipy import sparse
    
    # Create sparse Y matrix (90% zeros)
    Y_data = np.random.randn(1000, 10000)
    Y_data[np.random.rand(*Y_data.shape) < 0.9] = 0
    Y_sparse = sparse.csr_matrix(Y_data)
    
    # Run with GPU backend (best for sparse matrices)
    beta, se, zscore, pvalue = ridge_batch(
        X, Y_sparse, 
        lambda_val=1000, 
        n_rand=100,
        method='gpu',
        batch_size=500
    )

Notes
-----

- The batch size should be chosen based on available memory
- For GPU backend, smaller batch sizes help manage GPU memory
- Progress reporting is available with ``verbose > 0``
- The function automatically manages cleanup between batches
- Currently only supports permutation testing (n_rand > 0)

See Also
--------
:func:`ridge_inference.ridge.ridge`: Non-batched version of ridge regression
