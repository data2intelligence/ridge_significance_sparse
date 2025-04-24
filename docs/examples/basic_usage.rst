===========
Basic Usage
===========

This page provides step-by-step examples of how to use RidgeInference for basic ridge regression tasks.

Simple Ridge Regression
=======================

The most basic usage involves calling the ``ridge`` function with input matrices:

.. code-block:: python

    import numpy as np
    from ridge_inference import ridge
    
    # Create sample data
    n_genes, n_features, n_samples = 1000, 10, 5
    X = np.random.randn(n_genes, n_features)
    Y = np.random.randn(n_genes, n_samples)
    
    # Run ridge regression with permutation testing
    result = ridge(X, Y, lambda_=1000, n_rand=100)
    
    # Access results
    beta = result['beta']
    se = result['se']
    zscore = result['zscore']
    pvalue = result['pvalue']
    
    print(f"Method used: {result['method_used']}")
    print(f"Execution time: {result['execution_time']:.2f} seconds")
    
    # Find significant coefficients
    significant = (pvalue < 0.05)
    print(f"Number of significant coefficients: {np.sum(significant)}")

Using T-test Instead of Permutation
===================================

For t-test based significance testing, set ``n_rand=0``:

.. code-block:: python

    # Run ridge regression with t-test
    result_ttest = ridge(X, Y, lambda_=1000, n_rand=0)
    
    # Compare p-values from t-test vs. permutation
    pvalue_ttest = result_ttest['pvalue']
    
    # Print average p-values
    print(f"Average p-value (permutation): {np.mean(pvalue):.4f}")
    print(f"Average p-value (t-test): {np.mean(pvalue_ttest):.4f}")

Working with Pandas DataFrames
==============================

RidgeInference accepts pandas DataFrames as input, which is useful for keeping track of gene and sample names:

.. code-block:: python

    import pandas as pd
    
    # Create DataFrames with row and column names
    gene_names = [f"gene_{i}" for i in range(n_genes)]
    feature_names = [f"feature_{i}" for i in range(n_features)]
    sample_names = [f"sample_{i}" for i in range(n_samples)]
    
    X_df = pd.DataFrame(X, index=gene_names, columns=feature_names)
    Y_df = pd.DataFrame(Y, index=gene_names, columns=sample_names)
    
    # Run ridge regression
    result_df = ridge(X_df, Y_df, lambda_=1000, n_rand=100)
    
    # Results maintain feature and sample names
    beta_df = pd.DataFrame(
        result_df['beta'], 
        index=feature_names,
        columns=sample_names
    )
    
    # Display top coefficients for the first sample
    top_features = beta_df.iloc[:, 0].abs().nlargest(5).index
    print(beta_df.loc[top_features, sample_names[0]])

Specifying a Backend
====================

You can explicitly specify which backend to use:

.. code-block:: python

    # Use NumPy backend
    result_numpy = ridge(X, Y, lambda_=1000, n_rand=100, method='python')
    
    # Use GPU backend (if available)
    result_gpu = ridge(X, Y, lambda_=1000, n_rand=100, method='gpu')
    
    # Use C backend (if available)
    result_c = ridge(X, Y, lambda_=1000, n_rand=100, method='c')
    
    # Use Numba backend (if available)
    result_numba = ridge(X, Y, lambda_=1000, n_rand=100, method='numba')
    
    # Compare execution times
    backends = ['numpy', 'gpu', 'c', 'numba']
    times = [
        result_numpy.get('execution_time', float('nan')),
        result_gpu.get('execution_time', float('nan')),
        result_c.get('execution_time', float('nan')),
        result_numba.get('execution_time', float('nan'))
    ]
    
    for backend, time in zip(backends, times):
        if not np.isnan(time):
            print(f"{backend}: {time:.2f} seconds")

Working with Sparse Matrices
============================

For large, sparse datasets, you can use SciPy's sparse matrices:

.. code-block:: python

    from scipy import sparse
    
    # Create a sparse Y matrix (95% zeros)
    Y_dense = np.random.randn(n_genes, n_samples)
    Y_dense[np.random.rand(*Y_dense.shape) < 0.95] = 0
    Y_sparse = sparse.csr_matrix(Y_dense)
    
    # Run ridge regression with sparse Y
    result_sparse = ridge(X, Y_sparse, lambda_=1000, n_rand=100)
    
    print(f"Dense matrix size: {Y_dense.nbytes / 1e6:.2f} MB")
    print(f"Sparse matrix size: {Y_sparse.data.nbytes / 1e6:.2f} MB")
    print(f"Method used: {result_sparse['method_used']}")

Setting the Regularization Parameter
====================================

The regularization parameter (lambda) controls the strength of the ridge penalty:

.. code-block:: python

    # Create synthetic data with some noise
    n_genes, n_features, n_samples = 100, 5, 3
    X = np.random.randn(n_genes, n_features)
    true_beta = np.random.randn(n_features, n_samples)
    Y = X @ true_beta + 0.5 * np.random.randn(n_genes, n_samples)
    
    # Try different lambda values
    lambdas = [1, 10, 100, 1000, 10000]
    mse_values = []
    
    for lambda_val in lambdas:
        result = ridge(X, Y, lambda_=lambda_val, n_rand=100)
        beta_est = result['beta']
        mse = np.mean((beta_est - true_beta)**2)
        mse_values.append(mse)
        print(f"Lambda = {lambda_val}, MSE = {mse:.4f}")
    
    # Find optimal lambda
    optimal_idx = np.argmin(mse_values)
    print(f"Optimal lambda = {lambdas[optimal_idx]}")

Error Handling
==============

RidgeInference includes built-in error handling and fallback mechanisms:

.. code-block:: python

    try:
        # Try using the GPU backend
        result = ridge(X, Y, lambda_=1000, n_rand=100, method='gpu')
        print(f"Successfully used GPU: {result['method_used']}")
    except RuntimeError as e:
        print(f"GPU failed: {e}")
        # Fallback to CPU
        result = ridge(X, Y, lambda_=1000, n_rand=100, method='python')
        print(f"Fallback to: {result['method_used']}")
    
    # With auto method, fallbacks happen automatically
    auto_result = ridge(X, Y, lambda_=1000, n_rand=100, method='auto')
    print(f"Auto method selected: {auto_result['method_used']}")
