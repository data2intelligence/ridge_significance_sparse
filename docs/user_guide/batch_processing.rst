=================
Batch Processing
=================

RidgeInference provides batch processing capabilities to handle large datasets efficiently. This guide explains how to use batch processing, when it's beneficial, and how to optimize its performance.

What is Batch Processing?
=========================

Batch processing splits the computation into smaller chunks that are processed sequentially to reduce memory usage. In RidgeInference, batch processing:

1. Calculates the T matrix (``(X'X + λI)^-1 X'``) once upfront
2. Processes columns of the Y matrix in batches
3. Concatenates the results at the end

This approach significantly reduces peak memory usage while maintaining computational efficiency.

When to Use Batch Processing
============================

Batch processing is beneficial when:

- Your dataset has many samples (columns in Y)
- The full computation would exceed available memory
- You're performing permutation testing with high n_rand
- You're using the GPU backend and have limited GPU memory

General rule of thumb:

- If ``n_samples * n_features * n_rand`` > 10^8, consider batch processing
- If using a GPU with limited memory (e.g., 8GB), batch processing is recommended for large datasets

Using Batch Processing
======================

There are two ways to use batch processing in RidgeInference:

1. **Direct batch function call**

   .. code-block:: python
   
       from ridge_inference.batch import ridge_batch
       
       # Batch processing with explicit batch size
       beta, se, zscore, pvalue = ridge_batch(
           X, Y, 
           lambda_val=1000, 
           n_rand=100,
           method='auto',
           batch_size=500,
           verbose=1
       )

2. **Through high-level inference functions**

   .. code-block:: python
   
       from ridge_inference.inference import secact_inference
       
       result = secact_inference(
           expr_data, 
           sig_matrix='SecAct',
           lambda_val=5e5,
           n_rand=1000,
           batch_size=500,                 # Set batch size
           batch_threshold=50,             # Auto-batch if > 50 samples
           verbose=1
       )

Automatic Batch Processing
==========================

High-level functions like ``secact_inference`` can automatically determine when to use batch processing:

.. code-block:: python

    result = secact_inference(
        expr_data, 
        sig_matrix='SecAct',
        lambda_val=5e5,
        n_rand=1000,
        batch_threshold=50,     # Use batches if n_samples > 50
        verbose=1
    )

The ``batch_threshold`` parameter controls when batch processing is activated.

Choosing Batch Size
===================

The optimal batch size depends on:

- Available memory
- Number of features
- Number of permutations (n_rand)
- Backend used

Guidelines for choosing batch size:

1. **For GPU backend**:
   - Start with batch_size = 1000 for moderate GPUs (8GB)
   - Start with batch_size = 2000-5000 for high-end GPUs (16GB+)
   - Reduce batch size if you encounter out-of-memory errors

2. **For CPU backends**:
   - Start with batch_size = 5000 for systems with 16GB RAM
   - Start with batch_size = 10000 for systems with 32GB+ RAM
   - Higher n_rand values need smaller batch sizes

3. **Automatic batch sizing**:
   - If batch_size is not specified (None), a heuristic is used to estimate appropriate size
   - The estimate considers number of features and permutations

Monitoring Progress
===================

Enable progress reporting with the ``verbose`` parameter:

.. code-block:: python

    result = secact_inference(
        expr_data, 
        sig_matrix='SecAct',
        batch_size=500,
        verbose=2  # Detailed progress information
    )

Verbosity levels:
- 0: No progress information
- 1: Basic progress information
- 2: Detailed progress information

Memory Management
=================

Batch processing includes automatic memory management:

- Intermediates are cleared after each batch
- Explicit garbage collection is performed
- GPU memory pool is cleared when using the GPU backend

Backend Considerations
======================

Batch performance varies by backend:

- **GPU/CuPy**: Excellent for batch processing, automatically manages GPU memory
- **C/OpenMP**: Very efficient with batches, especially for many permutations
- **Numba**: Good performance with batches, requires dense matrices
- **Python/NumPy**: Slowest but supports all matrix types

For sparse Y matrices, the GPU backend with batch processing is highly recommended.

Limitations
===========

Current limitations of the batch processing approach:

1. Only supports permutation testing (n_rand > 0)
2. The C backend recalculates the T matrix for each batch (less efficient)
3. Uses more disk I/O for large sparse matrices when swapping batches

Examples
========

Example with large sparse matrix:

.. code-block:: python

    import numpy as np
    from scipy import sparse
    from ridge_inference.batch import ridge_batch
    
    # Create large sparse Y (10,000 genes × 50,000 samples with 95% sparsity)
    n_genes, n_features, n_samples = 10000, 20, 50000
    X = np.random.randn(n_genes, n_features)
    
    Y_data = np.random.randn(n_genes, n_samples)
    Y_data[np.random.rand(*Y_data.shape) < 0.95] = 0
    Y_sparse = sparse.csr_matrix(Y_data)
    del Y_data  # Free memory
    
    # Process in batches of 1000 samples using GPU
    beta, se, zscore, pvalue = ridge_batch(
        X, Y_sparse, 
        lambda_val=1000, 
        n_rand=100,
        method='gpu',
        batch_size=1000,
        verbose=1
    )
