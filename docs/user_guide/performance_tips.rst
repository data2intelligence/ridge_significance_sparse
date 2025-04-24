=================
Performance Tips
=================

This guide provides optimization tips to maximize the performance of RidgeInference for different scenarios.

Choosing the Right Backend
==========================

The most important performance decision is selecting the appropriate backend:

1. **GPU/CuPy Backend**
   
   - Best for: Large matrices, sparse data, many permutations
   - Requirements: CUDA-compatible GPU with recent drivers
   - Environment: ``RIDGE_INFERENCE_DISABLE_GPU=0`` (default)
   
   Performance tips:
   
   - Use batch processing for very large datasets
   - Monitor GPU memory usage (returned in results dictionary)
   - Pre-convert large Y matrices to sparse format if sparsity > 90%

2. **C/OpenMP Backend**
   
   - Best for: Dense matrices on multi-core CPUs
   - Requirements: Compiled C extension with OpenMP support
   - Environment: Set ``OMP_NUM_THREADS=N`` for N CPU cores
   
   Performance tips:
   
   - Use all available CPU cores: ``export OMP_NUM_THREADS=$(nproc)``
   - Ensure matrices are C-contiguous and float64 type
   - Pre-convert sparse matrices to dense if sparsity < 70%

3. **Numba Backend**
   
   - Best for: Medium-sized dense matrices, systems without GPU/C backend
   - Requirements: Numba package installed
   - Environment: Set ``NUMBA_NUM_THREADS=N`` for parallel execution
   
   Performance tips:
   
   - First-run compilation takes time, subsequent runs are faster
   - Ensure matrices are C-contiguous and float64 type
   - Pre-heat by running a small test calculation first

4. **Python/NumPy Backend**
   
   - Best for: Small datasets, sparse matrices without GPU
   - The only backend supporting t-test (n_rand=0)
   
   Performance tips:
   
   - Use sparse matrices if sparsity > 80%
   - Consider installing Intel MKL for faster NumPy operations
   - Use batch processing for large datasets

Memory Optimization
===================

Efficiently managing memory is crucial for large-scale analyses:

1. **Batch Processing**
   
   - Use batch processing for datasets with many samples
   - Tune batch size based on available memory:
     - CPU: ``batch_size = min(5000, memory_gb * 1000)``
     - GPU: ``batch_size = min(2000, gpu_memory_gb * 500)``
   - Enable garbage collection between batches

2. **Sparse Matrices**
   
   - Memory savings depend on sparsity level:
     - 90% zeros: ~10x memory reduction
     - 99% zeros: ~100x memory reduction
   - Use SciPy's CSR format for Y matrices
   - For very large sparse matrices, consider incrementally loading Y

3. **Mixed Precision**
   
   - Default: All computations use float64 (double precision)
   - For memory-constrained situations:
     - Input matrices can be float32 (will be converted internally)
     - Result matrices will always be float64 for precision

4. **Cleanup Between Runs**
   
   - Explicitly delete large matrices when no longer needed
   - Call ``gc.collect()`` to reclaim memory
   - For GPU: ``cp.get_default_memory_pool().free_all_blocks()``

Permutation Testing Optimization
================================

Permutation testing (n_rand > 0) is the most compute-intensive part:

1. **Choosing n_rand**
   
   - Typical values: 100-1000 for exploratory, 1000-10000 for publication
   - Minimum for reliable p-values: ~100 permutations
   - Each permutation adds linear computational cost
   
   Trade-off guidelines:
   
   +--------+-------------------+------------------+
   | n_rand | Precision at      | Relative         |
   |        | p=0.05            | Speed            |
   +========+===================+==================+
   | 100    | ±0.02             | Fastest          |
   +--------+-------------------+------------------+
   | 1000   | ±0.007            | Medium           |
   +--------+-------------------+------------------+
   | 10000  | ±0.002            | Slow             |
   +--------+-------------------+------------------+

2. **T-test Alternative**
   
   - For quick exploratory analyses, consider t-test (n_rand=0)
   - Requires method='python' or method='auto'
   - Much faster but less robust for non-normal data

3. **Lambda Parameter**
   
   - Higher lambda reduces variance but increases bias
   - Default (5e5) works well for most genomic analyses
   - For small datasets, consider smaller lambda (1e3-1e4)

Input Data Preparation
======================

Properly preparing input data can significantly improve performance:

1. **Data Types**
   
   - Ensure X and Y are NumPy arrays or pandas DataFrames
   - SciPy sparse matrices (CSR format) work well for sparse Y
   - DataFrames are converted to NumPy internally

2. **Column vs. Row Filtering**
   
   - Filter genes (rows) before running ridge regression
   - Keep only features (X columns) that are relevant
   - Y column filtering can be done efficiently in batches

3. **Missing Values**
   
   - Replace NaN values before analysis (imputation or removal)
   - Zero values are handled efficiently, NaN values are not

4. **Normalization**
   
   - Column-wise scaling improves numerical stability
   - Use ``scale_method='column'`` in high-level functions

Large-Scale Analysis
====================

For very large-scale analyses:

1. **Multi-Dataset Strategy**
   
   - Split datasets by sample groups
   - Process each group separately
   - Merge results afterward

2. **Checkpointing**
   
   - Save intermediate results after each batch
   - Resume from checkpoints if processing is interrupted

3. **Distribution**
   
   - For cluster environments, split Y columns across nodes
   - Each node processes a subset of samples
   - Combine results using pandas concatenation

Benchmarking
============

Approximate performance benchmarks on typical hardware:

+-------------------+----------------+---------------+----------------+-----------------+
| Matrix Size       | NumPy          | Numba         | C/OpenMP (8c)  | GPU (RTX 3080)  |
+===================+================+===============+================+=================+
| 1K genes, 10 feat | 0.3s           | 0.1s          | 0.05s          | 0.2s            |
| 100 samples       |                |               |                |                 |
+-------------------+----------------+---------------+----------------+-----------------+
| 10K genes, 50 feat| 15s            | 6s            | 3s             | 1s              |
| 1K samples        |                |               |                |                 |
+-------------------+----------------+---------------+----------------+-----------------+
| 20K genes, 100    | 25min          | 8min          | 4min           | 45s             |
| feat, 10K samples |                |               |                |                 |
+-------------------+----------------+---------------+----------------+-----------------+

*Times are for n_rand=100, will scale linearly with n_rand*

Interactive Performance Dashboard
---------------------------------

For a more comprehensive and interactive view of performance benchmarks across different
datasets, backends, and hardware configurations, check out the built-in performance dashboard:

.. raw:: html

   <div class="dashboard-link">
     <a href="../ComparePy2R/dashboard/index.html" class="dashboard-button">
       View Interactive Benchmarks
     </a>
   </div>

The dashboard provides visualizations of actual benchmark results from the comparison test suite,
allowing you to make informed decisions about which backend to use for your specific needs.

Profiling and Debugging
=======================

For performance troubleshooting:

1. **Logging Levels**
   
   - Set verbose=1 for basic progress information
   - Set verbose=2 for detailed timing information
   - Check backend selection with result['method_used']

2. **Common Performance Issues**
   
   - Slow first run: Numba compilation or GPU initialization
   - Out of memory: Reduce batch size or n_rand
   - GPU errors: Update CuPy/CUDA or use CPU backend

3. **Memory Profiling**
   
   - Track CPU memory: Use the ``memory_profiler`` package
   - Track GPU memory: Check result['peak_gpu_pool_mb']

