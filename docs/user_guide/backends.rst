====================
Backend Selection
====================

RidgeInference provides multiple backend implementations to optimize performance across different hardware configurations and data scenarios. This guide helps you choose the right backend for your specific needs.

Available Backends
==================

RidgeInference supports four backend implementations:

1. **Python/NumPy (``'python'``)**
   
   - Pure Python implementation using NumPy
   - Supports both dense and sparse matrices
   - Only backend that supports t-test significance testing (n_rand=0)
   - Slowest but most compatible option

2. **Numba (``'numba'``)**
   
   - JIT-compiled Python code using Numba
   - Requires dense matrices
   - Faster than pure Python for moderately sized matrices
   - Good option for laptops/desktops without a GPU

3. **GPU/CuPy (``'gpu'``)**
   
   - GPU-accelerated implementation using CuPy
   - Supports both dense and sparse matrices
   - Fastest option for large matrices when a compatible GPU is available
   - Automatically manages GPU memory

4. **C/OpenMP (``'c'``)**
   
   - C implementation with OpenMP parallelization
   - Requires dense matrices
   - Very fast on multi-core CPUs
   - Requires compilation of the C extension

Auto Selection Logic
====================

When ``method='auto'`` (the default), RidgeInference selects the best backend based on:

1. **Data characteristics**
   
   - For sparse matrices: Prefers GPU, falls back to Python
   - For dense matrices: Uses available backends in order of preference

2. **Test type**
   
   - For t-test (n_rand=0): Uses Python backend only
   - For permutation test: Considers all available backends

3. **Backend availability**
   
   - Checks which backends are available in the current environment
   - Falls back gracefully if preferred backends are unavailable

The automatic selection priority is: GPU → C → Numba → Python.

Performance Comparison
======================

Relative performance varies by data size and characteristics:

.. list-table:: Performance Comparison by Data Type
   :header-rows: 1
   :widths: 20 30 30 30

   * - Backend
     - Small Data
     - Large Dense Data
     - Large Sparse Data
   * - Python/NumPy
     - Slowest
     - Slowest
     - Medium
   * - Numba
     - Medium
     - Medium
     - Not Supported
   * - GPU/CuPy
     - Fast
     - Fastest
     - Fastest
   * - C/OpenMP
     - Fastest
     - Fast
     - Not Supported

*Note*: Performance depends on hardware. On systems without a GPU, C/OpenMP is typically fastest.

When to Choose Each Backend
===========================

**Python/NumPy Backend**

Use when:
- You need t-test significance testing (n_rand=0)
- You have sparse matrices and no GPU
- You need maximum compatibility
- Other backends are unavailable

**Numba Backend**

Use when:
- You have dense matrices of moderate size
- You want better performance than NumPy without compilation
- No GPU is available and C backend is unavailable

**GPU/CuPy Backend**

Use when:
- You have a compatible NVIDIA GPU
- You have very large matrices
- You have sparse matrices
- You need the fastest performance

**C/OpenMP Backend**

Use when:
- You have dense matrices
- You have a multi-core CPU
- No GPU is available
- You want the fastest CPU performance

Explicitly Selecting a Backend
==============================

To explicitly select a backend:

.. code-block:: python

    from ridge_inference import ridge
    
    # Use GPU backend
    result = ridge(X, Y, lambda_=1000, n_rand=100, method='gpu')
    
    # Use C backend
    result = ridge(X, Y, lambda_=1000, n_rand=100, method='c')
    
    # Use Numba backend
    result = ridge(X, Y, lambda_=1000, n_rand=100, method='numba')
    
    # Use Python backend
    result = ridge(X, Y, lambda_=1000, n_rand=100, method='python')

Fallback Mechanisms
===================

RidgeInference includes automatic fallback mechanisms:

1. If the selected backend fails, it tries the next available backend
2. If Y is sparse, it automatically avoids incompatible backends
3. For t-test, it automatically uses the Python backend

You can check which backend was actually used:

.. code-block:: python

    result = ridge(X, Y, lambda_=1000, n_rand=100, method='auto')
    print(f"Backend used: {result['method_used']}")

Checking Backend Availability
=============================

To check which backends are available in your environment:

.. code-block:: python

    from ridge_inference.c_bindings import is_c_available
    from ridge_inference.core import NUMBA_AVAILABLE, CUPY_AVAILABLE
    
    print(f"C backend: {is_c_available()}")
    print(f"Numba backend: {NUMBA_AVAILABLE}")
    print(f"GPU backend: {CUPY_AVAILABLE}")
