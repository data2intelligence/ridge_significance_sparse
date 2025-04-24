==========
C Bindings
==========

.. module:: ridge_inference.c_bindings

The ``c_bindings`` module provides Python bindings to the C implementation of ridge regression.

This module loads the RidgeInf C shared library and provides Python interfaces to call the
optimized C implementation. The C backend uses OpenMP parallelization for maximum
performance on multi-core CPUs.

Main Function
------------

.. function:: ridge_regression_c(X, Y, lambda_val, n_rand)

   Python wrapper for the C ridgeReg function.

   :param X: Input matrix of shape (n_genes, n_features)
   :param Y: Output matrix of shape (n_genes, n_samples)
   :param lambda_val: Ridge regularization parameter
   :param n_rand: Number of permutations (if 0, t-test is used)
   :return: Dict with beta, se, zscore, pvalue matrices

Utility Functions
---------------

.. function:: is_c_available()

   Check if the C implementation is available.

   :return: Boolean indicating availability

.. function:: _find_library()

   Internal function to find the C shared library.

   :return: Path to the library

Implementation Details
--------------------

Library Loading
~~~~~~~~~~~~

The module attempts to load the C shared library by searching in several locations:

1. Same directory as the Python module
2. src/ directory in the package root
3. Parent directory of the module
4. System search paths

If the library is found, the module configures the function prototypes and sets
``C_AVAILABLE = True``.

C Function Interface
~~~~~~~~~~~~~~~~~

The C implementation exposes a single function:

.. code-block:: c

   void ridgeReg(
       double *X_vec,          // Input matrix (flattened)
       double *Y_vec,          // Output matrix (flattened)
       int *n_pt,              // Number of genes
       int *p_pt,              // Number of features
       int *m_pt,              // Number of samples
       double *lambda_pt,      // Ridge parameter
       double *nrand_pt,       // Number of permutations
       double *beta_vec,       // Output: Beta matrix (flattened)
       double *se_vec,         // Output: SE matrix (flattened)
       double *zscore_vec,     // Output: Z-score matrix (flattened)
       double *pvalue_vec      // Output: P-value matrix (flattened)
   );

The Python wrapper handles conversion between NumPy arrays and C data structures.

Test Types
~~~~~~~~

The C implementation supports two types of significance tests:

1. **Permutation Test** (n_rand > 0): Permutes the rows of Y to establish a null distribution.
2. **T-Test** (n_rand = 0): Uses a t-distribution for significance testing.

OpenMP Parallelization
~~~~~~~~~~~~~~~~~~~

The C implementation uses OpenMP for parallel processing:

- The number of threads can be controlled with the ``OMP_NUM_THREADS`` environment variable
- By default, it uses all available cores

Limitations
----------

The C backend has some limitations:

- Requires dense matrices (does not support sparse input)
- Input matrices must be convertible to float64 (double precision)
- Requires OpenMP support in the compiler

Error Handling
------------

The module includes robust error handling:

- Detailed error messages for library load failures
- Validation of input shapes and types
- Safe propagation of errors from C to Python

Examples
--------

Using the C backend directly::

    from ridge_inference.c_bindings import ridge_regression_c, is_c_available
    
    # Check if the C backend is available
    if is_c_available():
        # Create sample data
        X = np.random.randn(1000, 10)
        Y = np.random.randn(1000, 5)
        
        # Run ridge regression with the C backend
        result = ridge_regression_c(X, Y, lambda_val=1000, n_rand=100)
        
        # Access results
        beta = result['beta']
        pvalue = result['pvalue']
    else:
        print("C backend not available.")

Setting the number of OpenMP threads::

    import os
    
    # Set the number of threads before importing the module
    os.environ['OMP_NUM_THREADS'] = '4'
    
    from ridge_inference.c_bindings import ridge_regression_c
    
    # Now the C backend will use 4 threads
    result = ridge_regression_c(X, Y, lambda_val=1000, n_rand=100)

See Also
--------
:func:`ridge_inference.ridge.ridge`: High-level ridge function
