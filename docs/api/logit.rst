=====
Logit
=====

.. module:: ridge_inference.logit

The ``logit`` module implements logistic regression with significance testing and Firth correction.

This module provides functionality for logistic regression with optional Firth correction to handle
separation issues. It is useful for classification problems and complements the ridge regression
functionality in the package.

Main Function
------------

.. autofunction:: logit

Function Details
--------------

The ``logit`` function implements logistic regression with the following features:

- Maximum likelihood estimation of coefficients
- Standard error calculation
- Z-score and p-value computation
- Optional Firth correction for separation issues
- Robust convergence handling

Firth Correction
--------------

The Firth correction helps address separation issues in logistic regression. Separation occurs when
a predictor or combination of predictors perfectly separates the binary outcome, which can lead to
infinite coefficient estimates.

The correction works by modifying the score function to reduce bias:

.. math::

   U^*(β) = U(β) - 0.5 \text{trace}(I(β)^{-1} \frac{\partial I(β)}{\partial β})

where :math:`U` is the score function and :math:`I` is the information matrix.

In practice, this is implemented by adding a penalty term to the likelihood function.

Error Codes
---------

The module defines error codes for common issues:

- ``REG_FAIL = -1``: Regression setup failure
- ``CONVERGE_FAIL = -2``: Convergence failure after maximum iterations

Convergence Parameters
-------------------

The function allows control over the convergence process:

- ``tol``: Convergence tolerance for gradient magnitude
- ``max_delta``: Maximum step size in each iteration
- ``maxIter``: Maximum number of iterations

Implementation Details
-------------------

The implementation follows these steps:

1. **Initialization**: Set up initial beta vector (all zeros)
2. **Iterative Optimization**: Newton-Raphson method with step size control
3. **Firth Correction**: Apply correction to error vector if requested
4. **Standard Error Calculation**: Compute from diagonal elements of the inverse information matrix
5. **Statistical Tests**: Compute Z-scores and p-values using Wald test

Examples
--------

Basic usage::

    from ridge_inference.logit import logit
    import numpy as np
    
    # Generate synthetic binary outcome data
    n_samples = 1000
    n_features = 5
    
    # Create predictor matrix
    X = np.random.randn(n_samples, n_features)
    
    # Create binary outcome
    true_beta = np.array([0.5, -0.3, 0.2, 0.8, -0.5])
    prob = 1.0 / (1.0 + np.exp(-X @ true_beta))
    Y = (np.random.rand(n_samples) < prob).astype(float)
    
    # Run logistic regression
    beta, stderr, zscore, pvalue = logit(X, Y, tol=1e-6, maxIter=100)
    
    # Print results
    for i in range(n_features):
        print(f"Feature {i+1}: beta={beta[i]:.4f}, SE={stderr[i]:.4f}, Z={zscore[i]:.4f}, p={pvalue[i]:.4f}")

With Firth correction for separation issues::

    from ridge_inference.logit import logit
    import numpy as np
    
    # Generate data with perfect separation
    n_samples = 100
    X = np.vstack([np.random.randn(50, 2) - 2, np.random.randn(50, 2) + 2])
    Y = np.array([0] * 50 + [1] * 50)
    
    # Run logistic regression with Firth correction
    beta_firth, stderr_firth, zscore_firth, pvalue_firth = logit(
        X, Y, correction=1, tol=1e-6, maxIter=100
    )
    
    # Run logistic regression without correction
    beta_std, stderr_std, zscore_std, pvalue_std = logit(
        X, Y, correction=0, tol=1e-6, maxIter=100
    )
    
    # Compare results
    print("With Firth correction:")
    print(f"Beta: {beta_firth}")
    print("Without Firth correction:")
    print(f"Beta: {beta_std}")  # May show very large values due to separation

Implementation Notes
------------------

- The module uses NumPy for matrix operations
- The implementation follows the standard Newton-Raphson algorithm for logistic regression
- The module provides detailed logging at different verbosity levels
- The function gracefully handles singular matrices and convergence issues

References
---------

- Firth, D. (1993). Bias reduction of maximum likelihood estimates. *Biometrika*, 80(1), 27-38.
- Heinze, G., & Schemper, M. (2002). A solution to the problem of separation in logistic regression. *Statistics in Medicine*, 21(16), 2409-2419.

See Also
--------
:func:`ridge_inference.ridge.ridge`: Main ridge regression function
