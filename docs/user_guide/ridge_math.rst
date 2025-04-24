.. _ridge_math:

Mathematical Explanation of Ridge Inference
===========================================

This document provides a detailed mathematical overview of the Ridge Inference package, explaining how it uses ridge regression and significance testing to assess associations between features and samples in gene expression data.

.. contents:: Table of Contents
   :depth: 2
   :local:

Problem Setup
-------------

- **Signature Matrix** :math:`\mathbf{X}`: A matrix :math:`\mathbf{X} \in \mathbb{R}^{n \times p}`, where:
  - :math:`n` is the number of genes.
  - :math:`p` is the number of features (e.g., biological signatures or proteins).
- **Gene Expression Matrix** :math:`\mathbf{Y}`: A matrix :math:`\mathbf{Y} \in \mathbb{R}^{n \times m}`, where:
  - :math:`m` is the number of samples.
- **Objective**: Estimate a coefficient matrix :math:`\mathbf{B} \in \mathbb{R}^{p \times m}` that quantifies the association between each feature in :math:`\mathbf{X}` and each sample in :math:`\mathbf{Y}`, while accounting for potential multicollinearity and overfitting.

Ridge Regression
----------------

Objective Function
^^^^^^^^^^^^^^^^^^

The package performs multi-output ridge regression to estimate :math:`\mathbf{B}`. The optimization problem is defined as:

.. math::

   \min_{\mathbf{B}} \left\| \mathbf{Y} - \mathbf{X} \mathbf{B} \right\|_F^2 + \lambda \left\| \mathbf{B} \right\|_F^2

- :math:`\left\| \cdot \right\|_F`: The Frobenius norm, defined for a matrix :math:`\mathbf{A}` as :math:`\left\| \mathbf{A} \right\|_F = \sqrt{\sum_{i,j} a_{ij}^2}`.
- :math:`\lambda > 0`: The regularization parameter, controlling the trade-off between fitting the data and penalizing the magnitude of :math:`\mathbf{B}`.

This formulation treats each column of :math:`\mathbf{Y}` (each sample) as a separate output, with :math:`\mathbf{B}` mapping features to all samples simultaneously.

Closed-Form Solution
^^^^^^^^^^^^^^^^^^^^

The solution to the ridge regression problem can be derived by setting the gradient of the objective function to zero. The closed-form expression for :math:`\mathbf{B}` is:

.. math::

   \mathbf{B} = (\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^T \mathbf{Y}

- :math:`\mathbf{X}^T \mathbf{X}`: A :math:`p \times p` matrix capturing feature correlations.
- :math:`\lambda \mathbf{I}`: A :math:`p \times p` identity matrix scaled by :math:`\lambda`, ensuring the matrix :math:`\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I}` is invertible even if :math:`\mathbf{X}^T \mathbf{X}` is singular.
- :math:`\mathbf{X}^T \mathbf{Y}`: A :math:`p \times m` matrix of cross-correlations between features and samples.

To enhance computational efficiency, the package computes this in two steps:

1. **Compute** :math:`\mathbf{T}`: :math:`\mathbf{T} = (\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^T`, a :math:`p \times n` matrix.

2. **Compute** :math:`\mathbf{B}`: :math:`\mathbf{B} = \mathbf{T} \mathbf{Y}`, a :math:`p \times m` matrix.

This factorization allows :math:`\mathbf{T}` to be reused across computations, particularly beneficial in batch processing or permutation testing.

Significance Testing
--------------------

Permutation Test
^^^^^^^^^^^^^^^^

When the number of permutations :math:`n_{\text{rand}} > 0`, the package employs a permutation test to assess significance non-parametrically:

1. **Procedure**:

   - For :math:`k = 1, 2, \dots, n_{\text{rand}}`:
   
     - Randomly permute the rows of :math:`\mathbf{Y}` to create :math:`\mathbf{Y}_{\text{perm}}`.
     
     - Compute the permuted coefficients: :math:`\mathbf{B}_{\text{rand}}^{(k)} = \mathbf{T} \mathbf{Y}_{\text{perm}}`.
   
   - For each coefficient :math:`b_{ij}` in :math:`\mathbf{B}`:
   
     - **Mean**: :math:`\mu_{ij} = \frac{1}{n_{\text{rand}}} \sum_{k=1}^{n_{\text{rand}}} b_{\text{rand}, ij}^{(k)}`.
     
     - **Variance**: :math:`\sigma_{ij}^2 = \frac{1}{n_{\text{rand}}} \sum_{k=1}^{n_{\text{rand}}} \left( b_{\text{rand}, ij}^{(k)} \right)^2 - \mu_{ij}^2`, clamped to :math:`\geq 0`.
     
     - **Standard Error**: :math:`\text{se}_{ij} = \sqrt{\sigma_{ij}^2}`.
     
     - **Z-Score**: :math:`z_{ij} = \frac{b_{ij} - \mu_{ij}}{\text{se}_{ij}}` (set to 0 if :math:`\text{se}_{ij} < \epsilon`, where :math:`\epsilon` is a small tolerance).
     
     - **P-Value**: :math:`p_{ij} = \frac{\text{count}(|b_{\text{rand}, ij}^{(k)}| \geq |b_{ij}|) + 1}{n_{\text{rand}} + 1}`, where "count" is the number of permutations satisfying the condition.

2. **Purpose**:

   - This method evaluates the null hypothesis that the association between features and samples is due to chance, by comparing the observed coefficients against a distribution generated under random gene permutations.

T-Test
^^^^^^

When :math:`n_{\text{rand}} = 0`, the package uses a parametric t-test, suitable only with the Python (NumPy) backend:

1. **Procedure**:

   - Compute predicted values: :math:`\hat{\mathbf{Y}} = \mathbf{X} \mathbf{B}`.
   
   - Compute residuals: :math:`\mathbf{R} = \mathbf{Y} - \hat{\mathbf{Y}}`.
   
   - **Degrees of Freedom (df)**:
   
     - If :math:`\lambda = 0` (Ordinary Least Squares, OLS): :math:`df = n - p`.
     
     - If :math:`\lambda > 0` (Ridge): :math:`df = n - \text{trace}(\mathbf{H})`, where :math:`\mathbf{H} = \mathbf{X} \mathbf{T}` is the hat matrix, and :math:`\text{trace}(\mathbf{H})` is its effective degrees of freedom.
     
     - If :math:`df \leq 0`, it is set to 1 to avoid invalid computations.
   
   - **Residual Variance**: :math:`\sigma_j^2 = \frac{1}{df} \sum_{i=1}^n r_{ij}^2` for each sample :math:`j`.
   
   - **Standard Error**:
   
     - OLS: :math:`\text{se}_{ij} = \sqrt{\sigma_j^2 \cdot [\mathbf{X}^T \mathbf{X}]^{-1}_{ii}}`.
     
     - Ridge: :math:`\text{se}_{ij} = \sqrt{\sigma_j^2 \cdot \sum_{k=1}^n t_{ik}^2}`, where :math:`t_{ik}` are elements of :math:`\mathbf{T}`.
   
   - **T-Statistic**: :math:`t_{ij} = \frac{b_{ij}}{\text{se}_{ij}}` (set to 0 if :math:`\text{se}_{ij} < \epsilon`).
   
   - **P-Value**: :math:`p_{ij} = 2 \cdot (1 - F(|t_{ij}|; df))`, where :math:`F` is the cumulative distribution function of the t-distribution with :math:`df` degrees of freedom.

2. **Purpose**:

   - The t-test assumes a linear model under normality of residuals, providing a parametric alternative to the permutation test, though it requires dense inputs and is less flexible with sparse data.

Implementation Details
----------------------

Batch Processing
^^^^^^^^^^^^^^^^

For large datasets (e.g., when :math:`m` exceeds a threshold), the package supports batch processing:

- :math:`\mathbf{Y}` is split into batches along the sample dimension (columns).
- :math:`\mathbf{T}` is computed once, and :math:`\mathbf{B} = \mathbf{T} \mathbf{Y}_{\text{batch}}` is calculated for each batch.
- Results are concatenated to form the full :math:`\mathbf{B}`, :math:`\text{se}`, :math:`\mathbf{z}`, and :math:`\mathbf{p}` matrices.
- Currently, batching is implemented only for permutation testing.

Computational Backends
^^^^^^^^^^^^^^^^^^^^^^

The package optimizes performance with multiple backends:

- **Python (NumPy)**: Handles dense or sparse :math:`\mathbf{X}` and :math:`\mathbf{Y}`, supports both permutation and t-tests.
- **Numba**: Requires dense inputs, accelerates permutation testing with JIT compilation (no t-test support).
- **CuPy (GPU)**: Handles dense :math:`\mathbf{X}` and dense/sparse :math:`\mathbf{Y}`, optimized for permutation testing on GPUs (no t-test support).
- **C (GSL)**: Requires dense inputs, uses GSL for high-performance linear algebra with OpenMP parallelization, supports both permutation and t-tests.

Each backend computes :math:`\mathbf{T}` and :math:`\mathbf{B}` consistently, with variations in how sparse matrices and significance tests are handled based on their capabilities.

Dense vs. Sparse Performance Considerations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

While sparse matrix representations typically reduce memory usage for matrices with many zero elements, their computational efficiency depends on several factors:

1. **Permutation Operations**:

   - For row-based permutations in :math:`\mathbf{Y}_{\text{perm}}`, sparse matrices (particularly in CSR format) may experience overhead due to disruption of their optimized storage pattern.
   
   - The operation :math:`\mathbf{Y}_{\text{perm}} = \mathbf{Y}[perm\_idx, :]` requires significant reorganization for sparse matrices, potentially increasing both computation time and memory usage.

2. **Memory Management**:

   - During permutation testing, creating and destroying numerous sparse matrices can lead to memory fragmentation, especially on GPUs.
   
   - The overhead of sparse matrix indexing and construction during each permutation iteration can exceed the benefits of reduced memory footprint.

3. **Mixed Dense-Sparse Operations**:

   - The core computation :math:`\mathbf{B} = \mathbf{T} \mathbf{Y}` involves multiplying a dense matrix (:math:`\mathbf{T}`) with a potentially sparse matrix (:math:`\mathbf{Y}`).
   
   - The optimal performance characteristics of dense-sparse matrix multiplication vary by implementation (NumPy, CuPy) and can be counterintuitive.
   
   - In the GPU implementation, :math:`\mathbf{X}` is always densified regardless of input format, while :math:`\mathbf{Y}` can remain sparse, creating a hybrid approach that isn't always optimal.

4. **Density Threshold**:

   - A matrix's sparsity level (percentage of zero elements) determines whether sparse representation offers benefits.
   
   - For matrices with sparsity below a certain threshold (typically 90-95%), dense representation may actually be more efficient for both memory and computation.

5. **GPU-Specific Considerations**:

   - GPU implementations of sparse matrix operations may have different performance characteristics than CPU implementations.
   
   - The overhead of transferring permutation indices between CPU and GPU can impact performance.
   
   - GPU memory allocation and deallocation patterns are particularly important for sparse matrices.

For optimal performance, the package can dynamically select between sparse and dense representations based on matrix density and operation characteristics. In practice, dense matrix operations often perform better than expected, especially on GPUs with their highly optimized dense linear algebra routines.

Interpretation
--------------

- :math:`\mathbf{B}`: Each element :math:`b_{ij}` represents the inferred association strength between feature :math:`i` and sample :math:`j`.
- **Standard Error** (:math:`\text{se}`): Indicates the variability of :math:`b_{ij}` estimates.
- **Z-Score/T-Statistic**: Measures the magnitude of :math:`b_{ij}` relative to its standard error.
- **P-Value**: Assesses the probability of observing :math:`b_{ij}` (or a more extreme value) under the null hypothesis of no association.

This framework enables robust inference of feature-sample relationships in gene expression data, balancing computational efficiency and statistical rigor.
