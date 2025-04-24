=====
Utils
=====

.. module:: ridge_inference.utils

The ``utils`` module provides utility functions for data loading, preprocessing, visualization, and other common operations.

This module contains helper functions used throughout the RidgeInference package for tasks like loading signature matrices, data scaling, and visualization of results.

Data Loading Functions
--------------------

.. autofunction:: load_signature_matrix

This function provides memory-efficient loading of signature matrices, with special handling for built-in matrices:

- ``"SecAct"``: Default SecAct signature matrix
- ``"CytoSig"``: CytoSig signature matrix

Data Preprocessing Functions
--------------------------

.. autofunction:: scale_dataframe_columns

.. autofunction:: scale_data_matrix

.. autofunction:: find_overlapping_genes

.. autofunction:: is_sparse_beneficial

.. autofunction:: convert_to_sparse

These functions help prepare data for ridge regression:

- Scaling data (column-wise or global)
- Finding common genes between expression data and signature matrices
- Determining when sparse representation is beneficial
- Converting to sparse matrices when appropriate

Visualization Functions
---------------------

.. autofunction:: visualize_activity

This function creates visualizations of ridge regression results, showing top proteins by activity and significance.

Implementation Details
--------------------

Signature Matrix Loading
~~~~~~~~~~~~~~~~~~~~

The ``load_signature_matrix`` function uses several strategies to find and load signature matrices efficiently:

1. First checks if a DataFrame was passed directly
2. Looks for built-in matrices in the package data directory
3. Falls back to looking in alternative locations
4. Supports direct paths to custom matrices

The function handles both compressed (.gz) and uncompressed files, and uses memory mapping when possible for large uncompressed files.

Data Scaling
~~~~~~~~~

The module provides two scaling approaches:

1. **Column-wise scaling**: Each column (sample) is scaled independently
2. **Global scaling**: The entire matrix is scaled using global mean and standard deviation

Both approaches handle zero-variance columns gracefully.

Sparse Matrix Handling
~~~~~~~~~~~~~~~~~~

The module includes utilities to determine when sparse representation is beneficial and to convert matrices to sparse format:

- Default density threshold: 0.1 (10% non-zero elements)
- Minimum size threshold: 10^6 elements
- Uses SciPy's CSR format for efficient row slicing

Examples
--------

Loading a signature matrix::

    from ridge_inference.utils import load_signature_matrix
    
    # Load the default SecAct signature matrix
    sig_matrix = load_signature_matrix("SecAct")
    
    # Load a custom signature matrix
    custom_sig = load_signature_matrix("/path/to/custom_signature.tsv")
    
    # Print basic information
    print(f"Shape: {sig_matrix.shape}")
    print(f"Features: {sig_matrix.columns.tolist()[:5]}...")

Scaling data::

    from ridge_inference.utils import scale_dataframe_columns
    import pandas as pd
    
    # Create sample data
    data = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50]
    })
    
    # Scale column-wise
    scaled_data = scale_dataframe_columns(data)
    
    print("Original data:")
    print(data)
    print("\nScaled data:")
    print(scaled_data)

Finding overlapping genes::

    from ridge_inference.utils import find_overlapping_genes
    import pandas as pd
    
    # Create sample data
    expr_data = pd.DataFrame(index=['gene1', 'gene2', 'gene3', 'gene4'])
    sig_matrix = pd.DataFrame(index=['gene2', 'gene3', 'gene5', 'gene6'])
    
    # Find overlapping genes
    common_genes = find_overlapping_genes(expr_data, sig_matrix)
    
    print(f"Found {len(common_genes)} common genes: {common_genes}")

Visualizing results::

    from ridge_inference.utils import visualize_activity
    
    # Assume we have results from secact_inference
    result = {...}  # Output from secact_inference
    
    # Create visualization of top 10 proteins
    fig = visualize_activity(result, top_n=10, pvalue_threshold=0.05)
    
    # Save the figure
    fig.savefig("activity_visualization.png", dpi=300, bbox_inches="tight")

See Also
--------
:func:`ridge_inference.inference.secact_inference`: Main inference function
:func:`ridge_inference.inference.secact_activity_inference`: High-level inference wrapper
