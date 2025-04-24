==========
Inference
==========

.. module:: ridge_inference.inference

The ``inference`` module provides high-level functions for performing inference on genomic data.

This module implements the SecAct (Secreted protein Activity) inference methodology, which analyzes
gene expression data to infer the activity of secreted proteins. It builds on the core ridge regression
functionality and adds domain-specific preprocessing and interpretation.

Main Functions
-------------

.. autofunction:: secact_inference

.. autofunction:: secact_activity_inference

Function Details
---------------

``secact_inference`` is the core inference function:

- Accepts expression data and signature matrices
- Handles preprocessing of inputs (scaling, common genes filtering)
- Automatically selects the most efficient computational approach
- Supports batch processing for large datasets

``secact_activity_inference`` is a higher-level wrapper:

- Calculates differential expression profiles
- Supports paired and unpaired designs
- Handles single-sample and aggregated analysis
- Provides options for signature filtering
- Returns results formatted for biological interpretation

Data Formats
-----------

Expected input formats:

- ``expr_data``: pandas DataFrame with genes as rows and samples as columns
- ``sig_matrix``: String name of predefined signature or pandas DataFrame
- Expression and signature values should be log-transformed

Available predefined signatures:

- ``"SecAct"``: Default signature for secreted protein activity
- ``"CytoSig"``: Cytokine signaling signature

Return Value
-----------

Both functions return a dictionary with the following keys:

- ``'beta'``: Activity coefficient matrix (pandas DataFrame)
- ``'se'``: Standard error matrix (pandas DataFrame)
- ``'zscore'``: Z-score matrix (pandas DataFrame)
- ``'pvalue'``: P-value matrix (pandas DataFrame)
- ``'method'``: Backend method used
- ``'execution_time'``: Total execution time in seconds
- ``'batched'``: Whether batch processing was used
- ``'batch_size'``: Batch size used (if applicable)

All result matrices have features (e.g., proteins) as rows and samples as columns.

Examples
--------

Basic usage::

    from ridge_inference.inference import secact_activity_inference
    import pandas as pd
    
    # Load expression data (genes × samples)
    treatment_expr = pd.read_csv("treatment_expr.csv", index_col=0)
    control_expr = pd.read_csv("control_expr.csv", index_col=0)
    
    # Run inference
    result = secact_activity_inference(
        treatment_expr,
        control_expr,
        is_differential=False,
        is_paired=True,
        lambda_val=5e5,
        n_rand=1000,
        method="auto",
        scale_method="column"
    )
    
    # Access results
    activities = result['beta']    # Protein activity scores
    pvalues = result['pvalue']     # Statistical significance

Using batch processing for large datasets::

    # For large datasets, enable batch processing
    result = secact_activity_inference(
        treatment_expr,
        control_expr,
        is_differential=False,
        is_paired=True,
        lambda_val=5e5,
        n_rand=1000,
        method="auto",
        batch_size=1000,            # Process in batches of 1000 samples
        batch_threshold=50          # Auto-batch if > 50 samples
    )

Notes
-----

- Scaling (``scale_method``) is recommended for most analyses
- For paired analysis, samples must have matching column names in treatment and control
- Adding a background column (``add_background=True``) can improve interpretability
- Batch processing is recommended for datasets with many samples

References
----------

For more information on the SecAct methodology:

1. Author et al. (20XX). "SecAct: Inference of Secreted Protein Activity from Gene Expression."
   *Journal Name*, XX(X), XXX-XXX.

See Also
--------
:func:`ridge_inference.utils.visualize_activity`: Visualization function for results
:func:`ridge_inference.ridge.ridge`: Underlying ridge regression function
