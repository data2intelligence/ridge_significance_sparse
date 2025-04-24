==============
Advanced Usage
==============

This page provides more advanced examples for using RidgeInference in complex scenarios.

Working with Genomic Data
=========================

Using RidgeInference for inference on real genomic datasets:

.. code-block:: python

    import pandas as pd
    import numpy as np
    from ridge_inference.inference import secact_activity_inference
    
    # Load gene expression data (genes × samples)
    treatment_expr = pd.read_csv("treatment_expr.csv", index_col=0)
    control_expr = pd.read_csv("control_expr.csv", index_col=0)
    
    # Run inference with paired samples
    result = secact_activity_inference(
        treatment_expr,
        control_expr,
        is_differential=False,  # Calculate differential from treatment and control
        is_paired=True,         # Samples are paired
        lambda_val=5e5,
        n_rand=1000,
        method="auto",
        scale_method="column",  # Scale each column independently
        add_background=True     # Add background term
    )
    
    # Access results
    activities = result['beta']
    pvalues = result['pvalue']
    
    # Find significant proteins (FDR < 0.05)
    from statsmodels.stats.multitest import multipletests
    
    # Apply FDR correction across all proteins
    flat_pvals = pvalues.values.flatten()
    _, qvals_flat, _, _ = multipletests(flat_pvals, method='fdr_bh', alpha=0.05)
    qvals = pd.DataFrame(qvals_flat.reshape(pvalues.shape), 
                         index=pvalues.index, columns=pvalues.columns)
    
    # Get significant proteins
    sig_mask = qvals < 0.05
    sig_proteins = activities[sig_mask].dropna(how='all')
    
    print(f"Found {len(sig_proteins)} significant proteins")
    
    # Save results
    activities.to_csv("protein_activities.csv")
    pvalues.to_csv("protein_pvalues.csv")
    qvals.to_csv("protein_qvalues.csv")

Handling Very Large Datasets
============================

Processing datasets with millions of samples using batch processing:

.. code-block:: python

    import numpy as np
    from scipy import sparse
    from ridge_inference.batch import ridge_batch
    import time
    
    # Create large synthetic dataset
    n_genes = 10000
    n_features = 20
    n_samples = 1000000  # 1 million samples
    sparsity = 0.95      # 95% zeros
    
    # Create dense X
    X = np.random.randn(n_genes, n_features)
    
    # Create sparse Y
    Y_data = np.random.randn(n_genes, n_samples)
    Y_data[np.random.rand(*Y_data.shape) < sparsity] = 0
    Y = sparse.csr_matrix(Y_data)
    del Y_data  # Free memory
    
    print(f"Y matrix shape: {Y.shape}, Memory usage: {Y.data.nbytes / 1e9:.2f} GB")
    
    # Process in batches with progress tracking
    start_time = time.time()
    beta, se, zscore, pvalue = ridge_batch(
        X, Y,
        lambda_val=1000,
        n_rand=100,
        method='gpu',     # Use GPU for best performance with sparse data
        batch_size=10000, # Process 10,000 samples at a time
        verbose=1         # Show progress information
    )
    elapsed = time.time() - start_time
    
    print(f"Processing completed in {elapsed:.2f} seconds")
    print(f"Results shape: {beta.shape}")

Creating Custom Cross-Validation
================================

Implementing cross-validation to find the optimal lambda parameter:

.. code-block:: python

    import numpy as np
    import pandas as pd
    from ridge_inference import ridge
    from sklearn.model_selection import KFold
    
    # Create synthetic data
    n_genes = 1000
    n_features = 10
    n_samples = 5
    
    X = np.random.randn(n_genes, n_features)
    true_beta = np.random.randn(n_features, n_samples)
    Y = X @ true_beta + 0.5 * np.random.randn(n_genes, n_samples)
    
    # Define lambda values to test
    lambda_values = [0.1, 1, 10, 100, 1000, 10000, 100000]
    
    # Setup cross-validation
    n_folds = 5
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Store results
    cv_scores = pd.DataFrame(index=lambda_values, columns=['MSE_mean', 'MSE_std'])
    
    # Run cross-validation
    for lambda_val in lambda_values:
        fold_mses = []
        
        for train_idx, test_idx in kf.split(X):
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]
            
            # Fit model
            result = ridge(X_train, Y_train, lambda_=lambda_val, n_rand=0)
            beta_est = result['beta']
            
            # Predict and evaluate
            Y_pred = X_test @ beta_est
            mse = np.mean((Y_test - Y_pred)**2)
            fold_mses.append(mse)
        
        # Store results
        cv_scores.loc[lambda_val, 'MSE_mean'] = np.mean(fold_mses)
        cv_scores.loc[lambda_val, 'MSE_std'] = np.std(fold_mses)
    
    # Find optimal lambda
    optimal_lambda = cv_scores['MSE_mean'].idxmin()
    print(f"Optimal lambda: {optimal_lambda}")
    print(cv_scores)
    
    # Fit final model with optimal lambda
    final_result = ridge(X, Y, lambda_=optimal_lambda, n_rand=100)

Multi-Backend Performance Comparison
====================================

Benchmark different backends for your specific dataset:

.. code-block:: python

    import numpy as np
    import pandas as pd
    import time
    from ridge_inference import ridge
    from ridge_inference.c_bindings import is_c_available
    from ridge_inference.core import NUMBA_AVAILABLE, CUPY_AVAILABLE
    
    # Create test data
    n_genes = 5000
    n_features = 20
    n_samples = 100
    
    X = np.random.randn(n_genes, n_features)
    Y = np.random.randn(n_genes, n_samples)
    
    # Define backends to test
    backends = ['python']
    if NUMBA_AVAILABLE:
        backends.append('numba')
    if is_c_available():
        backends.append('c')
    if CUPY_AVAILABLE:
        backends.append('gpu')
    
    # Define n_rand values to test
    n_rand_values = [0, 10, 100, 1000]
    
    # Initialize results dataframe
    results = pd.DataFrame(index=pd.MultiIndex.from_product([backends, n_rand_values],
                                               names=['Backend', 'n_rand']),
                          columns=['Time (s)', 'Memory (MB)', 'Method Used'])
    
    # Run benchmarks
    for backend in backends:
        for n_rand in n_rand_values:
            # Skip incompatible combinations
            if n_rand == 0 and backend != 'python':
                results.loc[(backend, n_rand), 'Time (s)'] = np.nan
                results.loc[(backend, n_rand), 'Method Used'] = 'N/A'
                continue
                
            print(f"Testing backend={backend}, n_rand={n_rand}")
            
            try:
                # Run with timing
                start_time = time.time()
                result = ridge(X, Y, lambda_=1000, n_rand=n_rand, method=backend)
                elapsed = time.time() - start_time
                
                # Record results
                results.loc[(backend, n_rand), 'Time (s)'] = elapsed
                results.loc[(backend, n_rand), 'Method Used'] = result.get('method_used', backend)
                
                # Record memory if available
                if 'peak_gpu_pool_mb' in result:
                    results.loc[(backend, n_rand), 'Memory (MB)'] = result['peak_gpu_pool_mb']
            
            except Exception as e:
                print(f"Error: {e}")
                results.loc[(backend, n_rand), 'Time (s)'] = np.nan
                results.loc[(backend, n_rand), 'Method Used'] = f"Error: {type(e).__name__}"
    
    # Display results
    print("\nBenchmark Results:")
    print(results.sort_index())
    
    # Plot results if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        
        for backend in backends:
            data = results.loc[backend].dropna(subset=['Time (s)'])
            if not data.empty:
                plt.plot(data.index, data['Time (s)'], marker='o', label=backend)
        
        plt.xlabel('Number of Permutations (n_rand)')
        plt.ylabel('Execution Time (seconds)')
        plt.title('Ridge Regression Performance by Backend')
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.tight_layout()
        plt.savefig('ridge_benchmark.png', dpi=100)
        plt.close()
    except ImportError:
        print("Matplotlib not available for plotting")

Integrating with scikit-learn
=============================

Using RidgeInference with scikit-learn pipelines:

.. code-block:: python

    import numpy as np
    from sklearn.base import BaseEstimator, RegressorMixin
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from ridge_inference import ridge
    
    class RidgeInferenceRegressor(BaseEstimator, RegressorMixin):
        """Wrapper for RidgeInference to fit scikit-learn API"""
        
        def __init__(self, lambda_=1000, method='auto', n_rand=0):
            self.lambda_ = lambda_
            self.method = method
            self.n_rand = n_rand
            self.beta_ = None
            
        def fit(self, X, y):
            # Handle 1D y
            if y.ndim == 1:
                y = y.reshape(-1, 1)
                
            # Ensure X is genes × features, y is genes × samples
            if X.shape[0] != y.shape[0]:
                raise ValueError("X and y must have same number of rows (genes)")
                
            # Run ridge regression
            result = ridge(X, y, lambda_=self.lambda_, n_rand=self.n_rand, 
                           method=self.method)
            
            self.beta_ = result['beta']
            self.se_ = result['se']
            self.zscore_ = result['zscore']
            self.pvalue_ = result['pvalue']
            
            return self
            
        def predict(self, X):
            if self.beta_ is None:
                raise ValueError("Estimator not fitted yet. Call 'fit' before using 'predict'.")
                
            # Make prediction: X @ beta
            y_pred = X @ self.beta_
            
            # Return 1D array if original y was 1D
            if self.beta_.shape[1] == 1:
                return y_pred.flatten()
            return y_pred
    
    # Create synthetic data
    X = np.random.randn(1000, 10)
    y = X @ np.random.randn(10, 1) + 0.1 * np.random.randn(1000, 1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and fit pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', RidgeInferenceRegressor(lambda_=100, method='auto'))
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    train_score = pipeline.score(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)
    
    print(f"R² score (train): {train_score:.4f}")
    print(f"R² score (test): {test_score:.4f}")
