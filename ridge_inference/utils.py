"""
Utility functions for the Ridge Inference package.

This module contains helper functions for data loading, preprocessing,
visualization, and other common operations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import sparse as sps # Use alias
import warnings
import os
import logging

logger = logging.getLogger(__name__)


def is_sparse_beneficial(X, Y, threshold=0.1, min_size=1e6):
    """
    DEPRECATED/INTERNAL USE: Use specific checks like _is_Y_sparse_beneficial_for_cpu.
    Original function to determine if sparse matrix representation might be beneficial
    based on density and size.

    Parameters
    ----------
    X : numpy.ndarray or scipy.sparse matrix
        Feature matrix
    Y : numpy.ndarray or scipy.sparse matrix
        Target matrix
    threshold : float
        Density threshold below which sparse is preferred
    min_size : float
        Minimum matrix size (elements) for sparse consideration

    Returns
    -------
    bool
        Whether sparse representation might be beneficial
    """
    warnings.warn("is_sparse_beneficial is deprecated. Use format-specific checks.", DeprecationWarning)
    # Check if matrices are already sparse
    is_X_sparse = sps.issparse(X)
    is_Y_sparse = sps.issparse(Y)
    if is_X_sparse or is_Y_sparse:
        return True # Already sparse, benefit is assumed

    # Calculate density only if dense numpy arrays
    avg_density = 1.0 # Default to dense
    X_size = 0
    Y_size = 0

    try:
        if isinstance(X, np.ndarray) and X.size > 0:
            X_density = np.count_nonzero(X) / X.size
            X_size = X.size
        else: X_density = 1.0

        if isinstance(Y, np.ndarray) and Y.size > 0:
            Y_density = np.count_nonzero(Y) / Y.size
            Y_size = Y.size
        else: Y_density = 1.0

        # Average density only makes sense if both are dense numpy
        if isinstance(X, np.ndarray) and isinstance(Y, np.ndarray):
             avg_density = (X_density * X_size + Y_density * Y_size) / (X_size + Y_size) if (X_size + Y_size) > 0 else 1.0
        else:
             # If one is sparse already or not numpy, hard to say, default no benefit unless already sparse
             avg_density = 1.0

    except MemoryError:
         logger.warning("MemoryError calculating density in is_sparse_beneficial.")
         return False # Assume not beneficial if density check fails
    except Exception as e:
         logger.warning(f"Error calculating density in is_sparse_beneficial: {e}")
         return False # Assume not beneficial on error

    # Return decision based on average density and size
    return (avg_density < threshold and (X_size > min_size or Y_size > min_size))


def convert_to_sparse(X, Y, format='csr'):
    """
    Convert dense NumPy arrays to sparse matrices if they are not already sparse.

    Parameters
    ----------
    X : numpy.ndarray or scipy.sparse matrix
        Feature matrix
    Y : numpy.ndarray or scipy.sparse matrix
        Target matrix
    format : str
        Target sparse format ('csr', 'csc', 'coo', etc.)

    Returns
    -------
    tuple
        (X_sparse, Y_sparse) as sparse matrices in the specified format.
    """
    if not isinstance(format, str) or format not in ['csr', 'csc', 'coo', 'lil', 'dok', 'dia', 'bsr']:
        raise ValueError(f"Invalid sparse format: {format}")

    format_func_map = {
        'csr': sps.csr_matrix, 'csc': sps.csc_matrix, 'coo': sps.coo_matrix,
        'lil': sps.lil_matrix, 'dok': sps.dok_matrix, 'dia': sps.dia_matrix,
        'bsr': sps.bsr_matrix
    }
    convert_func = format_func_map[format]
    convert_method = f'to{format}'

    if sps.issparse(X):
        # If already sparse, convert to the desired format if necessary
        if X.format != format:
            X_sparse = getattr(X, convert_method)()
        else:
            X_sparse = X
    elif isinstance(X, np.ndarray):
        X_sparse = convert_func(X)
    else:
        raise TypeError(f"Input X must be a NumPy array or SciPy sparse matrix, got {type(X)}")

    if sps.issparse(Y):
        if Y.format != format:
            Y_sparse = getattr(Y, convert_method)()
        else:
            Y_sparse = Y
    elif isinstance(Y, np.ndarray):
        Y_sparse = convert_func(Y)
    else:
        raise TypeError(f"Input Y must be a NumPy array or SciPy sparse matrix, got {type(Y)}")

    return X_sparse, Y_sparse


def find_overlapping_genes(expr, sig_matrix):
    """
    Find overlapping genes (index) between expression data and signature matrix.

    Parameters
    ----------
    expr : pandas.DataFrame
        Expression data with genes as index.
    sig_matrix : pandas.DataFrame
        Signature matrix with genes as index.

    Returns
    -------
    pandas.Index
        Index object containing the overlapping gene identifiers.
    """
    if not isinstance(expr, pd.DataFrame) or not isinstance(expr.index, pd.Index):
        raise TypeError("expr must be a pandas DataFrame with an index.")
    if not isinstance(sig_matrix, pd.DataFrame) or not isinstance(sig_matrix.index, pd.Index):
        raise TypeError("sig_matrix must be a pandas DataFrame with an index.")

    # Ensure indices are of compatible types (e.g., string) for robust intersection
    try:
        expr_index = expr.index.astype(str)
        sig_index = sig_matrix.index.astype(str)
    except Exception as e:
        logger.warning(f"Could not ensure string indices for overlap check: {e}. Proceeding with original types.")
        expr_index = expr.index
        sig_index = sig_matrix.index

    # Use pandas Index intersection method for efficiency
    overlapping = expr_index.intersection(sig_index)
    n_overlap = len(overlapping)

    if n_overlap < 10:
        warnings.warn(f"Warning: Only {n_overlap} genes overlap between expression data "
                      f"(n={len(expr_index)}) and signature matrix (n={len(sig_index)}).", RuntimeWarning)
    elif n_overlap == 0:
        logger.error("No overlapping genes found between expression data and signature matrix.")
        # Return empty index, let caller handle it
    else:
        logger.debug(f"Found {n_overlap} overlapping genes.")

    return overlapping


def scale_dataframe_columns(df, epsilon=1e-8):
    """
    Performs column-wise standard scaling (z-score) on a DataFrame.

    Sets columns with near-zero standard deviation to zero.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    epsilon : float
        Threshold below which standard deviation is considered zero.

    Returns
    -------
    pandas.DataFrame
        Column-scaled DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    if df.empty:
         logger.warning("Input DataFrame for scaling is empty.")
         return df.copy() # Return an empty copy

    # Ensure numeric types for scaling
    df_numeric = df.select_dtypes(include=np.number)
    if df_numeric.shape[1] != df.shape[1]:
        logger.warning(f"Non-numeric columns detected in DataFrame passed to scale_dataframe_columns. Scaling only numeric columns.")
        if df_numeric.empty:
             logger.warning("No numeric columns found for scaling.")
             return df.copy() # Return original if no numeric columns

    df_scaled = df.copy() # Start with a copy of the original

    try:
        df_mean = df_numeric.mean(axis=0)
        df_std = df_numeric.std(axis=0)

        for col in df_numeric.columns:
             std_val = df_std[col]
             if pd.isna(std_val) or std_val < epsilon:
                  # Set column to 0 if std dev is NaN or near zero
                  df_scaled[col] = 0.0
                  if pd.isna(std_val):
                       logger.warning(f"Column '{col}' has NaN std dev, setting scaled values to 0.")
                  else:
                       logger.warning(f"Column '{col}' has std dev ({std_val:.2e}) near zero (epsilon={epsilon:.1e}), setting scaled values to 0.")
             else:
                  # Perform scaling only on this numeric column in the copied DataFrame
                  df_scaled[col] = (df_numeric[col] - df_mean[col]) / std_val

        # Fill any remaining NaNs that might occur (e.g., if original data had NaNs)
        # Only fill NaNs in the columns that were actually scaled
        cols_scaled = df_numeric.columns
        df_scaled[cols_scaled] = df_scaled[cols_scaled].fillna(0)

    except Exception as e:
         logger.error(f"Error during column-wise scaling: {e}", exc_info=True)
         # Return the unscaled copy in case of error? Or raise? Raising is safer.
         raise RuntimeError("Failed to perform column-wise scaling.") from e

    return df_scaled


def visualize_activity(activity_result, top_n=20, pvalue_threshold=0.05, figsize=(10, 8)):
    """
    Creates a basic visualization of SecAct activity results (beta values).

    Shows top proteins by absolute beta value, colored by significance.
    Handles both single-sample/comparison (bar plot) and multi-sample (heatmap) cases.

    Parameters
    ----------
    activity_result : dict
        Result dictionary from `secact_inference` or `secact_activity_inference`.
        Must contain 'beta' and 'pvalue' DataFrames.
    top_n : int
        Number of top proteins (features) to visualize based on absolute beta.
    pvalue_threshold : float
        P-value threshold for determining significance (used for coloring/markers).
    figsize : tuple
        Figure size for the plot.

    Returns
    -------
    matplotlib.figure.Figure or None
        The generated matplotlib Figure object, or None if plotting libraries
        are unavailable or an error occurs.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("Matplotlib and/or Seaborn not installed. Cannot create visualization.")
        return None

    if not isinstance(activity_result, dict):
        logger.error("Visualization error: activity_result must be a dictionary.")
        return None
    if 'beta' not in activity_result or 'pvalue' not in activity_result:
        logger.error("Visualization error: activity_result dictionary missing 'beta' or 'pvalue' key.")
        return None
    if not isinstance(activity_result['beta'], pd.DataFrame) or not isinstance(activity_result['pvalue'], pd.DataFrame):
        logger.error("Visualization error: 'beta' and 'pvalue' must be pandas DataFrames.")
        return None

    beta_df = activity_result["beta"]
    pvalue_df = activity_result["pvalue"]

    if beta_df.empty or pvalue_df.empty:
        logger.warning("Visualization warning: Beta or P-value DataFrame is empty. Cannot plot.")
        return None
    if beta_df.shape != pvalue_df.shape:
        logger.error(f"Visualization error: Shape mismatch between beta {beta_df.shape} and pvalue {pvalue_df.shape}.")
        return None

    n_features, n_samples = beta_df.shape
    actual_top_n = min(top_n, n_features) # Ensure top_n doesn't exceed available features

    plt.figure(figsize=figsize)
    fig = plt.gcf() # Get current figure

    if n_samples == 1:
        # --- Single Sample / Comparison: Bar Plot ---
        logger.debug(f"Visualizing single sample/comparison (top {actual_top_n}).")
        # Calculate absolute beta values for sorting
        abs_beta = beta_df.iloc[:, 0].abs()
        # Sort by absolute beta and get top N indices
        sorted_indices = abs_beta.sort_values(ascending=False).index
        top_indices = sorted_indices[:actual_top_n]

        # Select data for top N proteins
        plot_data = beta_df.loc[top_indices].iloc[:, 0]
        plot_pvals = pvalue_df.loc[top_indices].iloc[:, 0]
        significant = plot_pvals < pvalue_threshold

        # Create DataFrame for seaborn plotting (sorted by value for better viz)
        plot_df = pd.DataFrame({
            'Protein': plot_data.index,
            'Activity (beta)': plot_data.values,
            'Significant': significant.values
        }).sort_values('Activity (beta)', ascending=False) # Sort for plotting order

        # Create bar plot
        colors = ['#1f77b4' if sig else 'lightgrey' for sig in plot_df['Significant']] # Blue if significant
        ax = sns.barplot(x='Activity (beta)', y='Protein', data=plot_df, palette=colors, orient='h')

        # Add significance markers (*)
        # Adjust marker position based on bar value (positive/negative)
        for i, row in plot_df.iterrows():
            if row['Significant']:
                x_pos = row['Activity (beta)']
                offset = abs(x_pos) * 0.05 # Small offset from the bar end
                text_x = x_pos + offset if x_pos >= 0 else x_pos - offset
                ha = 'left' if x_pos >= 0 else 'right'
                ax.text(text_x, i, '*', ha=ha, va='center', color='red', fontsize=14, weight='bold')

        plt.title(f'Top {actual_top_n} Protein Activities (beta)\n(* p < {pvalue_threshold})')
        plt.xlabel("Inferred Activity (beta)")
        plt.ylabel("") # Remove Y label, proteins are clear
        plt.axvline(0, color='black', linewidth=0.8, linestyle='--') # Line at zero
        plt.tight_layout()

    else:
        # --- Multiple Samples: Heatmap ---
        logger.debug(f"Visualizing multiple samples ({n_samples}) heatmap (top {actual_top_n}).")
        # Find top N proteins based on mean absolute beta across samples
        mean_abs_beta = beta_df.abs().mean(axis=1)
        sorted_indices = mean_abs_beta.sort_values(ascending=False).index
        top_indices = sorted_indices[:actual_top_n]

        # Select data for top N proteins
        heatmap_data = beta_df.loc[top_indices]

        # Determine appropriate color limits (center around 0)
        lim = max(abs(heatmap_data.min().min()), abs(heatmap_data.max().max()))
        lim = max(lim, 1e-9) # Ensure limit is not zero

        # Create heatmap
        sns.heatmap(heatmap_data, cmap='coolwarm', center=0, vmin=-lim, vmax=lim,
                   linewidths=0.5, linecolor='lightgray', cbar_kws={'label': 'Activity (beta)'})
        plt.title(f'Top {actual_top_n} Protein Activities (beta) across Samples')
        plt.xlabel("Samples")
        plt.ylabel("Proteins")
        plt.yticks(rotation=0) # Keep protein names horizontal
        plt.xticks(rotation=90) # Rotate sample names if many
        plt.tight_layout()

    return fig