#!/usr/bin/env python
"""
Sparsity-Preserving Preprocessor for Ridge Inference Input.

Loads dense expression (Y - Feather/CSV/etc.) and signature (X - CSV/TSV),
aligns them by common genes, performs column-wise Z-score scaling while
PRESERVING SPARSITY (only scaling non-zero values), converts Y to sparse CSR,
and saves:

1.  Sparse Scaled Y: `<output_dir>/<dataset_name>_scaled_aligned.sparse.npz`
2.  Dense Scaled Aligned X: `<output_dir>/<dataset_name>_scaled_aligned_Xsig.csv`
3.  Sample Names JSON: `<output_dir>/<dataset_name>_scaled_aligned_samples.json`

These outputs are used by `run_ridge_comparison.py --data_format sparse`
and the `ridge-inference-main --sparse-x ...` CLI.
"""

import os
import sys
import gc
import json
import time
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from scipy import sparse as sps
import logging
import warnings

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
logger = logging.getLogger("SparsePreprocessor")

# Helper Functions (copied/adapted from run_ridge_comparison.py)
def load_dataset(dataset_file: Path) -> pd.DataFrame:
    logger.info(f"Loading dense expression data (Y): {dataset_file.name}")
    dataset_file = Path(dataset_file).resolve()
    if not dataset_file.exists(): raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
    df = None; file_suffix = dataset_file.suffix.lower()
    if file_suffix == '.feather':
        try:
            df = pd.read_feather(dataset_file)
            potential_index_cols = ['index', 'Unnamed: 0', 'gene', 'genes', 'gene_id', 'gene_symbol', 'Index']
            found_index = False
            if isinstance(df.index, pd.Index) and (df.index.name in potential_index_cols or not pd.api.types.is_numeric_dtype(df.index)): pass # Already good
            else:
                for col_name in potential_index_cols:
                    if col_name in df.columns and not pd.api.types.is_numeric_dtype(df[col_name]):
                        logger.info(f"Setting '{col_name}' as index."); df = df.set_index(col_name); found_index = True; break
                if not found_index: logger.warning("Could not auto-set gene index for Feather.")
            logger.info(f"Loaded from Feather format.")
        except Exception as e: raise IOError(f"Failed to load Feather {dataset_file}: {e}") from e
    else:
        compression = 'gzip' if dataset_file.name.endswith('.gz') else None; common_seps = ['\t', ',']
        for sep in common_seps:
            try:
                df_try = pd.read_csv(dataset_file, sep=sep, index_col=0, compression=compression, low_memory=False)
                if df_try.shape[1] > 0 and not pd.api.types.is_numeric_dtype(df_try.index): logger.info(f"Loaded text with sep: {repr(sep)}"); df = df_try; break
            except Exception: continue
        if df is None: raise IOError(f"Failed load text {dataset_file}")
    if df is None or df.empty: raise ValueError(f"Loaded DF is None/empty: {dataset_file.name}.")
    if pd.api.types.is_numeric_dtype(df.index): raise ValueError(f"Numeric index for {dataset_file.name}. Need gene IDs.")
    logger.info(f"Loaded dense expression data (Y) shape: {df.shape}")
    return df

def load_signature_matrix(file_path: Path) -> pd.DataFrame:
    logger.info(f"Loading signature matrix (X): {file_path.name}")
    file_path = Path(file_path).resolve()
    if not file_path.exists(): raise FileNotFoundError(f"Signature file not found: {file_path}")
    try:
        compression = 'gzip' if file_path.name.endswith('.gz') else None
        sep = '\t'; sig_matrix = pd.read_csv(file_path, sep=sep, index_col=0, compression=compression)
        if sig_matrix.shape[1] < 2 and compression is None: logger.warning("Tab sep failed, trying comma..."); sig_matrix = pd.read_csv(file_path, sep=',', index_col=0, compression=compression)
        if sig_matrix.shape[1] == 0: raise ValueError("Sig matrix 0 cols.")
        if not isinstance(sig_matrix.index, pd.Index) or pd.api.types.is_numeric_dtype(sig_matrix.index): raise ValueError("Sig matrix needs non-numeric index.")
        logger.info(f"Loaded signature matrix (X) shape: {sig_matrix.shape}")
        return sig_matrix
    except Exception as e: raise IOError(f"Failed load sig matrix {file_path}: {e}") from e

def prepare_data_for_ridge(expr_data: pd.DataFrame, signature_matrix: pd.DataFrame):
    logger.info("Aligning genes...")
    expr_data.index = expr_data.index.astype(str)
    signature_matrix.index = signature_matrix.index.astype(str)
    common_genes = expr_data.index.intersection(signature_matrix.index)
    logger.info(f"Found {len(common_genes)} common genes.")
    if len(common_genes) == 0: raise ValueError("No common genes found.")
    X_prep = signature_matrix.loc[common_genes].copy()
    Y_prep = expr_data.loc[common_genes].copy()
    logger.info(f"Aligned shapes: X={X_prep.shape}, Y={Y_prep.shape}")
    return X_prep, Y_prep

def sparse_preserving_scale(data, axis=0, with_mean=True, with_std=True, eps=1e-8):
    """
    Scales data while preserving sparsity by only scaling non-zero values.
    
    Parameters:
    -----------
    data : ndarray or sparse matrix
        The data to scale
    axis : int, default=0
        Axis along which to scale (0=columns, 1=rows)
    with_mean : bool, default=True
        Center data by subtracting mean (only for non-zero values)
    with_std : bool, default=True
        Scale data to unit variance (only for non-zero values)
    eps : float, default=1e-8
        Small value to avoid division by zero
        
    Returns:
    --------
    scaled_data : same type as input
        Scaled data with preserved sparsity pattern
    """
    logger.debug(f"Sparse-preserving scaling, shape: {data.shape}, axis: {axis}")
    
    # Check if input is sparse
    is_sparse = sps.issparse(data)
    
    # Convert pandas to numpy if needed
    if isinstance(data, pd.DataFrame):
        index = data.index
        columns = data.columns
        data_values = data.values
    else:
        data_values = data
        index = None
        columns = None
    
    # If already sparse, ensure CSR format for row operations or CSC for column operations
    if is_sparse:
        if axis == 0:  # Operating on columns
            data_values = data_values.tocsc()
        else:  # Operating on rows
            data_values = data_values.tocsr()
    
    # Get dimensions
    n_rows, n_cols = data_values.shape
    
    # Initialize arrays for means and standard deviations
    if axis == 0:  # Scale each column
        means = np.zeros(n_cols)
        stds = np.ones(n_cols)
        
        # Process each column
        for j in range(n_cols):
            if is_sparse:
                # Get indices and data for this column
                start_idx = data_values.indptr[j]
                end_idx = data_values.indptr[j+1]
                col_data = data_values.data[start_idx:end_idx]
                
                if len(col_data) > 0:  # If column has non-zero values
                    if with_mean:
                        means[j] = np.mean(col_data)
                    if with_std:
                        stds[j] = max(np.std(col_data), eps)
            else:
                # Get column and find non-zero indices
                col = data_values[:, j]
                nz_mask = col != 0
                nz_values = col[nz_mask]
                
                if len(nz_values) > 0:  # If column has non-zero values
                    if with_mean:
                        means[j] = np.mean(nz_values)
                    if with_std:
                        stds[j] = max(np.std(nz_values), eps)
        
        # Scale the data (only non-zero values)
        if is_sparse:
            scaled_data = data_values.copy()
            
            # Process each column separately
            for j in range(n_cols):
                start_idx = scaled_data.indptr[j]
                end_idx = scaled_data.indptr[j+1]
                
                if with_mean:
                    scaled_data.data[start_idx:end_idx] -= means[j]
                if with_std:
                    scaled_data.data[start_idx:end_idx] /= stds[j]
        else:
            # For dense data, create a copy with zeros preserved
            scaled_data = np.zeros_like(data_values)
            
            for j in range(n_cols):
                col = data_values[:, j]
                nz_mask = col != 0
                
                if with_mean and with_std:
                    scaled_data[nz_mask, j] = (col[nz_mask] - means[j]) / stds[j]
                elif with_mean:
                    scaled_data[nz_mask, j] = col[nz_mask] - means[j]
                elif with_std:
                    scaled_data[nz_mask, j] = col[nz_mask] / stds[j]
                else:
                    scaled_data[:, j] = col  # No transformation
    
    else:  # Scale each row - similar logic but for rows
        means = np.zeros(n_rows)
        stds = np.ones(n_rows)
        
        # Same process for rows (omitted for brevity, would need to be implemented)
        # This would be similar but operate on rows instead of columns
        logger.warning("Row-wise scaling not implemented in this version, falling back to column scaling")
        return sparse_preserving_scale(data.T, axis=0, with_mean=with_mean, with_std=with_std).T
    
    # Replace potential NaN or Inf values with 0
    if is_sparse:
        # NaN/Inf checking for sparse matrix data array
        mask = ~np.isfinite(scaled_data.data)
        if np.any(mask):
            logger.warning(f"Replacing {np.sum(mask)} NaN/Inf values with 0 in sparse matrix")
            scaled_data.data[mask] = 0
    else:
        # NaN/Inf checking for dense array, preserving zeros
        non_zeros = scaled_data != 0
        non_finite = ~np.isfinite(scaled_data) & non_zeros
        if np.any(non_finite):
            logger.warning(f"Replacing {np.sum(non_finite)} NaN/Inf values with 0 in dense matrix")
            scaled_data[non_finite] = 0
    
    # Convert back to DataFrame if input was DataFrame
    if isinstance(data, pd.DataFrame) and not is_sparse:
        scaled_data = pd.DataFrame(scaled_data, index=index, columns=columns)
    
    return scaled_data

def scale_dataframes(X_df: pd.DataFrame, Y_df: pd.DataFrame):
    """Scale dataframes while preserving sparsity patterns"""
    logger.info("Scaling data with sparsity preservation...")
    
    try:
        # Analyze X sparsity before scaling
        X_vals = X_df.values
        X_zeros = np.sum(X_vals == 0)
        X_total = X_vals.size
        X_sparsity = X_zeros / X_total
        logger.info(f"X matrix sparsity before scaling: {X_sparsity:.4f} ({X_zeros}/{X_total} zeros)")
        
        # Scale X preserving zeros
        logger.debug(f"Scaling X ({X_df.shape})...")
        X_scaled_np = sparse_preserving_scale(X_vals, axis=0, with_mean=True, with_std=True)
        X_scaled_df = pd.DataFrame(X_scaled_np, index=X_df.index, columns=X_df.columns)
        
        # Verify X sparsity after scaling
        X_zeros_after = np.sum(X_scaled_np == 0)
        logger.info(f"X matrix sparsity after scaling: {X_zeros_after/X_total:.4f} ({X_zeros_after}/{X_total} zeros)")
        
        del X_vals, X_scaled_np
        gc.collect()

        # Analyze Y sparsity before scaling
        Y_vals = Y_df.values
        Y_zeros = np.sum(Y_vals == 0)
        Y_total = Y_vals.size
        Y_sparsity = Y_zeros / Y_total
        logger.info(f"Y matrix sparsity before scaling: {Y_sparsity:.4f} ({Y_zeros}/{Y_total} zeros)")
        
        # Scale Y preserving zeros
        logger.debug(f"Scaling Y ({Y_df.shape})...")
        Y_scaled_np = sparse_preserving_scale(Y_vals, axis=0, with_mean=True, with_std=True)
        
        # Verify Y sparsity after scaling
        Y_zeros_after = np.sum(Y_scaled_np == 0)
        logger.info(f"Y matrix sparsity after scaling: {Y_zeros_after/Y_total:.4f} ({Y_zeros_after}/{Y_total} zeros)")
        
        del Y_vals
        gc.collect()

        logger.info("Sparsity-preserving scaling complete.")
        return X_scaled_df, Y_scaled_np
        
    except MemoryError as e:
        logger.error(f"MemoryError during scaling: {e}")
        raise
    except Exception as e:
        logger.error(f"Scaling failed: {e}", exc_info=True)
        raise

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess dense data for sparse ridge input with sparsity preservation.')
    parser.add_argument('--dataset_name', type=str, required=True, help='Unique name for dataset output files.')
    parser.add_argument('--dense_y_path', type=str, required=True, help='Path to input dense expression matrix file (Y).')
    parser.add_argument('--sig_matrix_path', type=str, required=True, help='Path to signature matrix file (X).')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save preprocessed files.')
    return parser.parse_args()

def main():
    args = parse_args()
    start_time = time.time()
    output_dir = Path(args.output_dir).resolve()
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    sparse_y_out_path = output_dir / f"{args.dataset_name}_scaled_aligned.sparse.npz"
    dense_x_out_path = output_dir / f"{args.dataset_name}_scaled_aligned_Xsig.csv"
    samples_out_path = output_dir / f"{args.dataset_name}_scaled_aligned_samples.json"
    
    try:
        y_dense_df = load_dataset(Path(args.dense_y_path))
        x_sig_df = load_signature_matrix(Path(args.sig_matrix_path))
        
        # Save sample names for future use
        sample_names = y_dense_df.columns.astype(str).tolist()
        logger.info(f"Extracted {len(sample_names)} sample names from Y matrix")
        
        x_aligned_df, y_aligned_df = prepare_data_for_ridge(y_dense_df, x_sig_df)
        del y_dense_df, x_sig_df
        gc.collect()
        
        # Scale while preserving sparsity
        x_scaled_df, y_scaled_np = scale_dataframes(x_aligned_df, y_aligned_df)
        del x_aligned_df, y_aligned_df
        gc.collect()

        logger.info(f"Converting scaled Y (shape {y_scaled_np.shape}) to sparse CSR...")
        y_sparse_scaled = sps.csr_matrix(y_scaled_np)
        logger.info(f"Sparse conversion complete. NNZ: {y_sparse_scaled.nnz}, Density: {y_sparse_scaled.nnz / np.prod(y_sparse_scaled.shape):.4f}")
        del y_scaled_np
        gc.collect()

        logger.info(f"Saving sparse scaled Y to: {sparse_y_out_path}")
        sps.save_npz(sparse_y_out_path, y_sparse_scaled, compressed=True)
        
        logger.info(f"Saving dense scaled aligned X to: {dense_x_out_path}")
        x_scaled_df.to_csv(dense_x_out_path, index=True, header=True)
        
        logger.info(f"Saving sample names to: {samples_out_path}")
        with open(samples_out_path, 'w', encoding='utf-8') as f:
            json.dump(sample_names, f)

        logger.info("Preprocessing and saving complete.")
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}", exc_info=True)
        return 1

    logger.info(f"Total preprocessing time: {time.time() - start_time:.2f} seconds.")
    return 0

if __name__ == "__main__":
    sys.exit(main())