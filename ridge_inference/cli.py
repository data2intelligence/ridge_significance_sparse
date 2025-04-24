# ridge_inference/cli.py

import argparse
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import time
import gc
import json
import os
import psutil
import traceback
import platform
import socket
from datetime import datetime
from scipy import sparse as sps

# Imports from within the package
from .loaders import load_matrix_data, load_dataset, load_signature_matrix
# Import the main ridge function and availability flags FROM __init__
from . import (
    ridge,
    secact_inference,
    MKL_AVAILABLE,
    GSL_CYTHON_AVAILABLE,
    NUMBA_AVAILABLE,
    CUPY_AVAILABLE,
    PYTHON_AVAILABLE,
    list_available_backends, # Use the function from the package level
    __version__
)
# Check if anndata is available for h5ad support
try: from .loaders import anndata
except ImportError: anndata = None

# Setup Logging
# (Keep setup_logging, log_system_info, get_memory_usage,
#  log_dataframe_stats, log_matrix_stats as before)
main_cli_logger = logging.getLogger("ridge-main-cli")
secact_cli_logger = logging.getLogger("secact-cli")

def setup_logging(verbosity=0):
    """Configure logging with consistent formatting and output levels"""
    root_logger = logging.getLogger('ridge_inference')
    log_format = '%(asctime)s [%(levelname)s] %(name)s - %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')
    console = logging.StreamHandler(sys.stdout) # Ensure output to stdout
    console.setFormatter(formatter)
    if verbosity == 0: level = logging.WARNING
    elif verbosity == 1: level = logging.INFO
    else: level = logging.DEBUG
    root_logger.setLevel(level)
    # Ensure only one handler is added if called multiple times
    if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
        root_logger.addHandler(console)
    return root_logger # Return root logger for potential file handler addition

def log_system_info(logger):
    """Log detailed system information"""
    logger.info("-" * 40 + " SYSTEM INFO " + "-" * 40)
    logger.info(f"ridge-inference version: {__version__}")
    logger.info(f"Hostname: {socket.gethostname()}")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python: {platform.python_version()}")
    try: logger.info(f"CPU Count: {os.cpu_count()}")
    except NotImplementedError: logger.info("CPU Count: Not Available")
    logger.info(f"CPU Info: {platform.processor()}")
    for var in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS']:
        logger.info(f"ENV: {var}={os.environ.get(var, 'Not set')}")
    try: total_mem = psutil.virtual_memory().total / (1024**3); logger.info(f"Total System Memory: {total_mem:.2f} GB")
    except Exception: logger.info("Total System Memory: Error querying")
    logger.info("-" * 90)

def get_memory_usage():
    """Get current memory usage of the process"""
    try: return psutil.Process(os.getpid()).memory_info().rss / (1024**2) # MB
    except Exception: return -1.0 # Indicate error

def log_dataframe_stats(df, logger, name):
    """Log detailed statistics about a DataFrame"""
    mem_mb = df.memory_usage(deep=True).sum() / (1024**2)
    logger.debug(f"{name} shape: {df.shape}, memory: {mem_mb:.2f} MB")
    logger.debug(f"{name} index type: {df.index.dtype}, columns sample: {list(df.columns[:5])}...")
    logger.debug(f"{name} dtypes: {df.dtypes.value_counts().to_dict()}")

def log_matrix_stats(matrix, logger, name):
    """Log detailed statistics about a matrix"""
    if sps.issparse(matrix):
        mem_mb = matrix.data.nbytes / (1024**2)
        density = matrix.nnz / (matrix.shape[0] * matrix.shape[1]) if matrix.shape[0]*matrix.shape[1] > 0 else 0
        logger.debug(f"{name} sparse: {type(matrix).__name__}, shape: {matrix.shape}, density: {density:.4f}, memory: {mem_mb:.2f} MB")
    else:
        mem_mb = matrix.nbytes / (1024**2)
        logger.debug(f"{name} dense: {type(matrix).__name__}, shape: {matrix.shape}, dtype: {matrix.dtype}, memory: {mem_mb:.2f} MB")


# --- Main Logic for ridge-inference-main ---
def run_main():
    parser = argparse.ArgumentParser(description="General Ridge Regression using ridge-inference package.")
    parser.add_argument("-x", "--xmatrix", required=True, help="Path to input feature matrix X (genes x features).")
    parser.add_argument("-y", "--ymatrix", required=True, help="Path to input target matrix Y (genes x samples).")
    parser.add_argument("-o", "--outdir", required=True, help="Output directory.")
    parser.add_argument("-l", "--lambda", type=float, default=5e5, dest='lambda_val', help="Ridge lambda (default: 5e5)")
    parser.add_argument("-n", "--nrand", type=int, default=1000, help="Permutations (default: 1000). 0=t-test.")
    # Add method argument to allow user selection
    parser.add_argument("-m", "--method", type=str, default="auto", choices=['auto', 'python', 'mkl', 'gsl_cython', 'numba', 'gpu'], help="Backend method (default: auto)")
    parser.add_argument("-v", "--verbose", action='count', default=0, help="Increase verbosity (-v, -vv, -vvv).")
    args = parser.parse_args()

    logger = setup_logging(args.verbose)
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    if args.verbose >= 2: # Log to file only if very verbose
        log_file = outdir / f"ridge_run_{datetime.now():%Y%m%d_%H%M%S}.log"
        file_handler = logging.FileHandler(log_file); file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s - %(message)s'))
        logging.getLogger('ridge_inference').addHandler(file_handler)
        main_cli_logger.info(f"Detailed log saved to: {log_file}")

    main_cli_logger.info("="*15 + f" RIDGE INFERENCE RUN ({datetime.now():%Y-%m-%d %H:%M:%S}) " + "="*15)
    log_system_info(main_cli_logger)
    main_cli_logger.info("--- Available Backends ---")
    list_available_backends(verbose=True) # Use the imported function
    main_cli_logger.info("--- Command Line Arguments ---")
    for arg, value in vars(args).items(): main_cli_logger.info(f"  {arg}: {value}")
    main_cli_logger.info(f"--- Initial Memory Usage: {get_memory_usage():.2f} MB ---")
    main_cli_logger.info(f"Output directory: {outdir}")

    start_time = time.time()
    x_load_time, y_load_time, align_time, ridge_time, save_time = 0, 0, 0, 0, 0 # Initialize timers

    try:
        # Load X
        main_cli_logger.info(f"Loading X matrix from: {args.xmatrix}")
        x_path = Path(args.xmatrix); x_size = x_path.stat().st_size/(1024**2) if x_path.exists() else -1
        main_cli_logger.debug(f"X file size: {x_size:.2f} MB")
        x_load_start = time.time(); X_mat, X_genes, X_features = load_matrix_data(args.xmatrix); x_load_time = time.time() - x_load_start
        if X_mat is None: raise ValueError("Failed to load X matrix.")
        main_cli_logger.info(f"Loaded X in {x_load_time:.2f}s: {type(X_mat).__name__}, genes={len(X_genes)}, features={len(X_features)}")
        log_matrix_stats(X_mat, main_cli_logger, "X_mat")

        # Load Y
        main_cli_logger.info(f"Loading Y matrix from: {args.ymatrix}")
        y_path = Path(args.ymatrix); y_size = y_path.stat().st_size/(1024**2) if y_path.exists() else -1
        main_cli_logger.debug(f"Y file size: {y_size:.2f} MB")
        y_load_start = time.time(); Y_mat, Y_genes, Y_samples = load_matrix_data(args.ymatrix); y_load_time = time.time() - y_load_start
        if Y_mat is None: raise ValueError("Failed to load Y matrix.")
        main_cli_logger.info(f"Loaded Y in {y_load_time:.2f}s: {type(Y_mat).__name__}, genes={len(Y_genes)}, samples={len(Y_samples)}")
        log_matrix_stats(Y_mat, main_cli_logger, "Y_mat")

        # Align genes
        main_cli_logger.info("Aligning genes between X and Y...")
        align_start = time.time()
        common_genes = sorted(list(set(X_genes) & set(Y_genes)))
        if not common_genes: raise ValueError("No common genes found between X and Y.")
        main_cli_logger.info(f"Found {len(common_genes)} common genes (X:{len(X_genes)}, Y:{len(Y_genes)})")
        x_gene_map = {gene: idx for idx, gene in enumerate(X_genes)}; y_gene_map = {gene: idx for idx, gene in enumerate(Y_genes)}
        x_indices = [x_gene_map[gene] for gene in common_genes]; y_indices = [y_gene_map[gene] for gene in common_genes]
        # Align matrices (handle sparse/dense)
        X_run = X_mat[x_indices, :] if sps.issparse(X_mat) else X_mat[np.ix_(x_indices)]
        Y_run = Y_mat[y_indices, :] if sps.issparse(Y_mat) else Y_mat[np.ix_(y_indices)]
        align_time = time.time() - align_start
        main_cli_logger.info(f"Aligned matrices in {align_time:.2f}s. Shapes: X={X_run.shape}, Y={Y_run.shape}")
        del X_mat, Y_mat, x_gene_map, y_gene_map, x_indices, y_indices; gc.collect()
        main_cli_logger.debug(f"Memory after alignment: {get_memory_usage():.2f} MB")

        # Run ridge regression (selection happens inside ridge())
        main_cli_logger.info(f"Running ridge regression (λ={args.lambda_val}, n_rand={args.nrand}, method='{args.method}')...")
        ridge_start = time.time()
        results = ridge(X=X_run, Y=Y_run, lambda_=args.lambda_val, n_rand=args.nrand, method=args.method, verbose=args.verbose)
        ridge_time = time.time() - ridge_start
        main_cli_logger.info(f"Ridge regression completed in {ridge_time:.2f}s using method: {results.get('method_used')}")

        # Analyze results
        main_cli_logger.info("--- Results Analysis ---")
        for key in ['beta', 'se', 'zscore', 'pvalue']:
            if key in results and isinstance(results[key], np.ndarray):
                nan_count = np.isnan(results[key]).sum(); inf_count = np.isinf(results[key]).sum()
                main_cli_logger.debug(f"{key} shape: {results[key].shape}, dtype: {results[key].dtype}, NaNs: {nan_count}, Infs: {inf_count}")
                if nan_count>0: main_cli_logger.warning(f"{key} contains {nan_count} NaNs")
                if inf_count>0: main_cli_logger.warning(f"{key} contains {inf_count} Infs")
            else: main_cli_logger.warning(f"Result key '{key}' missing or not ndarray")

        # Save results
        main_cli_logger.info(f"Saving results to {outdir}...")
        save_start = time.time()
        beta_df = pd.DataFrame(results['beta'], index=X_features, columns=Y_samples)
        se_df = pd.DataFrame(results['se'], index=X_features, columns=Y_samples)
        zscore_df = pd.DataFrame(results['zscore'], index=X_features, columns=Y_samples)
        pvalue_df = pd.DataFrame(results['pvalue'], index=X_features, columns=Y_samples)
        beta_df.to_csv(outdir / "beta.csv.gz", compression="gzip")
        se_df.to_csv(outdir / "se.csv.gz", compression="gzip")
        zscore_df.to_csv(outdir / "zscore.csv.gz", compression="gzip")
        pvalue_df.to_csv(outdir / "pvalue.csv.gz", compression="gzip")
        main_cli_logger.debug("Output files sizes (MB):")
        for f in ["beta.csv.gz", "se.csv.gz", "zscore.csv.gz", "pvalue.csv.gz"]:
            f_path = outdir/f; f_size = f_path.stat().st_size/(1024**2) if f_path.exists() else -1
            main_cli_logger.debug(f"  {f}: {f_size:.2f}")

        # Save metadata
        metadata = { "command": "ridge-inference-main", "args": vars(args), "version": __version__,
                     "timestamp": datetime.now().isoformat(), "hostname": socket.gethostname(),
                     "input_x": str(args.xmatrix), "input_y": str(args.ymatrix), "output_dir": str(outdir),
                     "n_genes_common": len(common_genes), "n_features": X_run.shape[1], "n_samples": Y_run.shape[1],
                     "method_used": results.get('method_used'), "total_time_sec": time.time() - start_time,
                     "load_x_time_sec": x_load_time, "load_y_time_sec": y_load_time, "align_time_sec": align_time,
                     "ridge_time_sec": ridge_time, "peak_memory_mb": get_memory_usage(),
                     "peak_gpu_mem_mb": results.get('peak_gpu_pool_mb'), "df_ttest": results.get('df') }
        with open(outdir / "run_metadata.json", 'w') as f: json.dump(metadata, f, indent=2, default=str)
        save_time = time.time() - save_start
        main_cli_logger.info(f"Results saved in {save_time:.2f}s")

    except Exception as e:
        main_cli_logger.error("--- ERROR ---")
        main_cli_logger.error(f"{type(e).__name__}: {e}", exc_info=(args.verbose >= 1))
        if args.verbose >= 2: # Print full traceback if very verbose
             tb_lines = traceback.format_exception(type(e), e, e.__traceback__)
             main_cli_logger.error("Stack trace:\n" + "".join(tb_lines))
        sys.exit(1)

    total_time = time.time() - start_time
    main_cli_logger.info("--- SUMMARY ---")
    main_cli_logger.info(f"Total execution time: {total_time:.2f}s")
    main_cli_logger.info(f"Final memory usage: {get_memory_usage():.2f} MB")
    phases = [("Load X", x_load_time), ("Load Y", y_load_time), ("Align", align_time), ("Ridge", ridge_time), ("Save", save_time)]
    main_cli_logger.info("Time breakdown:")
    for name, t in phases: main_cli_logger.info(f"  {name}: {t:.2f}s ({(t/total_time*100) if total_time else 0:.1f}%)")
    main_cli_logger.info("="*25 + " RUN COMPLETE " + "="*25)
    sys.exit(0)


# --- Main Logic for ridge-inference-secact ---
def run_secact():
    parser = argparse.ArgumentParser(description="SecAct Inference using ridge-inference package.")
    parser.add_argument("-y", "--ymatrix", required=True, help="Path to expression matrix Y (genes x samples).")
    parser.add_argument("-s", "--signature", default="SecAct", help="Signature matrix name or path (default: SecAct)")
    parser.add_argument("-o", "--outdir", required=True, help="Output directory.")
    parser.add_argument("-l", "--lambda", type=float, default=5e5, dest='lambda_val', help="Ridge lambda (default: 5e5)")
    parser.add_argument("-n", "--nrand", type=int, default=1000, help="Permutations (default: 1000). 0=t-test.")
    parser.add_argument("--scale", default=None, choices=[None, 'column', 'global'], help="Internal scaling method (default: None)")
    parser.add_argument("--add_background", action='store_true', help="Add background column to signature.")
    # Add method argument
    parser.add_argument("-m", "--method", type=str, default="auto", choices=['auto', 'python', 'mkl', 'gsl_cython', 'numba', 'gpu'], help="Backend method (default: auto)")
    parser.add_argument("--batch_threshold", type=int, default=50000, help="Sample threshold for batching (default: 50000)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size if batching (default: auto)")
    parser.add_argument("-v", "--verbose", action='count', default=0, help="Increase verbosity (-v, -vv, -vvv).")
    args = parser.parse_args()

    logger = setup_logging(args.verbose)
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    if args.verbose >= 2: # Log to file only if very verbose
        log_file = outdir / f"secact_run_{datetime.now():%Y%m%d_%H%M%S}.log"
        file_handler = logging.FileHandler(log_file); file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s - %(message)s'))
        logging.getLogger('ridge_inference').addHandler(file_handler)
        secact_cli_logger.info(f"Detailed log saved to: {log_file}")

    secact_cli_logger.info("="*15 + f" SECACT INFERENCE RUN ({datetime.now():%Y-%m-%d %H:%M:%S}) " + "="*15)
    log_system_info(secact_cli_logger)
    secact_cli_logger.info("--- Available Backends ---")
    list_available_backends(verbose=True)
    secact_cli_logger.info("--- Command Line Arguments ---")
    for arg, value in vars(args).items(): secact_cli_logger.info(f"  {arg}: {value}")
    secact_cli_logger.info(f"--- Initial Memory Usage: {get_memory_usage():.2f} MB ---")
    secact_cli_logger.info(f"Output directory: {outdir}")

    start_time = time.time()
    data_load_time, sig_load_time, inference_time, save_time = 0, 0, 0, 0 # Initialize timers

    try:
        # Load data
        secact_cli_logger.info(f"Loading expression data from: {args.ymatrix}")
        y_path = Path(args.ymatrix); y_size = y_path.stat().st_size/(1024**2) if y_path.exists() else -1
        secact_cli_logger.debug(f"Expression file size: {y_size:.2f} MB")
        data_load_start = time.time(); y_df = load_dataset(y_path); data_load_time = time.time() - data_load_start
        secact_cli_logger.info(f"Loaded expression data in {data_load_time:.2f}s")
        log_dataframe_stats(y_df, secact_cli_logger, "Expression data")

        # Load signature (don't load here, pass name/path to secact_inference)
        sig_arg = args.signature
        secact_cli_logger.info(f"Using signature: {sig_arg}")

        # Run secact_inference (handles signature loading, alignment, ridge call)
        secact_cli_logger.info(f"Running SecAct inference (λ={args.lambda_val}, n_rand={args.nrand}, method='{args.method}', scale={args.scale}, add_background={args.add_background})...")
        inference_start = time.time()
        results = secact_inference(
            expr_data=y_df, sig_matrix=sig_arg, lambda_val=args.lambda_val,
            n_rand=args.nrand, method=args.method, add_background=args.add_background,
            scale_method=args.scale, batch_size=args.batch_size,
            batch_threshold=args.batch_threshold, verbose=args.verbose
        )
        inference_time = time.time() - inference_start
        secact_cli_logger.info(f"SecAct inference completed in {inference_time:.2f}s")
        secact_cli_logger.info(f"Method used: {results.get('method')}")
        secact_cli_logger.info(f"Batching used: {results.get('batched', False)}")
        if results.get('batched'): secact_cli_logger.info(f"Actual batch size: {results.get('batch_size')}")

        # Analyze results
        secact_cli_logger.info("--- Results Analysis ---")
        for key in ['beta', 'se', 'zscore', 'pvalue']:
            if key in results and isinstance(results[key], pd.DataFrame):
                nan_count = results[key].isna().sum().sum(); inf_count = results[key].isin([np.inf, -np.inf]).sum().sum()
                secact_cli_logger.debug(f"{key} shape: {results[key].shape}, NaNs: {nan_count}, Infs: {inf_count}")
                if nan_count > 0: secact_cli_logger.warning(f"{key} contains {nan_count} NaNs")
                if inf_count > 0: secact_cli_logger.warning(f"{key} contains {inf_count} Infs")
            else: secact_cli_logger.warning(f"Result key '{key}' missing or not DataFrame")

        # Save results
        secact_cli_logger.info(f"Saving results to {outdir}...")
        save_start = time.time()
        results['beta'].to_csv(outdir / "beta.csv.gz", compression="gzip")
        results['se'].to_csv(outdir / "se.csv.gz", compression="gzip")
        results['zscore'].to_csv(outdir / "zscore.csv.gz", compression="gzip")
        results['pvalue'].to_csv(outdir / "pvalue.csv.gz", compression="gzip")
        secact_cli_logger.debug("Output files sizes (MB):")
        for f in ["beta.csv.gz", "se.csv.gz", "zscore.csv.gz", "pvalue.csv.gz"]:
            f_path = outdir/f; f_size = f_path.stat().st_size/(1024**2) if f_path.exists() else -1
            secact_cli_logger.debug(f"  {f}: {f_size:.2f}")

        # Save metadata (extract relevant info from results dict)
        metadata = { "command": "ridge-inference-secact", "args": vars(args), "version": __version__,
                     "timestamp": datetime.now().isoformat(), "hostname": socket.gethostname(),
                     "input_y": str(args.ymatrix), "signature": str(args.signature), "output_dir": str(outdir),
                     "method_used": results.get('method'), "total_time_sec": time.time() - start_time,
                     "load_data_time_sec": data_load_time, "inference_time_sec": inference_time,
                     "ridge_internal_time_sec": results.get('internal_ridge_time'), # Time from ridge() itself
                     "batched": results.get('batched'), "batch_size": results.get('batch_size'),
                     "peak_memory_mb": get_memory_usage(), "peak_gpu_mem_mb": results.get('peak_gpu_pool_mb'),
                     "df_ttest": results.get('df')}
        with open(outdir / "run_metadata.json", 'w') as f: json.dump(metadata, f, indent=2, default=str)
        save_time = time.time() - save_start
        secact_cli_logger.info(f"Results saved in {save_time:.2f}s")

    except Exception as e:
        secact_cli_logger.error("--- ERROR ---")
        secact_cli_logger.error(f"{type(e).__name__}: {e}", exc_info=(args.verbose >= 1))
        if args.verbose >= 2: # Print full traceback if very verbose
             tb_lines = traceback.format_exception(type(e), e, e.__traceback__)
             secact_cli_logger.error("Stack trace:\n" + "".join(tb_lines))
        sys.exit(1)

    total_time = time.time() - start_time
    secact_cli_logger.info("--- SUMMARY ---")
    secact_cli_logger.info(f"Total execution time: {total_time:.2f}s")
    secact_cli_logger.info(f"Final memory usage: {get_memory_usage():.2f} MB")
    phases = [("Load Data", data_load_time), ("Inference", inference_time), ("Save", save_time)]
    secact_cli_logger.info("Time breakdown:")
    for name, t in phases: secact_cli_logger.info(f"  {name}: {t:.2f}s ({(t/total_time*100) if total_time else 0:.1f}%)")
    secact_cli_logger.info("="*25 + " RUN COMPLETE " + "="*25)
    sys.exit(0)


if __name__ == '__main__':
    # This allows running directly, but entry points in setup.py are preferred
    if sys.argv[0].endswith('cli.py') and len(sys.argv) > 1:
        # Poor man's CLI selection based on script name idea - not robust
        # Better to use entry points from setup.py
        if 'secact' in sys.argv[1]: # Check if first arg contains 'secact'
             run_secact()
        else:
             run_main()
    else:
         print("Use 'ridge-inference-main' or 'ridge-inference-secact' command-line tools.")