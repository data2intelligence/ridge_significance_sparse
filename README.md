# Ridge Inference

A high-performance Python package for ridge regression with significance testing, providing multiple optimized backends and specialized functionality for bioinformatics applications.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/ridge-inference.svg)](https://badge.fury.io/py/ridge-inference)

## Features

-   **Multiple Optimized Backends**: Choose between Python (NumPy/SciPy), Numba-accelerated, C implementations (via Cython using GSL or Intel MKL), or GPU (CuPy).
-   **Comprehensive Statistical Testing**: Permutation tests and t-tests (for dense X with GSL/Python backends) with robust p-value calculation.
-   **Sparse Matrix Support**: Efficiently handles sparse input matrices (Y for most backends, X only for Python backend).
-   **Batch Processing**: Process large datasets (many samples) in chunks to manage memory constraints (Python/GPU backends).
-   **GPU Acceleration**: Leverage NVIDIA GPUs via CuPy for significant performance improvements on large dense data.
-   **SecAct Inference**: Specialized function to compute signaling activities of secreted proteins from gene expression data.
-   **Logistic Regression**: Includes logistic regression with optional Firth correction for separation issues.
-   **Command-Line Interface**: Ready-to-use CLI tools (`ridge-inference-main`, `ridge-inference-secact`) for easy integration into bioinformatics workflows.
-   **Flexible Data Loading**: Supports various file formats (Feather, CSV/TSV, AnnData, Zarr, NPZ).

## Installation

### Prerequisites

-   Python 3.8+
-   A C compiler (like GCC or Clang) for building extensions.
-   Core dependencies (installed automatically): NumPy, SciPy, Pandas, psutil, tqdm.

### Basic Installation

Installs the core package with the Python backend:

```bash
pip install ridge-inference
```

### Installation with Optional Accelerated Backends

To enable specific accelerated backends, you need to install their system prerequisites first, then install the package using the corresponding "extra".

#### Intel MKL Backend (Recommended for CPU Performance):

**Prerequisite**: Install Intel MKL Development Libraries (headers and libraries). This can be done via:

- **Conda (Recommended)**: `conda install -c conda-forge mkl-devel`
- **Intel Installers**: Download and install from the Intel oneAPI HPC Toolkit.
- **System Package Manager** (Less common): e.g., `apt install intel-mkl-devel` (availability varies).

Ensure the `MKLROOT` environment variable is set, or that MKL is installed in your Conda environment.

Install with MKL extra:

```bash
pip install ridge-inference[mkl]
```

(This extra triggers the build of the MKL Cython extension during installation.)

#### GSL Backend (Good Alternative CPU Performance):

**Prerequisite**: Install GSL Development Libraries (headers and libraries).

- **Ubuntu/Debian**: `sudo apt-get update && sudo apt-get install libgsl-dev`
- **macOS (Homebrew)**: `brew install gsl`
- **Conda**: `conda install -c conda-forge gsl`

Ensure `gsl-config` is on your PATH or set the `GSL_HOME` environment variable.

Install with GSL extra:

```bash
pip install ridge-inference[gsl]
```

(This extra triggers the build of the GSL Cython extension during installation.)

#### GPU Acceleration (NVIDIA):

**Prerequisite**: Install NVIDIA drivers, CUDA Toolkit, and CuPy that matches your CUDA version. See CuPy Installation Guide.

Install with GPU extra:

```bash
pip install ridge-inference[gpu]
```

(Installs the cupy Python package.)

#### Numba Acceleration:

**Prerequisite**: None (Numba is pure Python).

Install with Numba extra:

```bash
pip install ridge-inference[numba]
```

(Installs the numba Python package.)

#### All Optional Python Dependencies:

Install prerequisites for MKL/GSL/GPU separately if desired.

Install all optional Python packages (Numba, CuPy, IO libs, Viz libs):

```bash
pip install ridge-inference[full]
```

## Quick Start

### Basic Ridge Regression

```python
import numpy as np
from ridge_inference import ridge, get_backend_info

# Example data
X = np.random.rand(1000, 20)  # 1000 genes, 20 features
Y = np.random.rand(1000, 50)  # 1000 genes, 50 samples

# See available backends
print(get_backend_info())

# Run ridge regression with permutation test (auto-selects best backend)
# method="auto" is the default
results = ridge(X, Y, lambda_=5e5, n_rand=1000)

# Explicitly choose MKL backend (if installed)
# results_mkl = ridge(X, Y, lambda_=5e5, n_rand=1000, method="mkl")

# Access results
beta = results['beta']          # Regression coefficients (features x samples)
se = results['se']              # Standard errors
zscore = results['zscore']      # Z-scores (or t-stats if n_rand=0)
pvalue = results['pvalue']      # P-values
method = results['method_used'] # Backend actually used (e.g., 'mkl', 'python')

# T-test instead of permutation test (requires dense X, uses gsl_cython or python)
# results_ttest = ridge(X, Y, lambda_=5e5, n_rand=0, alternative="two-sided")
```

### SecAct Inference

```python
import pandas as pd
from ridge_inference import secact_inference, load_signature_matrix

# Load expression data (genes x samples)
# expr_data = pd.read_csv("expression_data.csv", index_col=0)
# Placeholder example:
expr_data = pd.DataFrame(np.random.rand(500, 30), index=[f'Gene_{i}' for i in range(500)], columns=[f'Sample_{j}' for j in range(30)])


# Load built-in signature matrix (genes x pathways)
sig_matrix = load_signature_matrix("SecAct")  # or "CytoSig", or path to custom file

# Run inference (auto-selects best backend)
results = secact_inference(
    expr_data=expr_data,
    sig_matrix=sig_matrix,
    lambda_val=5e5,
    n_rand=1000,
    method="auto",        # Let the library choose the best backend
    add_background=True,  # Optional: add data-driven background feature
    scale_method="column" # Optional: scale columns before regression
)

# Access results (as DataFrames)
beta_df = results['beta']       # Pathway activities (pathways x samples)
pvalue_df = results['pvalue']   # Significance values
method_used = results['method'] # Backend used

print(f"SecAct inference completed using: {method_used}")
print("Beta DataFrame head:\n", beta_df.head())
```

### Setting Default Backend

```python
from ridge_inference import set_backend, get_backend_info

# Set MKL as the preferred backend for subsequent 'auto' calls
# (Only takes effect if MKL backend was successfully built/installed)
set_backend("mkl")

print("Backend preference set.")
print(get_backend_info()['current_settings'])

# Run using the configured preference if method="auto"
# results = ridge(X, Y, lambda_=5e5, n_rand=1000)
```

## Command-Line Interface

Ridge Inference provides two command-line tools for convenience:

### General Ridge Regression (ridge-inference-main)

Performs ridge regression between two input matrices (X and Y).

```bash
# Example: Using auto backend selection, verbose output
ridge-inference-main \
    -x features.csv \
    -y targets.csv \
    -o results_main/ \
    -l 5e5 \
    -n 1000 \
    -v

# Example: Explicitly request MKL backend via environment variable
RIDGE_BACKEND=mkl ridge-inference-main \
    -x features.csv \
    -y targets.csv \
    -o results_mkl/ \
    -l 5e5 \
    -n 1000 \
    -v
```

Use `ridge-inference-main --help` for all options.

### SecAct Inference (ridge-inference-secact)

Performs SecAct inference using gene expression data and a signature matrix.

```bash
# Example: Using built-in 'SecAct' signature, adding background, auto backend
ridge-inference-secact \
    -y expression.csv \
    -s SecAct \
    -o results_secact/ \
    -l 5e5 \
    -n 1000 \
    --add_background \
    -v

# Example: Using a custom signature file and requesting GPU backend
RIDGE_BACKEND=gpu ridge-inference-secact \
    -y expression.csv \
    -s ./custom_signature.tsv.gz \
    -o results_secact_gpu/ \
    -l 5e5 \
    -n 1000 \
    -v
```

Use `ridge-inference-secact --help` for all options.

## Advanced Features

### Batch Processing for Large Datasets

For datasets with a very large number of samples (columns in Y), batch processing can significantly reduce peak memory usage. This is automatically handled by secact_inference and can be controlled via arguments. Currently supported by python and gpu backends for permutation tests (n_rand > 0).

```python
from ridge_inference import secact_inference

# Assume large_expression_data has > 50000 samples
results = secact_inference(
    expr_data=large_expression_data,
    sig_matrix="SecAct",
    lambda_val=5e5,
    n_rand=1000,
    method="gpu",          # Specify GPU if available
    batch_size=10000,      # Process 10000 samples per batch (optional, default is auto-estimated)
    batch_threshold=50000, # Enable batching if samples > 50000 (default)
    verbose=1
)

print(f"Batching used: {results.get('batched')}")
print(f"Batch size: {results.get('batch_size')}")
```

### GPU Acceleration

If CuPy is installed and a compatible GPU is available, select the gpu backend for potentially faster computation, especially with large, dense matrices and permutation tests.

```python
# Ensure CuPy is installed: pip install ridge-inference[gpu]
results = ridge(X, Y, lambda_=5e5, n_rand=1000, method="gpu")

if results['method_used'] == 'gpu':
    print(f"GPU Execution time: {results['execution_time']:.2f}s")
    if results['peak_gpu_pool_mb']:
        print(f"Peak GPU memory: {results['peak_gpu_pool_mb']:.2f} MB")
```

### Sparse Matrix Support

The library handles sparse input matrices (scipy.sparse formats).

- **Y Matrix**: Sparse Y is supported by python, mkl, numba, and gpu backends. MKL is often highly optimized for dense X @ sparse Y.
- **X Matrix**: Sparse X is currently only supported by the python backend using an iterative solver (LSMR) for permutation tests. This is significantly slower than direct methods used for dense X.

```python
import scipy.sparse as sps
import numpy as np
from ridge_inference import ridge

# Example: Dense X, Sparse Y (common use case)
X_dense = np.random.rand(1000, 20)
Y_sparse = sps.random(1000, 500, density=0.1, format='csr')

# MKL is often a good choice here
# results = ridge(X_dense, Y_sparse, lambda_=1e3, n_rand=500, method="mkl")

# Example: Sparse X, Sparse Y (uses Python backend)
X_sparse = sps.random(1000, 20, density=0.2, format='csr')
# results = ridge(X_sparse, Y_sparse, lambda_=1e3, n_rand=500, method="python") # Auto will select python
```

## Backend Selection Guide

The `method` parameter in `ridge()` and `secact_inference()` controls the backend. `method="auto"` (default) selects the best available backend based on input types and installed libraries.

| Backend (method=) | Description | Best For | T-Test (n_rand=0) | Sparse X | Sparse Y | Prerequisites |
|-------------------|-------------|----------|-------------------|----------|----------|---------------|
| `gsl` | GSL (via Cython) | Dense X, Dense Y, CPU tasks w/o MKL | Yes | No | No (densified) | GSL dev libs |
| `numba` | Numba JIT (Python) | Dense X, Sparse/Dense Y, Medium CPU tasks w/o C libs | No | No | Yes | numba package |
| `python` | NumPy/SciPy (Python) | Universal fallback, T-tests, Sparse X (iterative) | Yes | Yes | Yes | None |
| `mkl` | Intel MKL (via Cython) | Dense X, Sparse/Dense Y, Large CPU tasks, Multithreading | No | No | Yes | MKL dev libs |
| `gpu` | NVIDIA GPU (via CuPy) | Dense X, Dense/Sparse Y, Very Large Dense tasks | No | No | Yes | CUDA + CuPy |


Notes:

- "Prerequisites" listed are needed before running `pip install ridge-inference[extra]`.
- `auto` selection prioritizes: mkl > gpu > gsl_cython > numba > python (adjusted for input type compatibility).
- T-tests (n_rand=0) are currently only implemented for gsl_cython and python backends and require dense X.

## Performance Considerations

- For large datasets (many samples), use batch processing (`batch_threshold`/`batch_size` in `secact_inference` or use `ridge_batch` directly) with `gpu` or `python` backends to manage memory.
- For large dense matrices, `gpu` or `mkl` backends typically provide the best performance.
- For dense X and sparse Y, the `mkl` backend is often highly optimized.
- Sparse X input significantly reduces performance as it requires an iterative solver (`python` backend only).
- Control CPU threading for MKL/GSL backends using environment variables:

```bash
export OMP_NUM_THREADS=N  # For OpenMP parallelism (GSL, sometimes MKL)
export MKL_NUM_THREADS=N  # Specifically for MKL threading
```

Setting both is often recommended when using MKL.

## Compatibility Notes

### MKL Backend Prerequisites

Building the MKL backend requires Intel MKL development libraries (headers + shared libraries).

- **Conda**: The easiest way is `conda install -c conda-forge mkl-devel`.
- **Manual/Intel Install**: Download the oneAPI HPC Toolkit or Base Toolkit from Intel. Ensure the `MKLROOT` environment variable points to the installation directory (e.g., `/opt/intel/oneapi/mkl/latest`).

The build script (setup.py) will attempt to find MKL via `MKLROOT` or `CONDA_PREFIX`.

### GSL Backend Prerequisites

Building the GSL backend requires GSL development libraries (headers + shared libraries), version >= 2.5 recommended.

Install using your system package manager (e.g., apt, brew) or Conda (see Installation section).

The build script (setup.py) attempts to find GSL using pkg-config or the `GSL_HOME` environment variable.

## License

This project is licensed under the MIT License - see the LICENSE file for details.