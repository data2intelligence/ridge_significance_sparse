# ridge_inference/__init__.py

"""
Ridge Inference - High-performance ridge regression with multiple backend implementations

This package provides efficient ridge regression with permutation or t-test significance
testing. Multiple optimized backends are supported, including Intel MKL, GSL, Numba, and
GPU (CuPy) acceleration.

Main functions:
- ridge: Core ridge regression with multiple backend options
- secact_inference: Specialized inference for secreted protein activities
- activity_inference: Alias for secact_activity_inference
- load_signature_matrix: Load built-in or custom signature matrices
- set_backend: Configure the preferred computational backend
- set_threads: Control threading for multi-threaded backends (MKL, OpenMP, GSL-BLAS)
- get_backend_info: Get detailed information about available backends

Environment variables:
- RIDGE_BACKEND: Set default backend ('auto', 'mkl', 'gsl_cython', 'numba', 'gpu', 'python')
- RIDGE_LOG_LEVEL: Set logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
- RIDGE_FALLBACK: Enable/disable backend fallbacks ('0'=disabled, '1'=enabled)
- MKL_NUM_THREADS: Control Intel MKL threading
- OMP_NUM_THREADS: Control OpenMP threading (affects GSL, Numba, MKL OpenMP parts)

See https://github.com/data2intelligence/ridge_significance_sparse for more information.
"""

__version__ = "0.3.0" # Increment version reflecting backend changes

# Configure logging first (before other imports that might log)
import logging
import os

# Set up basic logging configuration if not already configured by the user
if not logging.root.handlers:
    log_level_str = os.environ.get("RIDGE_LOG_LEVEL", "WARNING").upper()
    valid_levels = {"DEBUG": logging.DEBUG, "INFO": logging.INFO,
                    "WARNING": logging.WARNING, "ERROR": logging.ERROR,
                    "CRITICAL": logging.CRITICAL}
    level = valid_levels.get(log_level_str, logging.WARNING)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)-8s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

# Now import other modules
from .backend_selection import (
    MKL_AVAILABLE,
    GSL_CYTHON_AVAILABLE,
    NUMBA_AVAILABLE,
    CUPY_AVAILABLE,
    PYTHON_AVAILABLE,
    set_backend,
    list_backends as list_available_backends, # Rename for clarity
    set_threads, # Use the unified set_threads function
    get_mkl_threads,
    get_gsl_blas_threads,
    get_gsl_omp_max_threads,
    get_mkl_omp_max_threads,
    BACKEND_PREFERENCE, # Export global settings if needed
    BACKEND_FALLBACK_ENABLED
)

# Import main functions AFTER backend selection is loaded
from .ridge import (
    ridge
    # Note: set_backend_preference and set_thread_count are now aliases
    # defined in ridge.py if needed, prefer direct use of set_backend/set_threads
)

# Import SecAct functionality
from .inference import (
    secact_inference,
    secact_activity_inference,
)
# Add shorter alias for secact_activity_inference
activity_inference = secact_activity_inference

# Import data loading utilities
from .loaders import (
    load_dataset,
    load_matrix_data,
    load_signature_matrix,
)

# Import utility functions
from .utils import (
    scale_dataframe_columns,
    visualize_activity,
)

# Optional: Logistic regression module
try:
    from .logit import logit
except ImportError:
    def logit(*args, **kwargs):
        raise ImportError("Logistic regression module not available.")

# Log basic info on import
logger = logging.getLogger(__name__)
logger.info(f"Initializing Ridge Inference v{__version__}")
logger.debug(f"Available backends: MKL={MKL_AVAILABLE}, GSL_CYTHON={GSL_CYTHON_AVAILABLE}, "
             f"NUMBA={NUMBA_AVAILABLE}, GPU={CUPY_AVAILABLE}, PYTHON={PYTHON_AVAILABLE}")

# Define function to get detailed backend information
def get_backend_info():
    """
    Get detailed information about available backends and their status.

    Returns
    -------
    dict
        Dictionary with detailed information about each backend.
    """
    # Get current thread settings using functions from backend_selection
    mkl_actual_threads = get_mkl_threads() if MKL_AVAILABLE else 'N/A'
    gsl_blas_actual_threads = get_gsl_blas_threads() if GSL_CYTHON_AVAILABLE else 'N/A'
    gsl_omp_actual_threads = get_gsl_omp_max_threads() if GSL_CYTHON_AVAILABLE else 'N/A'
    mkl_omp_actual_threads = get_mkl_omp_max_threads() if MKL_AVAILABLE else 'N/A'


    info = {
        'current_settings': {
            'preferred_backend': BACKEND_PREFERENCE,
            'fallback_enabled': BACKEND_FALLBACK_ENABLED,
            'mkl_threads_env': os.environ.get("MKL_NUM_THREADS", "auto"),
            'omp_threads_env': os.environ.get("OMP_NUM_THREADS", "auto"),
            'mkl_threads_actual': mkl_actual_threads,
            'gsl_blas_threads_actual': gsl_blas_actual_threads,
            'gsl_omp_threads_actual': gsl_omp_actual_threads,
            'mkl_omp_threads_actual': mkl_omp_actual_threads,
        },
        'backends': {
            'mkl': {
                'name': 'MKL (via Cython)',
                'available': MKL_AVAILABLE,
                'features': ['Dense X', 'Sparse Y (CSR)', 'Dense Y', 'Multithreading (MKL + OpenMP)', 'Permutation tests'],
                'limitations': ['T-test not implemented', 'Requires MKL library'],
                'best_for': 'Large matrices (dense X, sparse/dense Y), high-performance CPU',
            },
            'gsl_cython': {
                'name': 'GSL (via Cython)',
                'available': GSL_CYTHON_AVAILABLE,
                'features': ['Dense X', 'Dense Y', 'Multithreading (OpenMP + BLAS)', 'Permutation tests', 'T-tests'],
                'limitations': ['Requires dense Y input', 'Requires system GSL library'],
                'best_for': 'Medium/Large dense matrices on systems without MKL',
            },
            'numba': {
                'name': 'Numba',
                'available': NUMBA_AVAILABLE,
                'features': ['Dense X', 'Sparse Y (CSC)', 'Dense Y', 'JIT compilation', 'Permutation tests'], # Updated Y support
                'limitations': ['T-test not implemented'],
                'best_for': 'Medium dense X / sparse or dense Y without MKL/GSL/GPU',
            },
            'gpu': {
                'name': 'GPU (CuPy)',
                'available': CUPY_AVAILABLE,
                'features': ['Dense X', 'Sparse Y (CSR)', 'Dense Y', 'GPU acceleration', 'Permutation tests'],
                'limitations': ['Requires CUDA GPU and CuPy', 'T-test not implemented', 'Sparse Y performance may vary'],
                'best_for': 'Very large dense X matrices with GPU availability',
            },
            'python': {
                'name': 'Python (NumPy/SciPy)',
                'available': PYTHON_AVAILABLE,
                'features': ['Dense X', 'Sparse X (iterative)', 'Dense Y', 'Sparse Y', 'Permutation tests', 'T-tests (dense X)'],
                'limitations': ['Slower than accelerated backends', 'Sparse X uses iterative solver (slower)'],
                'best_for': 'Sparse X input, T-tests, compatibility guarantee, smaller datasets',
            }
        }
    }
    return info

# Define __all__ for explicit public API exposure
__all__ = [
    # Core functions
    "ridge",
    # Inference functions
    "secact_inference",
    "activity_inference",
    "secact_activity_inference",
    # Loaders
    "load_signature_matrix",
    "load_dataset",
    "load_matrix_data",
    # Configuration
    "set_backend",
    "set_threads",
    "list_available_backends",
    "get_backend_info",
    # Utilities
    "scale_dataframe_columns",
    "visualize_activity",
    # Optional modules
    "logit",
    # Availability flags
    "MKL_AVAILABLE",
    "GSL_CYTHON_AVAILABLE",
    "NUMBA_AVAILABLE",
    "CUPY_AVAILABLE",
    "PYTHON_AVAILABLE",
    # Metadata
    "__version__",
]