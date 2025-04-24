# ridge_inference/backend_selection.py

"""
Backend selection and management module for ridge_inference package.

Handles detection, initialization, and selection of computational backends
(MKL, GSL-Cython, Numba, CuPy, Python).
"""

import os
import logging
import platform
import warnings
from scipy import sparse as sps
import numpy as np

logger = logging.getLogger(__name__)

# --- Backend availability flags ---
MKL_AVAILABLE = False
GSL_CYTHON_AVAILABLE = False
NUMBA_AVAILABLE = False
CUPY_AVAILABLE = False
PYTHON_AVAILABLE = True  # Assume Python/NumPy backend is always available

# --- Backend settings ---
BACKEND_PREFERENCE = os.environ.get("RIDGE_BACKEND", "auto").lower()
BACKEND_FALLBACK_ENABLED = os.environ.get("RIDGE_FALLBACK", "1") == "1"

# --- Thread settings ---
try:
    # MKL threads controlled via its own API or env var
    _mkl_env_threads_str = os.environ.get("MKL_NUM_THREADS")
    MKL_NUM_THREADS = int(_mkl_env_threads_str) if _mkl_env_threads_str is not None and _mkl_env_threads_str.isdigit() else 0
    if _mkl_env_threads_str is not None and not _mkl_env_threads_str.isdigit():
         logger.warning(f"Invalid MKL_NUM_THREADS value '{_mkl_env_threads_str}' in environment, defaulting to 0 (auto).")
except ValueError: # Should not happen with isdigit check, but safeguard
    logger.warning("Error parsing MKL_NUM_THREADS value in environment, defaulting to 0 (auto).")
    MKL_NUM_THREADS = 0

try:
    # OMP threads control OpenMP regions (used by GSL, Numba, potentially MKL)
    _omp_env_threads_str = os.environ.get("OMP_NUM_THREADS")
    OMP_NUM_THREADS = int(_omp_env_threads_str) if _omp_env_threads_str is not None and _omp_env_threads_str.isdigit() else 0
    if _omp_env_threads_str is not None and not _omp_env_threads_str.isdigit():
         logger.warning(f"Invalid OMP_NUM_THREADS value '{_omp_env_threads_str}' in environment, defaulting to 0 (auto).")
except ValueError:
    logger.warning("Error parsing OMP_NUM_THREADS value in environment, defaulting to 0 (auto).")
    OMP_NUM_THREADS = 0


# --- Import backends and update availability flags ---

# MKL Backend (Intel Math Kernel Library via Cython)
try:
    # Import from the *new* MKL Cython wrapper
    from .ridge_mkl import (
        ridge_regression_mkl,
        set_mkl_threads,
        get_mkl_threads,
        set_mkl_omp_threads, # OMP control specific to MKL build env
        get_mkl_omp_max_threads
    )
    # *** SIMPLIFIED CHECK: Assume available if import succeeds ***
    MKL_AVAILABLE = True
    logger.info("MKL backend available (via ridge_mkl Cython wrapper - based on successful import).")
    # Optional: Add a check here later if specific MKL functions fail at runtime
    # try:
    #     _ = get_mkl_threads()
    #     logger.debug("get_mkl_threads() check passed.")
    # except Exception as mkl_runtime_e:
    #     logger.warning(f"MKL import succeeded, but runtime check failed: {mkl_runtime_e}")
    #     MKL_AVAILABLE = False # Revert if check fails

except ImportError as e:
    logger.info(f"MKL backend not available: {e}")
    MKL_AVAILABLE = False
    # Define dummy functions if MKL backend fails to import or initialize
    def ridge_regression_mkl(*args, **kwargs):
        raise ImportError("MKL backend not available. Build with MKL support enabled.")
    def set_mkl_threads(num_threads):
        warnings.warn("MKL backend not available, cannot set MKL threads.")
        return -1
    def get_mkl_threads():
        return -1 # Indicate unavailable
    def set_mkl_omp_threads(num_threads):
         warnings.warn("MKL backend not available, cannot set MKL OMP threads.")
         return -1
    def get_mkl_omp_max_threads():
         return 1 # Assume 1 if MKL build env OMP is unavailable

# GSL Backend via Cython
try:
    # Import from the *new* GSL Cython wrapper
    from .ridge_gsl import (
        ridge_regression_gsl,
        set_gsl_blas_threads,
        get_gsl_blas_threads,
        set_gsl_omp_threads, # OMP control specific to GSL build env
        get_gsl_omp_max_threads
    )
    # Assume available if import succeeds, relying on build to fail if not linkable
    GSL_CYTHON_AVAILABLE = True
    logger.info("GSL backend available (via ridge_gsl Cython wrapper).")
except ImportError as e:
    logger.info(f"GSL backend not available: {e}")
    GSL_CYTHON_AVAILABLE = False
    def ridge_regression_gsl(*args, **kwargs):
        raise ImportError("GSL backend not available. Install GSL and build with GSL support enabled.")
    # Dummy functions for GSL thread control
    def set_gsl_blas_threads(num_threads):
        warnings.warn("GSL backend not available, cannot set BLAS threads.")
        return -1
    def get_gsl_blas_threads():
        return -1
    def set_gsl_omp_threads(num_threads):
         warnings.warn("GSL backend not available, cannot set GSL OMP threads.")
         return -1
    def get_gsl_omp_max_threads():
         return 1

# Numba Backend
try:
    from .core import ridge_regression_numba, NUMBA_AVAILABLE as CORE_NUMBA_AVAILABLE
    NUMBA_AVAILABLE = CORE_NUMBA_AVAILABLE
    if NUMBA_AVAILABLE:
        import numba
        logger.info(f"Numba backend available: {numba.__version__}")
    else:
        raise ImportError("Numba reported unavailable by core module.")
except ImportError as e:
    logger.info(f"Numba backend not available: {e}")
    def ridge_regression_numba(*args, **kwargs):
        raise ImportError("Numba backend not available. Install with 'pip install ridge-inference[numba]'.")
    NUMBA_AVAILABLE = False

# GPU Backend (CuPy)
try:
    from .core import ridge_regression_cupy, CUPY_AVAILABLE as CORE_CUPY_AVAILABLE
    CUPY_AVAILABLE = CORE_CUPY_AVAILABLE
    if CUPY_AVAILABLE:
        import cupy as cp
        logger.info(f"GPU backend available: CuPy {cp.__version__}")
    else:
        raise ImportError("CuPy initialization failed or reported unavailable by core module.")
except (ImportError, AttributeError, Exception) as e:
    logger.info(f"GPU backend not available: {type(e).__name__} - {e}")
    def ridge_regression_cupy(*args, **kwargs):
        raise ImportError("GPU backend not available. Install with 'pip install ridge-inference[gpu]' and ensure CUDA is configured.")
    CUPY_AVAILABLE = False

# Python Backend (always available via core.py)
try:
    from .core import ridge_regression_numpy
except ImportError as e:
    logger.error(f"CRITICAL: NumPy backend (core.ridge_regression_numpy) not available: {e}")
    def ridge_regression_numpy(*args, **kwargs):
        raise ImportError("NumPy backend (core.ridge_regression_numpy) not available. This indicates a critical setup error.")
    PYTHON_AVAILABLE = False


def set_backend(backend_name, fallback=True):
    """
    Set the preferred backend for ridge operations.

    Parameters
    ----------
    backend_name : str
        Backend: 'auto', 'mkl', 'gsl_cython', 'gsl', 'numba', 'gpu', 'python'.
    fallback : bool
        Allow fallback if selected backend fails.

    Returns
    -------
    str
        Actually selected backend preference name.
    """
    global BACKEND_PREFERENCE, BACKEND_FALLBACK_ENABLED
    valid_backends = ['auto', 'mkl', 'gsl_cython', 'gsl', 'numba', 'gpu', 'python']
    requested_backend = backend_name.lower()

    if requested_backend not in valid_backends:
        raise ValueError(f"Unknown backend: '{backend_name}'. Valid: {valid_backends}")

    # Handle 'gsl' alias -> 'gsl_cython'
    if requested_backend == 'gsl':
        effective_backend = 'gsl_cython'
        logger.info("'gsl' alias resolved to 'gsl_cython'.")
    else:
        effective_backend = requested_backend

    BACKEND_PREFERENCE = effective_backend
    BACKEND_FALLBACK_ENABLED = fallback

    # Verify availability if specific backend requested
    if effective_backend not in ['auto', 'python']:
        available = is_backend_available(effective_backend)
        if not available:
            available_list = get_available_backends()
            logger.warning(f"Requested backend '{effective_backend}' is not available. Available: {available_list}")
            if not fallback:
                logger.error("Fallback is disabled - operations requiring this specific backend will likely fail.")
        else:
             logger.info(f"Requested backend '{effective_backend}' is available.")

    logger.info(f"Backend preference set to: {BACKEND_PREFERENCE} (fallback enabled: {BACKEND_FALLBACK_ENABLED})")
    return BACKEND_PREFERENCE


def list_available_backends(verbose=True):
    """List all available backends with their status (uses alias list_backends)."""
    return list_backends(verbose=verbose) # Call the main listing function


def is_backend_available(backend_name):
    """Check if a specific backend is available."""
    check_name = backend_name.lower()
    if check_name == 'mkl':
        return MKL_AVAILABLE
    elif check_name == 'gsl_cython' or check_name == 'gsl': # Check alias too
        return GSL_CYTHON_AVAILABLE
    elif check_name == 'numba':
        return NUMBA_AVAILABLE
    elif check_name == 'gpu':
        return CUPY_AVAILABLE
    elif check_name == 'python':
        return PYTHON_AVAILABLE
    elif check_name == 'auto':
        return True
    else:
        logger.warning(f"Checking availability for unknown backend name: '{backend_name}'")
        return False


def get_available_backends():
    """Get a list of names for all available backends."""
    available = []
    if MKL_AVAILABLE: available.append('mkl')
    if GSL_CYTHON_AVAILABLE: available.append('gsl_cython')
    if NUMBA_AVAILABLE: available.append('numba')
    if CUPY_AVAILABLE: available.append('gpu')
    if PYTHON_AVAILABLE: available.append('python')
    return available


def select_optimal_backend(X, Y, n_rand=None, prefer_mkl=None, user_selection=None):
    """
    Select the optimal backend based on input, availability, and capabilities.

    Parameters
    ----------
    X : array-like or scipy.sparse matrix
    Y : array-like or scipy.sparse matrix
    n_rand : int or None
        Number of permutations (0 for t-test). Default is None (~1000).
    prefer_mkl : bool or None
        Whether to prefer MKL. Default None behaves like True.
    user_selection : str or None
        User-requested backend name ('auto', 'mkl', etc.). Overrides auto-selection.

    Returns
    -------
    str
        Selected backend name.
    """
    # Handle n_rand=None default
    n_rand_eff = 1000 if n_rand is None else n_rand

    # 1. Handle User Request
    if user_selection is not None and user_selection != 'auto':
        user_req_lower = user_selection.lower()
        # Resolve 'gsl' alias
        effective_req = 'gsl_cython' if user_req_lower == 'gsl' else user_req_lower

        if is_backend_available(effective_req):
            # --- Compatibility checks for user requests ---
            is_X_sparse = sps.issparse(X)
            is_Y_sparse = sps.issparse(Y)

            # T-test compatibility (GSL or Python required)
            if n_rand_eff == 0 and effective_req not in ['gsl_cython', 'python']:
                 logger.warning(f"User requested '{user_selection}' but t-test requires 'gsl_cython' or 'python'. Overriding to 'gsl_cython' if available, else 'python'.")
                 return 'gsl_cython' if GSL_CYTHON_AVAILABLE else 'python'

            # Sparse X compatibility (Python required)
            if is_X_sparse and effective_req != 'python':
                 logger.warning(f"User requested '{user_selection}' but sparse X requires 'python' (iterative). Overriding to 'python'.")
                 return 'python'

            # MKL T-test incompatibility
            if n_rand_eff == 0 and effective_req == 'mkl':
                 logger.warning("MKL backend does not support t-test (n_rand=0). Overriding to 'gsl_cython' if available, else 'python'.")
                 return 'gsl_cython' if GSL_CYTHON_AVAILABLE else 'python'

            # GSL sparse Y notification (will be densified in ridge())
            if effective_req == 'gsl_cython' and is_Y_sparse:
                 logger.info(f"User requested 'gsl_cython' with sparse Y. Input Y will be densified by ridge() before calling backend.")

            logger.info(f"Using user-requested backend: '{effective_req}' (resolved from '{user_selection}')")
            return effective_req
        elif BACKEND_FALLBACK_ENABLED:
            logger.warning(f"User-requested backend '{user_selection}' not available, falling back to auto-selection.")
            # Fall through to auto-selection
        else:
            raise ValueError(f"Requested backend '{user_selection}' is not available and fallback is disabled.")

    # 2. Auto-Selection Logic
    logger.debug("Performing auto-selection of backend...")
    is_X_sparse = sps.issparse(X)
    is_Y_sparse = sps.issparse(Y)

    # Handle t-test (n_rand=0)
    if n_rand_eff == 0:
        if is_X_sparse:
            raise NotImplementedError("T-test (n_rand=0) requires dense X. Use permutation test for sparse X.")
        # Prefer GSL if available for t-test, else Python
        backend = 'gsl_cython' if GSL_CYTHON_AVAILABLE else 'python'
        logger.info(f"Auto-select: T-test requested, using '{backend}' backend.")
        return backend

    # Handle Sparse X (n_rand > 0) - Requires Python backend's iterative solver
    if is_X_sparse:
        logger.info("Auto-select: Sparse X detected, using 'python' backend (iterative solver).")
        return 'python'

    # --- Dense X Logic (n_rand > 0) ---
    logger.debug(f"Auto-select: Dense X detected (Y is {'sparse' if is_Y_sparse else 'dense'}). Permutation test.")

    # Preference Order: MKL > GPU > GSL > Numba > Python
    if (prefer_mkl is None or prefer_mkl) and MKL_AVAILABLE:
        logger.info("Auto-select: MKL available and preferred, using 'mkl' backend.")
        return 'mkl'

    if CUPY_AVAILABLE:
        logger.info("Auto-select: GPU available, using 'gpu' backend.")
        return 'gpu'

    if GSL_CYTHON_AVAILABLE:
         # Add a check here: GSL doesn't handle sparse Y directly
         if is_Y_sparse:
             logger.info("Auto-select: GSL available but Y is sparse. Considering Numba/Python instead.")
             if NUMBA_AVAILABLE:
                 logger.info("Auto-select: Numba available, using 'numba' backend for sparse Y.")
                 return 'numba'
             else:
                 logger.info("Auto-select: Numba not available, falling back to 'python' for sparse Y.")
                 return 'python'
         else:
             # Dense X, Dense Y, GSL is a good choice
             logger.info("Auto-select: GSL Cython available, using 'gsl_cython' backend.")
             return 'gsl_cython'

    if NUMBA_AVAILABLE:
        # Numba now handles both dense/sparse Y in core.py implementation
        logger.info("Auto-select: Numba available, using 'numba' backend.")
        return 'numba'

    logger.info("Auto-select: No other accelerated backend suitable/available, falling back to 'python'.")
    return 'python'


def get_backend_function(backend_name):
    """Get the backend implementation function by name."""
    name_lower = backend_name.lower()
    if name_lower == 'mkl':
        if not MKL_AVAILABLE: raise ImportError("MKL backend is not available.")
        return ridge_regression_mkl
    elif name_lower == 'gsl_cython' or name_lower == 'gsl': # Handle alias
        if not GSL_CYTHON_AVAILABLE: raise ImportError("GSL Cython backend is not available.")
        return ridge_regression_gsl
    elif name_lower == 'numba':
        if not NUMBA_AVAILABLE: raise ImportError("Numba backend is not available.")
        return ridge_regression_numba
    elif name_lower == 'gpu':
        if not CUPY_AVAILABLE: raise ImportError("GPU (CuPy) backend is not available.")
        return ridge_regression_cupy
    elif name_lower == 'python':
        if not PYTHON_AVAILABLE: raise ImportError("Python (NumPy) backend is not available (critical error).")
        return ridge_regression_numpy
    else:
        raise ValueError(f"Unknown backend requested: '{backend_name}'")


def list_backends(verbose=True):
    """List all available backends and their status."""
    backends = {
        'mkl': MKL_AVAILABLE,
        'gsl_cython': GSL_CYTHON_AVAILABLE,
        'numba': NUMBA_AVAILABLE,
        'gpu': CUPY_AVAILABLE,
        'python': PYTHON_AVAILABLE
    }

    if verbose:
        print("\n" + "="*10 + " Ridge Inference Backend Status " + "="*10)
        print(f"Current Preference : {BACKEND_PREFERENCE}")
        print(f"Fallback Enabled   : {BACKEND_FALLBACK_ENABLED}")
        print("-" * 46)
        print("Available Backends:")
        for name, available in backends.items():
            status = "✓ AVAILABLE" if available else "✗ UNAVAILABLE"
            preferred = " (PREFERRED)" if BACKEND_PREFERENCE == name else ""
            print(f"  - {name:<12} : {status}{preferred}")

        print("-" * 46)
        print("Thread Settings:")
        mkl_actual, gsl_blas_actual, gsl_omp_actual, mkl_omp_actual = ('N/A',) * 4
        try: mkl_actual = get_mkl_threads() if MKL_AVAILABLE else 'N/A'
        except NameError: pass # Function might not exist if MKL_AVAILABLE is False due to import error
        except Exception as e: logger.debug(f"Error calling get_mkl_threads: {e}"); mkl_actual = 'Error' # Capture runtime errors
        try: gsl_blas_actual = get_gsl_blas_threads() if GSL_CYTHON_AVAILABLE else 'N/A'
        except NameError: pass
        except Exception as e: logger.debug(f"Error calling get_gsl_blas_threads: {e}"); gsl_blas_actual = 'Error'
        try: gsl_omp_actual = get_gsl_omp_max_threads() if GSL_CYTHON_AVAILABLE else 'N/A' # Use GSL_CYTHON_AVAILABLE
        except NameError: pass
        except Exception as e: logger.debug(f"Error calling get_gsl_omp_max_threads: {e}"); gsl_omp_actual = 'Error'
        try: mkl_omp_actual = get_mkl_omp_max_threads() if MKL_AVAILABLE else 'N/A' # Use MKL_AVAILABLE
        except NameError: pass
        except Exception as e: logger.debug(f"Error calling get_mkl_omp_max_threads: {e}"); mkl_omp_actual = 'Error'

        print(f"  MKL Threads Env    : {MKL_NUM_THREADS if MKL_NUM_THREADS > 0 else 'auto'} (Actual: {mkl_actual})")
        print(f"  OMP Threads Env    : {OMP_NUM_THREADS if OMP_NUM_THREADS > 0 else 'auto'} (GSL OMP Max: {gsl_omp_actual}, MKL OMP Max: {mkl_omp_actual})")
        print(f"  GSL BLAS Threads   : (Actual: {gsl_blas_actual})")
        print("-" * 46)
        print("System Information:")
        print(f"  Platform       : {platform.system()} ({platform.machine()})")
        print(f"  Platform Detail: {platform.platform(aliased=True)}")
        print(f"  Python Version : {platform.python_version()}")
        try:
            print(f"  CPU Core Count : {os.cpu_count()}")
        except NotImplementedError:
             print("  CPU Core Count : Not Available")
        print("="*46)

    return backends


def set_threads(mkl_threads=None, omp_threads=None, gsl_blas_threads=None):
    """
    Set the number of threads used by different backends.

    Parameters
    ----------
    mkl_threads : int or None
        Threads for MKL backend. If None, unchanged. 0=auto.
    omp_threads : int or None
        Threads for OpenMP (used by GSL, Numba, potentially MKL's non-MKL parts).
        If None, unchanged. 0=auto. Sets OMP_NUM_THREADS env var.
        Also attempts to set GSL OMP and MKL OMP threads directly if available.
    gsl_blas_threads : int or None
        Threads specifically for GSL's linked BLAS (e.g., OpenBLAS).
        If None, unchanged. 0=auto.

    Returns
    -------
    dict
        Previous effective thread settings before changes were applied.
    """
    global MKL_NUM_THREADS, OMP_NUM_THREADS # Update tracked env vars

    # Read current effective settings to return
    prev_mkl = -1; prev_omp_gsl = 1; prev_omp_mkl = 1; prev_gsl_blas = -1
    try: prev_mkl = get_mkl_threads() if MKL_AVAILABLE else -1
    except Exception: pass
    try: prev_omp_gsl = get_gsl_omp_max_threads() if GSL_CYTHON_AVAILABLE else 1
    except Exception: pass
    try: prev_omp_mkl = get_mkl_omp_max_threads() if MKL_AVAILABLE else 1
    except Exception: pass
    try: prev_gsl_blas = get_gsl_blas_threads() if GSL_CYTHON_AVAILABLE else -1
    except Exception: pass

    previous = {
        'MKL_NUM_THREADS': prev_mkl,
        'OMP_NUM_THREADS_GSL': prev_omp_gsl,
        'OMP_NUM_THREADS_MKL': prev_omp_mkl,
        'GSL_BLAS_THREADS': prev_gsl_blas
    }

    # Set MKL threads if requested
    if mkl_threads is not None:
        if not isinstance(mkl_threads, int) or mkl_threads < 0:
            logger.warning(f"Invalid value for mkl_threads ({mkl_threads}), must be non-negative integer. Ignoring.")
        else:
            MKL_NUM_THREADS = mkl_threads # Update global setting track
            os.environ['MKL_NUM_THREADS'] = str(mkl_threads)
            if MKL_AVAILABLE:
                try:
                    # Use a temporary variable for previous to avoid shadowing
                    _prev = set_mkl_threads(mkl_threads)
                    logger.info(f"MKL threads set to {mkl_threads} (was {_prev})")
                except Exception as e:
                     logger.error(f"Failed to dynamically set MKL threads: {e}")
            else:
                logger.debug("MKL backend not available, MKL_NUM_THREADS environment variable set.")

    # Set OpenMP threads if requested
    if omp_threads is not None:
         if not isinstance(omp_threads, int) or omp_threads < 0:
            logger.warning(f"Invalid value for omp_threads ({omp_threads}), must be non-negative integer. Ignoring.")
         else:
            OMP_NUM_THREADS = omp_threads # Update global setting track
            os.environ['OMP_NUM_THREADS'] = str(omp_threads)
            logger.info(f"OMP_NUM_THREADS environment variable set to {omp_threads}.")
            # Attempt to set via specific backend functions if available
            if GSL_CYTHON_AVAILABLE:
                try: set_gsl_omp_threads(omp_threads)
                except Exception as e: logger.debug(f"Note: Error setting GSL OMP threads directly: {e}")
            if MKL_AVAILABLE:
                 try: set_mkl_omp_threads(omp_threads)
                 except Exception as e: logger.debug(f"Note: Error setting MKL OMP threads directly: {e}")

    # Set GSL BLAS threads if requested
    if gsl_blas_threads is not None:
         if not isinstance(gsl_blas_threads, int) or gsl_blas_threads < 0:
             logger.warning(f"Invalid value for gsl_blas_threads ({gsl_blas_threads}), must be non-negative int. Ignoring.")
         else:
             if GSL_CYTHON_AVAILABLE:
                 try:
                     _prev = set_gsl_blas_threads(gsl_blas_threads)
                     logger.info(f"GSL BLAS threads set to {gsl_blas_threads} (was {_prev})")
                 except Exception as e:
                     logger.error(f"Failed to set GSL BLAS threads: {e}")
             else:
                 logger.warning("Cannot set GSL BLAS threads: GSL backend not available.")

    return previous


# --- Module Initialization ---
def _initialize_module():
    """Initial setup when the module is imported."""
    available_list = get_available_backends()
    logger.info(f"Available backends: {available_list}")
    logger.info(f"Config: RIDGE_BACKEND='{BACKEND_PREFERENCE}', RIDGE_FALLBACK={BACKEND_FALLBACK_ENABLED}")
    logger.info(f"Config: MKL_NUM_THREADS={MKL_NUM_THREADS}, OMP_NUM_THREADS={OMP_NUM_THREADS}")

    # Apply initial thread settings based on environment vars
    # MKL setting
    if MKL_NUM_THREADS > 0 and MKL_AVAILABLE:
        try: set_mkl_threads(MKL_NUM_THREADS)
        except Exception as e: logger.error(f"Failed to initialize MKL threads from environment: {e}")
    # OMP setting (apply to GSL/MKL OMP if available)
    if OMP_NUM_THREADS > 0:
         if GSL_CYTHON_AVAILABLE:
              try: set_gsl_omp_threads(OMP_NUM_THREADS)
              except Exception: pass # Ignore errors, just try
         if MKL_AVAILABLE:
              try: set_mkl_omp_threads(OMP_NUM_THREADS)
              except Exception: pass # Ignore errors, just try

_initialize_module()