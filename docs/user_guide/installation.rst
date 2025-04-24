============
Installation
============

This guide explains how to install the RidgeInference package and set up its various backends for optimal performance.

Basic Installation
==================

You can install RidgeInference using pip:

.. code-block:: bash

    pip install ridge-inference

This will install the basic package with NumPy backend support.

Dependencies
============

RidgeInference has the following dependencies:

- **Required dependencies**:
  - numpy>=1.19.0
  - scipy>=1.5.0
  - pandas>=1.0.0

- **Optional dependencies**:
  - numba>=0.50.0 (for Numba acceleration)
  - cupy>=9.0.0 (for GPU acceleration)
  - matplotlib>=3.3.0 (for visualization)
  - seaborn>=0.11.0 (for visualization)

Installing with Optional Dependencies
=====================================

To install with all optional dependencies:

.. code-block:: bash

    pip install ridge-inference[all]

Or, to install with specific optional dependencies:

.. code-block:: bash

    # For Numba acceleration
    pip install ridge-inference[numba]
    
    # For GPU acceleration
    pip install ridge-inference[gpu]
    
    # For visualization
    pip install ridge-inference[viz]

Building from Source
====================

To build from source (for development or to use the C backend):

1. Clone the repository:

   .. code-block:: bash

       git clone https://github.com/username/ridge-inference.git
       cd ridge-inference

2. Install development dependencies:

   .. code-block:: bash

       pip install -e ".[dev]"

3. Build the C extension:

   .. code-block:: bash

       # Linux/macOS
       make

       # Windows
       python setup.py build_ext --inplace

C Backend Setup
===============

The C backend provides OpenMP-accelerated ridge regression. To use it:

1. Ensure you have a C compiler installed:
   - Linux: GCC
   - macOS: Clang via XCode Command Line Tools
   - Windows: Visual C++ Build Tools

2. Build the C extension as described above.

3. Verify the C backend is available:

   .. code-block:: python

       from ridge_inference.c_bindings import is_c_available
       print(is_c_available())  # Should return True

GPU Backend Setup
=================

To use the GPU backend:

1. Install CuPy with the appropriate CUDA version for your GPU:

   .. code-block:: bash

       # For CUDA 11.0
       pip install cupy-cuda110
       
       # For CUDA 11.2
       pip install cupy-cuda112
       
       # For CUDA 11.4+
       pip install cupy-cuda11x

2. Verify the GPU backend is available:

   .. code-block:: python

       from ridge_inference.core import CUPY_AVAILABLE
       print(CUPY_AVAILABLE)  # Should return True

Troubleshooting
===============

Common installation issues:

1. **C extension build fails**:
   
   - Ensure you have a C compiler installed
   - Check for OpenMP headers
   - On macOS, you may need to install OpenMP: ``brew install libomp``

2. **CuPy installation fails**:
   
   - Ensure your CUDA version matches the CuPy package
   - Verify you have CUDA drivers installed
   - Use the correct cupy package for your CUDA version

3. **Backend not available**:

   .. code-block:: python

       # Check all backends
       from ridge_inference.c_bindings import is_c_available
       from ridge_inference.core import NUMBA_AVAILABLE, CUPY_AVAILABLE
       
       print(f"C backend: {is_c_available()}")
       print(f"Numba backend: {NUMBA_AVAILABLE}")
       print(f"GPU backend: {CUPY_AVAILABLE}")

Environment Variables
=====================

RidgeInference respects the following environment variables:

- ``OMP_NUM_THREADS``: Controls number of threads for OpenMP parallelization (C backend)
- ``RIDGE_INFERENCE_DISABLE_GPU``: Set to "1" to disable GPU backend
- ``NUMBA_NUM_THREADS``: Controls number of threads for Numba
