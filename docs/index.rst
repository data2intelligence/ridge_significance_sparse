===================
RidgeInference
===================

**High-performance ridge regression with multiple backend implementations**

.. image:: https://img.shields.io/badge/Python-3.7%2B-blue
   :target: https://www.python.org/downloads/
   :alt: Python 3.7+

.. image:: https://img.shields.io/badge/License-MIT-green
   :target: https://opensource.org/licenses/MIT
   :alt: MIT License

**RidgeInference** is a Python package for performing efficient ridge regression analysis with significance testing, designed for large genomic datasets. It provides multiple backend implementations for optimal performance across different hardware configurations.

Features
========

- Fast ridge regression with permutation-based significance testing
- Multiple backend implementations:
  - Pure Python (NumPy)
  - Numba-accelerated
  - GPU-accelerated (CuPy)
  - C/OpenMP-accelerated
- Batch processing for memory-efficient handling of large datasets
- Support for sparse matrices
- Inference utilities for genomic data analysis

Interactive Dashboard
=====================

RidgeInference comes with an interactive dashboard for visualizing performance comparisons across different backend implementations:

.. raw:: html

   <div class="dashboard-link">
     <a href="../ComparePy2R/dashboard/index.html" class="dashboard-button">
       Launch Performance Dashboard
     </a>
   </div>

.. image:: _static/dashboard_preview.png
   :width: 600px
   :alt: Dashboard Preview
   :target: ../ComparePy2R/dashboard/index.html

The dashboard provides:

- Comparative benchmarks across different backends
- Performance metrics for various dataset sizes
- Execution time and memory usage statistics

Installation
============

.. code-block:: bash

   pip install ridge-inference

Quick Start
===========

.. code-block:: python

   from ridge_inference import ridge
   import numpy as np
   
   # Create sample data
   X = np.random.randn(1000, 10)  # 1000 observations, 10 features
   Y = np.random.randn(1000, 5)   # 5 samples
   
   # Run ridge regression with significance testing
   result = ridge(X, Y, lambda_=1000, n_rand=100)
   
   # Access results
   beta = result['beta']        # Coefficient matrix
   pvalue = result['pvalue']    # P-values

Documentation Contents
======================

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   
   user_guide/installation
   user_guide/backends
   user_guide/batch_processing
   user_guide/ridge_inference
   user_guide/ridge_math
   user_guide/performance_tips

.. toctree::
   :maxdepth: 1
   :caption: Examples
   
   examples/basic_usage
   examples/advanced_usage
   examples/visualization
   
.. toctree::
   :maxdepth: 2
   :caption: API Reference
   
   api/ridge
   api/batch
   api/inference
   api/c_bindings
   api/core
   api/logit
   api/utils

.. toctree::
   :maxdepth: 1
   :caption: Development
   
   modules
   dashboard
   
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
