=========
Dashboard
=========

The RidgeInference Performance Dashboard provides an interactive way to explore benchmark results and compare different backend implementations.

.. raw:: html

   <div class="dashboard-link">
     <a href="../ComparePy2R/dashboard/index.html" class="dashboard-button">
       Launch Performance Dashboard
     </a>
   </div>

Overview
========

The dashboard visualizes performance metrics collected from running the RidgeInference package with different:

* Backend implementations (Python, Numba, GPU, C)
* Dataset sizes and sparsity levels
* Number of permutations (n_rand)

This visualization helps users select the most appropriate backend for their specific use case and hardware configuration.

Dashboard Features
================

Interactive Charts
----------------

* **Performance by Backend**: Compare execution times across different backends
* **Scaling Charts**: Visualize how performance scales with dataset size
* **Memory Usage**: Track memory consumption for large datasets
* **GPU vs CPU**: Side-by-side comparison of GPU and CPU performance

Filtering and Selection
---------------------

Users can filter the dashboard by:

* Dataset name and size
* Backend implementation
* Matrix type (dense/sparse)
* Hardware configuration

Accessing the Dashboard
=====================

The dashboard is available in two ways:

1. **Online Version**: Access the latest version at our project website.

2. **Local Version**: Available in the package distribution at `ComparePy2R/dashboard/index.html`.

   To open the local dashboard:

   .. code-block:: bash

      # Navigate to the package directory
      cd path/to/ridge_inference
      
      # Open the dashboard in your browser
      python -m webbrowser ComparePy2R/dashboard/index.html

Technical Implementation
======================

The dashboard is built using:

* **Plotly.js**: For interactive visualizations
* **HTML/CSS/JavaScript**: For the user interface
* **JSON**: For storing benchmark data

Data for the dashboard is generated using the `generate_dashboard.py` script, which processes benchmark results and creates the necessary JSON files.

Contributing Benchmark Data
=========================

Users can contribute their own benchmark data to improve the dashboard's coverage:

1. Run benchmarks on your hardware using the provided scripts
2. Submit the results via pull request or issue on GitHub
3. The maintainers will review and incorporate valid benchmark data

This collaborative approach helps build a comprehensive picture of performance across different hardware configurations and use cases.
