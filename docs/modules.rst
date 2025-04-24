=======
Modules
=======

RidgeInference consists of several modules that work together to provide efficient ridge regression functionality. This page gives an overview of each module's purpose and how they interact.

Module Architecture
-------------------

.. graphviz::

   digraph RidgeInference {
       rankdir=TB;
       node [shape=box, style=filled, fillcolor=lightblue];
       
       User [label="User Code", fillcolor=white];
       
       ridge [label="ridge.py\n(Main API)"];
       batch [label="batch.py\n(Batch Processing)"];
       inference [label="inference.py\n(High-level API)"];
       
       core [label="core.py\n(Core Implementations)"];
       c_bindings [label="c_bindings.py\n(C Interface)"];
       logit [label="logit.py\n(Logistic Regression)"];
       utils [label="utils.py\n(Utilities)"];
       
       User -> inference;
       User -> ridge;
       User -> batch;
       
       inference -> ridge;
       inference -> batch;
       inference -> utils;
       
       ridge -> core;
       ridge -> c_bindings;
       
       batch -> core;
       batch -> c_bindings;
   }

Core Modules
------------

.. toctree::
   :maxdepth: 1
   
   ridge_inference.ridge
   ridge_inference.batch
   ridge_inference.inference
   ridge_inference.core
   ridge_inference.c_bindings
   ridge_inference.logit
   ridge_inference.utils

Module Details
--------------

ridge.py
~~~~~~~

The main interface module providing the :func:`ridge` function, which serves as the primary entry point for ridge regression. It handles backend selection, input validation, and fallback mechanisms.

batch.py
~~~~~~~

Implements memory-efficient batch processing through the :func:`ridge_batch` function. This module enables processing of large datasets by handling the Y matrix in smaller chunks.

inference.py
~~~~~~~~~~

Provides high-level functions for genomic data analysis:

- :func:`secact_inference`: Core inference function
- :func:`secact_activity_inference`: Higher-level wrapper with differential expression calculation

core.py
~~~~~~

Contains the core computational implementations for different backends:

- NumPy backend
- Numba backend
- CuPy/GPU backend

Each backend implements the same functionality with different optimizations.

c_bindings.py
~~~~~~~~~~~

Provides Python bindings to the C implementation of ridge regression. Handles loading the shared library, mapping Python arrays to C data structures, and error handling.

logit.py
~~~~~~

Implements logistic regression with significance testing. Includes Firth correction to handle separation issues.

utils.py
~~~~~~

Utility functions for data loading, preprocessing, visualization, and other common operations.

Code Organization
-----------------

The package follows a layered architecture:

1. **High-level API** (inference.py)
   - Domain-specific functions for genomic data analysis
   - Handles preprocessing and result formatting

2. **Mid-level API** (ridge.py, batch.py)
   - General-purpose ridge regression
   - Backend selection and orchestration

3. **Low-level implementation** (core.py, c_bindings.py)
   - Optimized computational kernels
   - Backend-specific code

Development Notes
-----------------

- The C extension is built during installation
- Backend availability is checked at runtime
- Appropriate fallbacks are provided if preferred backends are unavailable
- NumPy backend always serves as the final fallback

See also the :ref:`genindex` and :ref:`modindex`.
