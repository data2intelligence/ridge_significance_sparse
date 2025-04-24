# Ridge Inference C Implementation

This package provides a high-performance C implementation of ridge regression with both permutation testing and t-test significance testing. The implementation offers two backend options:

1. **GNU Scientific Library (GSL)** - Open-source scientific computing library
2. **Intel Math Kernel Library (MKL)** - High-performance math routines optimized for Intel processors

Both backends leverage OpenMP for parallel processing to maximize performance on multi-core systems.

## How to Compile the C Library

### Building with Default Settings

```bash
cd src
make clean  # Start fresh
make        # Builds all available backends
```

### Building Specific Backends

```bash
# Build only the GSL backend
make BACKEND=gsl

# Build only the MKL backend
make BACKEND=mkl MKLROOT=/path/to/mkl

# Build all available backends (default)
make BACKEND=all
```

### Running Tests

```bash
# Test GSL implementation
make test_gsl

# Test MKL implementation
make test_mkl MKLROOT=/path/to/mkl
```

### Installation

After compilation, install the library where Python can find it:

```bash
make install
```

This will copy the shared libraries (`RidgeInf.so` and backend-specific libraries) to the parent directory.

## Prerequisites

### GNU Scientific Library (GSL)

#### On Ubuntu/Debian:
```bash
sudo apt-get install libgsl-dev
```

#### On CentOS/RHEL/Fedora:
```bash
sudo yum install gsl-devel
# or
sudo dnf install gsl-devel
```

#### On macOS (using Homebrew):
```bash
brew install gsl
```

#### On Windows:
For Windows, it's recommended to use Windows Subsystem for Linux (WSL) or MinGW with MSYS2. Through MSYS2, you can install GSL with:

```bash
pacman -S mingw-w64-x86_64-gsl
```

### Intel Math Kernel Library (MKL)

#### Using oneAPI (Recommended):
```bash
# On Ubuntu/Debian
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt update
sudo apt install intel-oneapi-mkl intel-oneapi-mkl-devel

# On RHEL/CentOS
sudo yum-config-manager --add-repo https://yum.repos.intel.com/oneapi
sudo rpm --import https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
sudo yum install intel-oneapi-mkl intel-oneapi-mkl-devel
```

#### On macOS (using Homebrew):
```bash
brew install intel-mkl
```

#### Setting up environment variables:
After installing MKL, you need to set the `MKLROOT` environment variable:

```bash
# For oneAPI installation
source /opt/intel/oneapi/setvars.sh
# or add to .bashrc for persistence
echo 'source /opt/intel/oneapi/setvars.sh > /dev/null' >> ~/.bashrc
```

## Performance Optimization

The C implementation has been optimized for high performance with:

1. **Efficient Memory Management**: Minimizes copies and optimizes memory access patterns
2. **OpenMP Parallelization**: Utilizes multiple CPU cores where beneficial
3. **Improved Linear Algebra**: 
   - GSL backend: Optimized matrix operations for ridge regression
   - MKL backend: Leverages Intel's highly optimized BLAS/LAPACK implementations
4. **Enhanced Python Binding**: Reduces overhead between Python and C
5. **Backend-specific optimizations**:
   - GSL: Custom parallel matrix multiplication implementation
   - MKL: Sparse matrix support for efficiently handling large, sparse datasets

### Setting Thread Count

Both backends support controlling thread count:

#### GSL with OpenMP:
```bash
export OMP_NUM_THREADS=8  # Use 8 threads for OpenMP operations
```

#### MKL:
MKL thread count can be controlled programmatically using the provided API:
```c
// From C code
int prev_threads = ridge_mkl_set_threads(8); // Set to 8 threads
int current_threads = ridge_mkl_get_threads(); // Get current thread count
```

Or via environment variables:
```bash
export MKL_NUM_THREADS=8  # Use 8 threads for MKL operations
```

By default, both implementations will use a reasonable number of threads based on your system.

## Matrix Operation in Ridge Regression

The algorithm expects:
* X: Matrix of shape (n_genes, n_features)
* Y: Matrix of shape (n_genes, n_samples)

The C implementation then calculates:
* XtX = X'X
* T = (X'X + λI)^(-1) X'
* beta = T Y
* Then permutation testing or t-test for significance

## Common Issues and Solutions

1. **GSL Library Not Found**

   ```
   error: GSL library not found
   ```

   Solution: Install GSL development packages for your OS.

2. **MKL Library Not Found**

   ```
   warning: MKLROOT provided, but mkl.h not found
   ```

   Solution: Make sure the MKLROOT environment variable points to your MKL installation directory.

3. **Memory Errors With Large Datasets**
   
   For large matrices, you might encounter memory issues. In that case:
   * Reduce the number of permutations (n_rand)
   * Filter genes based on expression level
   * Use sparse matrices with MKL backend for large sparse datasets
   * Use batched processing for large sample counts

4. **Slow Performance**
   
   If performance is slower than expected:
   * Ensure OpenMP is properly detected and enabled
   * Check thread count settings are appropriate for your system
   * Verify you're using the optimal backend for your hardware (MKL for Intel CPUs)
   * Consider using sparse matrices for sparse data

5. **Numerical Stability Issues**

   For numerically challenging datasets:
   * Increase the regularization parameter (lambda)
   * Pre-scale/normalize your input data
   * MKL backend may provide better numerical stability for ill-conditioned problems

## Choosing Between Backends

* **GSL Backend**: Good general-purpose choice, works on all platforms, easier dependencies
* **MKL Backend**: Superior performance on Intel CPUs, better for very large datasets, includes sparse matrix support

The choice between GSL and MKL depends on your specific hardware, dataset characteristics, and performance requirements.