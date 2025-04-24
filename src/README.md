# Ridge Inference C Implementation

This package provides a high-performance C implementation of ridge regression with both permutation testing and t-test significance testing. The implementation leverages the GNU Scientific Library (GSL) for linear algebra operations and OpenMP for parallel processing.

## How to Compile the C Library

1. First, compile the shared library:

```bash
cd src
make clean  # Start fresh
make
```

2. Copy the compiled library to where it can be found:

```bash
make install
```

This will copy `RidgeInf.so` to the parent directory where Python can find it.

## Prerequisites

If you don't have GSL (GNU Scientific Library) installed, you'll need to install it first:

### On Ubuntu/Debian:
```bash
sudo apt-get install libgsl-dev
```

### On CentOS/RHEL/Fedora:
```bash
sudo yum install gsl-devel
# or
sudo dnf install gsl-devel
```

### On macOS (using Homebrew):
```bash
brew install gsl
```

### On Windows:
For Windows, it's recommended to use Windows Subsystem for Linux (WSL) or MinGW with MSYS2. Through MSYS2, you can install GSL with:

```bash
pacman -S mingw-w64-x86_64-gsl
```

## Performance Optimization

The C implementation has been optimized for high performance with:

1. **Efficient Memory Management**: Minimizes copies and optimizes memory access patterns
2. **OpenMP Parallelization**: Utilizes multiple CPU cores where beneficial
3. **Improved Linear Algebra**: Optimizes matrix operations for ridge regression
4. **Enhanced Python Binding**: Reduces overhead between Python and C

### Setting OpenMP Threads

You can control the number of threads used by setting the `OMP_NUM_THREADS` environment variable:

```bash
export OMP_NUM_THREADS=8  # Use 8 threads
```

By default, the implementation will use a reasonable number of threads based on your system.

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

2. **Memory Errors With Large Datasets**
   
   For large matrices, you might encounter memory issues. In that case:
   * Reduce the number of permutations (n_rand)
   * Filter genes based on expression level
   * Use dense matrices on GPU or batched processing for large sample counts

3. **Slow Performance with C Backend**
   
   If the C backend is slower than expected:
   * Ensure OpenMP is properly detected and enabled
   * Check `OMP_NUM_THREADS` setting is appropriate for your system
   * Verify the data format (dense vs sparse) is optimal for the operation