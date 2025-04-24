# ridge_inference/loaders.py
import pandas as pd
import numpy as np
from scipy import sparse as sps
from pathlib import Path
import logging
import warnings
import gzip # Import gzip for peeking
import os
import json

# Optional dependencies - import lazily or add checks
try: import anndata
except ImportError: anndata = None; warnings.warn("anndata not installed, .h5ad loading disabled.")
try: import zarr
except ImportError: zarr = None; warnings.warn("zarr not installed, .zarr loading disabled.")
try: import pyarrow
except ImportError: pyarrow = None; warnings.warn("pyarrow not installed, .feather loading might fail.")
try: import tiledb
except ImportError: tiledb = None; warnings.warn("tiledb not installed, .tdb loading disabled.")
try: import tiledbsoma
except ImportError: tiledbsoma = None; warnings.warn("tiledbsoma not installed, .soma loading disabled.")
try: 
    from scipy.io import mmread, mmwrite
    SCIPY_IO_AVAILABLE = True
except ImportError: 
    SCIPY_IO_AVAILABLE = False
    warnings.warn("scipy.io mmread/mmwrite not available, .mtx loading disabled.")

logger = logging.getLogger(__name__)

def load_matrix_data(file_path):
    """
    Loads matrix data from various file formats including H5AD (AnnData), TileDB, TileDB-SOMA, and MTX.
    Includes robust index finding and handles appropriate format-specific transposition.
    
    Parameters:
    -----------
    file_path : str or Path
        Path to the file to load
        
    Returns:
    --------
    tuple
        (matrix, row_names, col_names) where matrix is numpy ndarray or sparse matrix
    """
    path = Path(file_path).resolve()
    if not path.exists(): raise FileNotFoundError(f"Input file not found: {path}")
    
    # Improved file type detection - examine the filename directly
    filename = path.name.lower()
    logger.info(f"Loading data from: {filename}...")
    
    matrix, row_names, col_names = None, None, None
    df = None
    
    # Determine file type by checking endings in priority order
    if filename.endswith('.feather'):
        file_type = '.feather'
    elif filename.endswith('.npz'):
        file_type = '.npz'
    elif filename.endswith('.h5ad'):
        file_type = '.h5ad'
    elif filename.endswith('.zarr'):
        file_type = '.zarr'
    elif filename.endswith('.mtx') or filename.endswith('.mtx.gz'):
        file_type = '.mtx'
    elif filename.endswith('.tdb') or (path.is_dir() and (path / '.tdb').exists()):
        file_type = '.tdb'  # TileDB array
    elif filename.endswith('.soma') or (path.is_dir() and (path / '.soma').exists()): 
        file_type = '.soma'  # TileDB-SOMA array
    elif filename.endswith('.csv.gz'):
        file_type = '.csv.gz'
    elif filename.endswith('.tsv.gz'):
        file_type = '.tsv.gz'
    elif filename.endswith('.txt.gz'):
        file_type = '.txt.gz'
    elif filename.endswith('.csv'):
        file_type = '.csv'
    elif filename.endswith('.tsv'):
        file_type = '.tsv'
    elif filename.endswith('.txt'):
        file_type = '.txt'
    elif filename.endswith('.gz'):
        # Plain .gz file - most likely tab-separated in bioinformatics
        file_type = '.tsv.gz'
        logger.info(f"File {filename} has .gz extension without format indicator - assuming tab-separated data")
    else:
        raise ValueError(f"Unsupported file type: {filename}")
    
    logger.info(f"Detected file type: {file_type}")

    try:
        if file_type == '.feather':
            # Feather handling code remains the same
            if pyarrow is None: raise ImportError("pyarrow needed for .feather files.")
            df = pd.read_feather(path)
            if not (isinstance(df.index, pd.Index) and not pd.api.types.is_numeric_dtype(df.index) and df.index.name is not None):
                potential_index_cols = ['index', 'Unnamed: 0', 'gene', 'genes', 'gene_id', 'gene_symbol', 'Symbol', 'ID', 'Name']
                index_col_name = None
                for col in potential_index_cols:
                    if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]): index_col_name = col; break
                if index_col_name: logger.info(f"Setting '{index_col_name}' as index."); df = df.set_index(index_col_name)
                else: raise ValueError("Cannot identify non-numeric index column in Feather file.")
            row_names = df.index.astype(str).tolist(); col_names = df.columns.astype(str).tolist()
            matrix = df.values.astype(np.float64)
        
        elif file_type == '.npz':
            # NPZ handling code remains the same
            loaded = sps.load_npz(path)
            if isinstance(loaded, sps.spmatrix): matrix = loaded.tocsr()
            else: raise ValueError(f"NPZ file {path.name} did not contain a SciPy sparse matrix.")
            logger.warning(f"Loaded sparse matrix from {path.name}. Row/Col names not stored in NPZ, using placeholders.")
            row_names = [f"Row_{i}" for i in range(matrix.shape[0])]; col_names = [f"Col_{i}" for i in range(matrix.shape[1])]
        
        elif file_type == '.mtx':
            # Matrix Market format handling
            if not SCIPY_IO_AVAILABLE:
                raise ImportError("scipy.io module needed for .mtx file loading")
            
            # Load the sparse matrix
            logger.info(f"Loading MTX sparse matrix from {path}")
            matrix = mmread(path)
            
            # Ensure it's in CSR format for efficient operations
            if sps.issparse(matrix):
                matrix = matrix.tocsr()
            
            # By convention, MTX files from single-cell data have cells as columns and genes as rows
            # Most implementations follow this, so we don't transpose
            
            # Look for row and column names in accompanying files
            parent_dir = path.parent
            base_name = path.stem
            if base_name.endswith('.mtx'):  # Handle .mtx.gz case
                base_name = base_name[:-4]
            
            # Common file patterns for row names (genes)
            row_file_patterns = [
                f"{base_name}.genes.tsv", f"{base_name}.genes.tsv.gz",
                f"{base_name}.genes.txt", f"{base_name}.genes.txt.gz",
                f"{base_name}.features.tsv", f"{base_name}.features.tsv.gz",
                f"{base_name}_genes.tsv", f"{base_name}_features.tsv",
                f"{base_name}_genes.txt", f"{base_name}_features.txt",
                "genes.tsv", "features.tsv", "genes.txt", "features.txt",
                "rows.tsv", "rows.txt"
            ]
            
            # Common file patterns for column names (cells/barcodes)
            col_file_patterns = [
                f"{base_name}.barcodes.tsv", f"{base_name}.barcodes.tsv.gz",
                f"{base_name}.barcodes.txt", f"{base_name}.barcodes.txt.gz",
                f"{base_name}.cells.tsv", f"{base_name}.cells.tsv.gz",
                f"{base_name}_barcodes.tsv", f"{base_name}_cells.tsv",
                f"{base_name}_barcodes.txt", f"{base_name}_cells.txt",
                "barcodes.tsv", "cells.tsv", "barcodes.txt", "cells.txt",
                "columns.tsv", "columns.txt"
            ]
            
            # Try to find and load row names
            row_file = None
            for pattern in row_file_patterns:
                potential_file = parent_dir / pattern
                if potential_file.exists():
                    row_file = potential_file
                    break
            
            if row_file is not None:
                try:
                    compression = 'gzip' if str(row_file).endswith('.gz') else None
                    row_df = pd.read_csv(row_file, sep='\t', header=None, compression=compression)
                    if row_df.shape[1] >= 1:
                        # Use first column as names
                        row_names = row_df.iloc[:, 0].astype(str).tolist()
                        logger.info(f"Loaded {len(row_names)} row names from {row_file.name}")
                    else:
                        row_names = [f"Row_{i}" for i in range(matrix.shape[0])]
                        logger.warning(f"Row file {row_file.name} has no columns, using default row names")
                except Exception as e:
                    logger.warning(f"Failed to load row names from {row_file}: {e}")
                    row_names = [f"Row_{i}" for i in range(matrix.shape[0])]
            else:
                logger.warning(f"No row names file found, using default row names")
                row_names = [f"Row_{i}" for i in range(matrix.shape[0])]
            
            # Try to find and load column names
            col_file = None
            for pattern in col_file_patterns:
                potential_file = parent_dir / pattern
                if potential_file.exists():
                    col_file = potential_file
                    break
            
            if col_file is not None:
                try:
                    compression = 'gzip' if str(col_file).endswith('.gz') else None
                    col_df = pd.read_csv(col_file, sep='\t', header=None, compression=compression)
                    if col_df.shape[1] >= 1:
                        # Use first column as names
                        col_names = col_df.iloc[:, 0].astype(str).tolist()
                        logger.info(f"Loaded {len(col_names)} column names from {col_file.name}")
                    else:
                        col_names = [f"Col_{i}" for i in range(matrix.shape[1])]
                        logger.warning(f"Column file {col_file.name} has no columns, using default column names")
                except Exception as e:
                    logger.warning(f"Failed to load column names from {col_file}: {e}")
                    col_names = [f"Col_{i}" for i in range(matrix.shape[1])]
            else:
                logger.warning(f"No column names file found, using default column names")
                col_names = [f"Col_{i}" for i in range(matrix.shape[1])]
            
            # Check for dimension mismatch
            if len(row_names) != matrix.shape[0]:
                logger.warning(f"Row names count ({len(row_names)}) doesn't match matrix rows ({matrix.shape[0]}). Using defaults.")
                row_names = [f"Row_{i}" for i in range(matrix.shape[0])]
            
            if len(col_names) != matrix.shape[1]:
                logger.warning(f"Column names count ({len(col_names)}) doesn't match matrix columns ({matrix.shape[1]}). Using defaults.")
                col_names = [f"Col_{i}" for i in range(matrix.shape[1])]
            
            logger.info(f"Loaded MTX matrix with shape {matrix.shape} and density {matrix.nnz/(matrix.shape[0]*matrix.shape[1]):.6f}")
        
        elif file_type == '.h5ad':
            # Enhanced H5AD handling with better error recovery
            if anndata is None: raise ImportError("anndata needed for .h5ad files.")
            
            try:
                # Load AnnData object
                adata = anndata.read_h5ad(path)
                
                # Get expression matrix (X)
                matrix = adata.X
                
                # Check matrix type and convert if needed
                if not isinstance(matrix, (np.ndarray, sps.spmatrix)):
                    try:
                        matrix = matrix.toarray()
                    except AttributeError:
                        raise ValueError(f"Unsupported matrix type in AnnData: {type(matrix)}. Cannot convert to array.")
                
                # Ensure sparse matrices are in CSR format for efficient operations
                if sps.issparse(matrix):
                    matrix = matrix.tocsr()
                
                # Get cell/sample names and feature/gene names
                if hasattr(adata, 'obs_names') and hasattr(adata, 'var_names'):
                    obs_names = adata.obs_names.astype(str).tolist()
                    var_names = adata.var_names.astype(str).tolist()
                    
                    # Since AnnData has cells in rows and genes in columns, transpose to match
                    # our convention (genes in rows, cells in columns)
                    matrix = matrix.T
                    row_names, col_names = var_names, obs_names
                    logger.info(f"Transposed AnnData matrix (shape after transpose: {matrix.shape}) to put genes in rows.")
                else:
                    # Fallback if names aren't available
                    row_names = [f"Row_{i}" for i in range(matrix.shape[0])]
                    col_names = [f"Col_{i}" for i in range(matrix.shape[1])]
                    logger.warning("Could not find obs_names or var_names in AnnData, using default row/column names.")
                
                # Check for layer data
                if hasattr(adata, 'layers') and len(adata.layers) > 0:
                    logger.info(f"Note: AnnData contains {len(adata.layers)} additional layers that aren't being loaded.")
                
                logger.info(f"Loaded AnnData h5ad file: matrix shape: {matrix.shape}")
                
            except Exception as e:
                logger.error(f"Failed to load AnnData h5ad file: {e}", exc_info=True)
                raise ValueError(f"Failed to load AnnData h5ad file: {e}") from e
        
        elif file_type == '.zarr':
            # Zarr handling code remains the same
            if zarr is None: raise ImportError("zarr needed for .zarr files.")
            za = zarr.open(path, mode='r')
            if isinstance(za, zarr.Array): matrix = za[:]
            elif isinstance(za, zarr.Group) and 'X' in za: matrix = za['X'][:]
            else: raise ValueError("Unsupported Zarr structure. Expecting array or group with 'X'.")
            if isinstance(za, zarr.Array):
                 logger.warning("Loaded Zarr array. Row/Col names not stored, using placeholders.")
                 row_names = [f"Row_{i}" for i in range(matrix.shape[0])]; col_names = [f"Col_{i}" for i in range(matrix.shape[1])]
            else: # It's a Group, try to load names and transpose
                 try:
                     obs_names = za['obs_names'][:] if 'obs_names' in za else [f"Col_{i}" for i in range(matrix.shape[1])]
                     var_names = za['var_names'][:] if 'var_names' in za else [f"Row_{i}" for i in range(matrix.shape[0])]
                     matrix = matrix.T; row_names, col_names = var_names, obs_names
                     logger.info("Transposed Zarr 'X' group matrix to Genes x Samples.")
                 except Exception as e_zarr_meta:
                      logger.warning(f"Could not load names/transpose Zarr group, using placeholders/original shape. Error: {e_zarr_meta}")
                      row_names = [f"Row_{i}" for i in range(matrix.shape[0])]; col_names = [f"Col_{i}" for i in range(matrix.shape[1])]
            if sps.issparse(matrix): matrix = matrix.tocsr()
            
        elif file_type == '.tdb':
            # TileDB handling 
            if tiledb is None: raise ImportError("tiledb package needed for .tdb files.")
            
            # Check if path is directory (TileDB array) or a file (URI)
            if path.is_dir():
                tdb_path = str(path)
            else:
                tdb_path = str(path)  # Handle as URI
            
            # Open TileDB array
            with tiledb.open(tdb_path, 'r') as A:
                # Extract data
                if '__data' in A.meta:
                    # If there's a specific data attribute
                    matrix = A[:]['__data']
                else:
                    # Get first attribute as default
                    attrs = A.schema.attrs
                    if len(attrs) == 0:
                        raise ValueError(f"No attributes found in TileDB array")
                    first_attr = attrs[0].name
                    matrix = A[:][first_attr]
                
                # Extract dimension names for rows and columns
                dims = A.schema.domain.dimensions
                if len(dims) != 2:
                    raise ValueError(f"Expected 2D TileDB array, got {len(dims)}D array")
                
                # Extract row and column names from dimensions if possible
                try:
                    row_names = list(dims[0].values())
                    col_names = list(dims[1].values())
                except:
                    # Fallback if dimensions don't have labels
                    row_names = [f"Row_{i}" for i in range(matrix.shape[0])]
                    col_names = [f"Col_{i}" for i in range(matrix.shape[1])]
                
                # Check if names are stored in metadata
                if '__row_names' in A.meta:
                    row_names = json.loads(A.meta['__row_names'])
                if '__col_names' in A.meta:
                    col_names = json.loads(A.meta['__col_names'])
                
                # Convert matrix to numpy array or sparse matrix as needed
                if isinstance(matrix, np.ndarray):
                    if matrix.ndim != 2:
                        raise ValueError(f"Expected 2D matrix from TileDB, got {matrix.ndim}D array")
                else:
                    # Try to convert to numpy array
                    try:
                        matrix = np.array(matrix)
                    except:
                        # If conversion fails, raise error
                        raise ValueError(f"Failed to convert TileDB data to numpy array")
                
                logger.info(f"Loaded TileDB array: shape={matrix.shape}")
                
        elif file_type == '.soma':
            # TileDB-SOMA handling
            if tiledbsoma is None: raise ImportError("tiledbsoma package needed for .soma files.")
            
            # Open SOMA collection or experiment
            try:
                # Try opening as experiment first (common for single-cell data)
                exp = tiledbsoma.Experiment(str(path))
                # Get the measurement data
                ms = exp.ms
                # Get the X matrix (expression)
                if hasattr(ms, 'X'):
                    matrix = ms.X.read().to_numpy()
                    # Get row names (genes/features)
                    var_df = ms.var.read()
                    if 'feature_name' in var_df.columns:
                        row_names = var_df['feature_name'].tolist()
                    else:
                        row_names = var_df.index.tolist()
                    
                    # Get column names (cells/samples)
                    obs_df = ms.obs.read()
                    col_names = obs_df.index.tolist()
                else:
                    raise ValueError("No 'X' matrix found in SOMA experiment")
            except:
                # If not an experiment, try as collection
                try:
                    collection = tiledbsoma.Collection(str(path))
                    # Find the first array or matrix in collection
                    for key in collection:
                        obj = collection[key]
                        if isinstance(obj, (tiledbsoma.Array, tiledbsoma.SOMADenseNDArray, 
                                          tiledbsoma.SOMASparseNDArray)):
                            # Found an array, extract it
                            matrix_data = obj.read()
                            if hasattr(matrix_data, 'to_numpy'):
                                matrix = matrix_data.to_numpy()
                            else:
                                matrix = np.array(matrix_data)
                            
                            # Try to get dimension names
                            if hasattr(obj, 'dim_names'):
                                dims = obj.dim_names()
                                if len(dims) == 2:
                                    # Get values for dimensions if available
                                    try:
                                        row_dim = dims[0]
                                        col_dim = dims[1]
                                        row_names = obj.dim_values(row_dim).tolist()
                                        col_names = obj.dim_values(col_dim).tolist()
                                    except:
                                        # Fallback if can't get dimension values
                                        row_names = [f"Row_{i}" for i in range(matrix.shape[0])]
                                        col_names = [f"Col_{i}" for i in range(matrix.shape[1])]
                                else:
                                    # Default row/col names
                                    row_names = [f"Row_{i}" for i in range(matrix.shape[0])]
                                    col_names = [f"Col_{i}" for i in range(matrix.shape[1])]
                            else:
                                # Default row/col names
                                row_names = [f"Row_{i}" for i in range(matrix.shape[0])]
                                col_names = [f"Col_{i}" for i in range(matrix.shape[1])]
                            
                            break
                    else:
                        raise ValueError("No arrays found in SOMA collection")
                except Exception as e:
                    raise ValueError(f"Failed to load SOMA collection: {e}")
            
            # Handle sparse matrices
            if hasattr(matrix, 'todense'):
                matrix = matrix.tocsr()
            
            logger.info(f"Loaded SOMA data: shape={matrix.shape}")
        
        elif file_type in ['.csv', '.tsv', '.txt', '.csv.gz', '.tsv.gz', '.txt.gz']:
            # Determine compression and separator
            compression = 'gzip' if file_type.endswith('.gz') else None
            
            if '.csv' in file_type:
                sep = ','
            elif '.tsv' in file_type or '.tab' in file_type:
                sep = '\t'
            else:
                # For .txt or ambiguous files, try to detect separator
                sep = None
            
            # If separator still not determined, peek at file content for compressed files
            if sep is None and compression == 'gzip':
                try:
                    import gzip
                    with gzip.open(path, 'rt', errors='ignore') as f:
                        first_line = f.readline().strip()
                    tabs = first_line.count('\t')
                    commas = first_line.count(',')
                    
                    if tabs > 0 and (tabs >= commas or commas == 0):
                        sep = '\t'
                        logger.debug(f"Detected tab separator from content (tabs={tabs}, commas={commas})")
                    elif commas > 0:
                        sep = ','
                        logger.debug(f"Detected comma separator from content (tabs={tabs}, commas={commas})")
                    else:
                        # Default to tab for bioinformatics data
                        sep = '\t'
                        logger.debug("No clear separator in content, defaulting to tab for bioinformatics data")
                except Exception as peek_err:
                    logger.warning(f"Failed to peek at file content: {peek_err}. Defaulting to tab separator.")
                    sep = '\t'  # Default for bioinformatics data
            elif sep is None:
                # Default to tab for text files with unclear separator
                sep = '\t'
                logger.debug("Using default tab separator for text file")
            
            logger.info(f"Using separator: '{sep}' for text file")
            
            # IMPROVED: First try loading with index_col=0 (common in bioinformatics)
            try:
                logger.debug("Attempting to load with first column as index (index_col=0)")
                df = pd.read_csv(path, sep=sep, compression=compression, index_col=0)
                
                # Check if the index is appropriate (not numeric for gene IDs)
                if pd.api.types.is_numeric_dtype(df.index) and df.shape[1] > 0:
                    logger.warning("First column is numeric, not ideal for gene IDs. Trying alternative loading.")
                    df = None  # Reset to try alternative approach
                else:
                    logger.info("Successfully loaded with first column as index.")
            except Exception as e_idx0:
                logger.warning(f"Failed to load with index_col=0: {e_idx0}. Trying alternative approach.")
                df = None
            
            # If the above failed, try the standard approach
            if df is None:
                logger.debug("Attempting to load without specifying an index column")
                df = pd.read_csv(path, sep=sep, compression=compression, engine='python' if sep is None else 'c')
                
                # Try to identify the index column
                potential_index_cols = ['index', 'Unnamed: 0', 'gene', 'genes', 'gene_id', 'gene_symbol', 'Symbol', 'ID', 'Name']
                ordered_cols = [col for col in potential_index_cols if col in df.columns] + [col for col in df.columns if col not in potential_index_cols]
                index_col_name = None
                
                for col_name in ordered_cols:
                    if col_name in df.columns and not pd.api.types.is_numeric_dtype(df[col_name]): 
                        index_col_name = col_name
                        break
                        
                if index_col_name: 
                    logger.info(f"Setting '{index_col_name}' as index.")
                    df = df.set_index(index_col_name)
                else: 
                    # IMPROVED: Check first column even if it doesn't match standard names
                    if df.shape[1] > 0 and not pd.api.types.is_numeric_dtype(df.iloc[:, 0]):
                        first_col = df.columns[0]
                        logger.info(f"Using first column '{first_col}' as index since it contains non-numeric values.")
                        df = df.set_index(first_col)
                    else:
                        # Last resort: try again with index_col=0 and force it to string type
                        try:
                            logger.debug("Trying once more with index_col=0 and converting to string")
                            df = pd.read_csv(path, sep=sep, compression=compression, index_col=0, dtype={0: str})
                            logger.info("Successfully loaded with first column as string index.")
                        except Exception as e_last:
                            logger.error(f"All index detection methods failed: {e_last}")
                            # If we can't find a suitable index, create a dummy one as a last resort
                            df = pd.read_csv(path, sep=sep, compression=compression)
                            logger.warning("Creating dummy row indices as a last resort")
                            df.index = [f"Row_{i}" for i in range(df.shape[0])]
            
            # Final DataFrame processing
            row_names = df.index.astype(str).tolist()
            col_names = df.columns.astype(str).tolist()
            
            # DEBUG: Print sample rows/cols to aid in debugging
            logger.debug(f"Index sample: {df.index[:5]}")
            logger.debug(f"Columns sample: {df.columns[:5]}")
            
            matrix = df.values.astype(np.float64)
        
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        # Final validation
        if matrix is None or row_names is None or col_names is None: 
            raise ValueError("Matrix or names loading failed.")
        if not isinstance(matrix, (np.ndarray, sps.spmatrix)): 
            raise ValueError(f"Loaded matrix type error: {type(matrix)}")
        if matrix.ndim != 2: 
            raise ValueError(f"Loaded matrix not 2D: shape={matrix.shape}")
        if matrix.shape[0] != len(row_names): 
            raise ValueError(f"Matrix rows ({matrix.shape[0]}) != row names ({len(row_names)}).")
        if matrix.shape[1] != len(col_names): 
            raise ValueError(f"Matrix columns ({matrix.shape[1]}) != col names ({len(col_names)}).")
        
        logger.info(f"Successfully loaded matrix: shape={matrix.shape}, type={type(matrix).__name__}")
        return matrix, row_names, col_names
    
    except Exception as e:
        logger.error(f"Failed to load data from {path.name}: {type(e).__name__} - {e}", exc_info=True)
        return None, None, None

# --- load_dataset (Keep robust version from before) ---
def load_dataset(dataset_file: Path) -> pd.DataFrame:
    """
    Load expression dataset (Y) from feather or text formats into DataFrame.
    Includes robust index finding and handling for compressed files.
    """
    logger = logging.getLogger("ridge_inference.load_dataset")
    dataset_file = Path(dataset_file).resolve(); logger.info(f"Loading DataFrame from: {dataset_file.name}")
    if not dataset_file.exists(): raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

    df = None; suffix = "".join(dataset_file.suffixes).lower()

    try:
        if suffix == '.feather':
            if pyarrow is None: raise ImportError("pyarrow needed for .feather files.")
            df = pd.read_feather(dataset_file)
            # Robust Index Setting for Feather
            if not (isinstance(df.index, pd.Index) and not pd.api.types.is_numeric_dtype(df.index) and df.index.name is not None):
                potential_index_cols = ['index', 'Unnamed: 0', 'gene', 'genes', 'gene_id', 'gene_symbol', 'Symbol', 'ID', 'Name']
                index_col_name = None
                for col in potential_index_cols:
                    if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]): index_col_name = col; break
                if index_col_name: logger.info(f"Setting '{index_col_name}' as index."); df = df.set_index(index_col_name)
                else: logger.warning("Could not auto-identify non-numeric index column in Feather file. Using existing index.")

        # Modified condition to also handle plain .gz files
        elif suffix in ['.csv', '.tsv', '.txt', '.csv.gz', '.tsv.gz', '.txt.gz', '.tab.gz'] or suffix == '.gz':
            # Handle compression
            compression = None
            if suffix.endswith('.gz'): compression = 'gzip'
            elif suffix.endswith('.bz2'): compression = 'bz2'
            elif suffix.endswith('.zip'): compression = 'zip'
            elif suffix.endswith('.xz'): compression = 'xz'
            
            # --- Separator Detection Logic ---
            separator = None; filename_no_comp = dataset_file.name
            
            # First strip compression extension if present
            if compression:
                for ext in ['.gz', '.bz2', '.zip', '.xz']:
                    if filename_no_comp.lower().endswith(ext): 
                        filename_no_comp = filename_no_comp[:-len(ext)]
                        break
            
            # Check if the remaining filename has a recognizable extension
            if filename_no_comp.lower().endswith(('.tsv', '.tab', '.txt')): 
                separator = '\t'
            elif filename_no_comp.lower().endswith('.csv'): 
                separator = ','
            
            # If no separator determined yet and it's compressed, peek at content
            if separator is None and compression == 'gzip':
                logger.debug(f"No clear file extension hint, peeking into gzipped file: {dataset_file}")
                try:
                    with gzip.open(dataset_file, 'rt', errors='ignore') as f:
                        first_line = f.readline().strip()
                    tabs = first_line.count('\t')
                    commas = first_line.count(',')
                    
                    if tabs > 0 and (tabs >= commas or commas == 0):
                        separator = '\t'
                        logger.debug(f"Detected tab separator from content (tabs={tabs}, commas={commas})")
                    elif commas > 0:
                        separator = ','
                        logger.debug(f"Detected comma separator from content (tabs={tabs}, commas={commas})")
                    else:
                        # Default to tab for bioinformatics data
                        separator = '\t'
                        logger.debug("No clear separator in content, defaulting to tab for bioinformatics data")
                except Exception as peek_err:
                    logger.warning(f"Failed to peek at file content: {peek_err}. Defaulting to tab separator.")
                    separator = '\t'  # Default for bioinformatics data
            elif separator is None:
                # Default separator if we can't determine
                separator = '\t'  # Changed default to tab which is more common in bioinformatics
                logger.debug(f"No clear separator determined, defaulting to '{separator}'")
                
            logger.info(f"Using detected separator for text file: '{repr(separator)}'")
            # --- End Separator Detection ---

            df = pd.read_csv(dataset_file, sep=separator, index_col=0, compression=compression)
            if pd.api.types.is_numeric_dtype(df.index):
                 raise ValueError(f"Loaded numeric index for text file {dataset_file.name} using index_col=0.")

        else:
            # This else now correctly applies only if suffix is not .feather AND not a recognized text extension
            raise ValueError(f"Unsupported file type for loading directly into DataFrame by load_dataset: {suffix}. Use load_matrix_data for other types.")

        # Post-load checks (apply only if df was successfully loaded)
        if df is None or df.empty: raise ValueError(f"Loaded DataFrame is None/empty: {dataset_file.name}.")
        if pd.api.types.is_numeric_dtype(df.index): raise ValueError(f"Final index for {dataset_file.name} is numeric. Gene IDs required.")
        df.index = df.index.astype(str)

        logger.info(f"Loaded DataFrame shape: {df.shape}")
        return df

    except Exception as e:
        logger.error(f"Failed to load DataFrame from {dataset_file.name}: {e}", exc_info=True)
        raise # Re-raise exception

# --- load_signature_matrix (Corrected Separator Logic) ---
def load_signature_matrix(sig_matrix="SecAct", use_mmap=False):
    """
    Loads the signature matrix from package data or path into a DataFrame.
    Includes robust index detection and setting.
    Improved separator detection for compressed files.
    """
    if isinstance(sig_matrix, pd.DataFrame):
        logger.debug("Using provided DataFrame as signature matrix.")
        df_copy = sig_matrix.copy(); df_copy.index = df_copy.index.astype(str); return df_copy

    package_root = Path(__file__).parent.parent
    file_path = None
    expected_compression = None
    specified_path = False

    # Determine file path
    if sig_matrix == "SecAct":
        file_path = package_root / "data" / "signature_matrices" / "AllSigFilteredBy_MoranI_TCGA_ICGC_0.25_ds3.tsv.gz"
        expected_compression = 'gzip'
    elif sig_matrix == "CytoSig":
        file_path = package_root / "data" / "signature_matrices" / "signature.centroid.gz"
        expected_compression = 'gzip'
    elif isinstance(sig_matrix, (str, Path)):
        path_obj = Path(sig_matrix).resolve(); specified_path = True
        if path_obj.is_file():
            file_path = path_obj; suffix = "".join(path_obj.suffixes).lower()
            if suffix.endswith('.gz'): expected_compression = 'gzip'
            elif suffix.endswith('.bz2'): expected_compression = 'bz2'
            # Add elif for .zip, .xz if needed
            else: expected_compression = None
        else: raise FileNotFoundError(f"Specified file not found: {path_obj}")
    else: raise ValueError(f"Invalid sig_matrix type: {type(sig_matrix)}")

    if file_path is None or not file_path.exists():
         raise FileNotFoundError(f"Could not find signature matrix file '{sig_matrix}'. Expected: {file_path}")

    logger.info(f"Loading signature matrix file: {file_path}")
    df = None
    try:
        # --- Corrected Separator Detection Logic ---
        separator = None
        filename_no_comp = file_path.name
        if expected_compression: # Strip known compression suffix
            for ext in ['.gz', '.bz2', '.zip', '.xz']:
                 if filename_no_comp.lower().endswith(ext):
                     filename_no_comp = filename_no_comp[:-len(ext)]
                     break

        # Check extension of the uncompressed name
        if filename_no_comp.lower().endswith(('.tsv', '.tab', '.txt')):
            separator = '\t'
        elif filename_no_comp.lower().endswith('.csv'):
            separator = ','

        # If still unknown, peek at the first line (if compressed) or default
        if separator is None:
            if expected_compression == 'gzip':
                logger.debug(f"No clear file extension hint, peeking into gzipped file: {file_path}")
                try:
                    with gzip.open(file_path, 'rt', errors='ignore') as f: # Use text mode, ignore decoding errors
                        first_line = f.readline().strip()
                    tabs = first_line.count('\t')
                    commas = first_line.count(',')
                    if tabs > 0 and tabs >= commas: separator = '\t'; logger.debug(f"Detected tab from content (T={tabs}, C={commas}).")
                    elif commas > 0: separator = ','; logger.debug(f"Detected comma from content (T={tabs}, C={commas}).")
                    else: separator = '\t'; logger.debug("No clear separator in content, defaulting to tab.")
                except Exception as peek_err:
                    logger.warning(f"Failed to peek at file content ({peek_err}). Defaulting to tab separator.")
                    separator = '\t'
            else: # Not compressed or unknown compression
                separator = ',' # Default to comma for unknown uncompressed
                logger.debug(f"No clear file extension hint, defaulting to separator='{separator}'.")

        logger.info(f"Using detected separator: '{repr(separator)}'")

        # --- Load using pandas with index_col=0 ---
        df = pd.read_csv(
            file_path,
            sep=separator,
            index_col=0, # Trust pandas to use first col as index
            compression=expected_compression
        )

        # Check if index is numeric AFTER loading
        if pd.api.types.is_numeric_dtype(df.index):
            raise ValueError(f"Loaded signature matrix has a numeric index using index_col=0. File: {file_path.name}. Gene IDs required.")

        # Ensure index is string type
        df.index = df.index.astype(str)

    except Exception as e:
        logger.error(f"Error reading or processing signature matrix file {file_path}: {e}", exc_info=True)
        raise IOError(f"Error reading signature matrix file {file_path} (used separator='{repr(separator)}'). Check file format/content.") from e

    # Final check after setting index
    if df is None or df.empty or df.shape[1] == 0:
        raise ValueError(f"Loaded signature matrix is empty or has zero columns after processing: {file_path.name}. Shape: {df.shape if df is not None else 'None'}. Check detected separator ('{repr(separator)}') and file content.")

    logger.info(f"Successfully loaded and indexed signature matrix. Final shape: {df.shape}")
    return df
