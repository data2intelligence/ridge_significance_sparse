#!/usr/bin/env python
"""
Example of logistic regression with Firth correction using real infection data.
Applies PCA for dimensionality reduction before regression.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import time
import logging
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.preprocessing import StandardScaler # For scaling before PCA
from sklearn.decomposition import PCA            # For PCA

# --- Set OpenMP threads early if needed ---
os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "4")

# --- Setup Paths and Logging ---
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent # Assumes script is in examples/
sys.path.append(str(project_root)) # Add project root to allow importing ridge_inference

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
logger = logging.getLogger(__name__)

try:
    # Import necessary function from the package
    from ridge_inference import logit
except ImportError as e:
    logger.error(f"Error importing ridge_inference.logit: {e}"); sys.exit(1)

# --- Configuration ---
DATA_DIR = project_root / "data" / "sample_data"
EXPR_PATH = DATA_DIR / "infection_GSE147507.gz"
META_PATH = DATA_DIR / "infection_GSE147507_meta.csv"
OUTPUT_DIR = script_dir / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
TARGET_COLUMN = "Label"
SAMPLE_ID_COLUMN = None # Assume index is sample ID in meta if this is None
N_COMPONENTS = 5 # <<< Number of PCA components to use

# --- Logistic Regression Parameters ---
LOGIT_TOL = 1e-6
MAX_DELTA = 0.5
MAX_ITER = 100
LOGIT_VERBOSE = 0 # Set to 1 or 2 for more C++ output

# --- Load Data ---
logger.info(f"Loading expression data from: {EXPR_PATH}")
if not EXPR_PATH.exists(): logger.error(f"File not found: {EXPR_PATH}"); sys.exit(1)
try: expr_df = pd.read_csv(EXPR_PATH, sep='\t', index_col=0, compression='gzip')
except Exception as e: logger.error(f"Failed to load expression data: {e}"); sys.exit(1)

logger.info(f"Loading metadata from: {META_PATH}")
if not META_PATH.exists(): logger.error(f"File not found: {META_PATH}"); sys.exit(1)
try: meta_df = pd.read_csv(META_PATH, index_col=0) # Assume index is sample ID if SAMPLE_ID_COLUMN is None
except Exception as e: logger.error(f"Failed to load metadata: {e}"); sys.exit(1)

# --- Preprocessing ---
logger.info("Aligning expression data and metadata...")
# Determine sample IDs from metadata index or specified column
if SAMPLE_ID_COLUMN:
     if SAMPLE_ID_COLUMN not in meta_df.columns: logger.error(f"Sample ID column '{SAMPLE_ID_COLUMN}' not found in metadata."); sys.exit(1)
     meta_sample_ids = meta_df[SAMPLE_ID_COLUMN].astype(str)
     meta_df_indexed = meta_df.set_index(SAMPLE_ID_COLUMN) # Use this for filtering
else:
     meta_sample_ids = meta_df.index.astype(str)
     meta_df_indexed = meta_df # Already indexed correctly

expr_sample_ids = expr_df.columns.astype(str)
common_samples = sorted(list(set(meta_sample_ids) & set(expr_sample_ids)))
if not common_samples: logger.error("No common samples found."); sys.exit(1)
logger.info(f"Found {len(common_samples)} common samples.")

# Filter based on common samples
meta_filtered = meta_df_indexed.loc[common_samples]
expr_filtered = expr_df[common_samples] # Select columns directly

# Prepare data: X = samples x genes, y = labels
# Transpose expression data so samples are rows
X_data_raw = expr_filtered.T.values
# Get labels corresponding to the filtered samples
y_data = meta_filtered[TARGET_COLUMN].values
logger.info(f"Raw data shapes - X: {X_data_raw.shape}, y: {y_data.shape}")
if X_data_raw.shape[0] != y_data.shape[0]:
    logger.error(f"Sample count mismatch after alignment: X has {X_data_raw.shape[0]}, y has {y_data.shape[0]}"); sys.exit(1)

# Ensure y is numeric (0 or 1)
if not np.isin(y_data, [0, 1]).all():
    unique_labels = np.unique(y_data)
    logger.warning(f"Target labels are not 0/1 ({unique_labels}). Assuming first unique value is 0, second is 1.")
    if len(unique_labels) != 2: logger.error("Target label must have exactly two unique values."); sys.exit(1)
    label_map = {unique_labels[0]: 0, unique_labels[1]: 1}
    y_data = np.array([label_map[val] for val in y_data], dtype=int)


# --- Feature Scaling and PCA ---
logger.info("Scaling features (genes) before PCA...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_data_raw) # Scale samples x genes

# Check if N_COMPONENTS is valid before PCA
if N_COMPONENTS >= min(X_scaled.shape):
    logger.warning(f"N_COMPONENTS ({N_COMPONENTS}) >= min(n_samples, n_features) ({min(X_scaled.shape)}). Adjusting N_COMPONENTS.")
    N_COMPONENTS = min(X_scaled.shape) - 1
    if N_COMPONENTS < 1: logger.error("Cannot perform PCA with less than 1 component."); sys.exit(1)
    logger.warning(f"Using N_COMPONENTS = {N_COMPONENTS}")

logger.info(f"Applying PCA to reduce dimensions to {N_COMPONENTS} components...")
pca = PCA(n_components=N_COMPONENTS)
X_pca = pca.fit_transform(X_scaled)
logger.info(f"PCA transformed data shape: {X_pca.shape}") # samples x components
explained_variance = pca.explained_variance_ratio_.sum()
logger.info(f"Explained variance by {N_COMPONENTS} components: {explained_variance:.3f}")

# Add intercept term to PCA components
X_data_final = np.hstack([np.ones((X_pca.shape[0], 1)), X_pca])
feature_names_final = ["Intercept"] + [f"PC{i+1}" for i in range(N_COMPONENTS)]
logger.info(f"Final data shape with intercept: {X_data_final.shape}")

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X_data_final, y_data, test_size=0.3, random_state=42, stratify=y_data
)
logger.info(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
logger.info(f"Test shapes: X={X_test.shape}, y={y_test.shape}")
# Check PCA validity relative to train set size
if X_train.shape[1] > X_train.shape[0]: # Features > samples
     logger.warning(f"Number of features (PCA+Int={X_train.shape[1]}) > N train samples ({X_train.shape[0]}). Model might be unstable. Consider reducing N_COMPONENTS further.")


# --- Run Logistic Regression (on PCA components) ---
methods = { "Standard": {"correction": 0}, "Firth-corrected": {"correction": 1} }
results = {}

for name, params in methods.items():
    logger.info(f"\nRunning {name} logistic regression on PCA components...")
    start_time = time.time()
    # Explicitly initialize outputs for this iteration
    beta, se, z, pvalue = None, None, None, None
    accuracy, roc_auc = np.nan, np.nan
    fpr, tpr = np.array([0, 1]), np.array([0, 1]) # Default line

    try:
        # Use the PCA-transformed data
        beta, se, z, pvalue = logit(
            X_train, y_train,
            tol=LOGIT_TOL, max_delta=MAX_DELTA, maxIter=MAX_ITER,
            correction=params["correction"], verbose=LOGIT_VERBOSE
        )
        duration = time.time() - start_time
        logger.info(f"{name} regression finished in {duration:.2f} seconds.")

        # Check if output is valid before proceeding
        if beta is None or se is None or z is None or pvalue is None:
            raise RuntimeError(f"{name} logit function returned None result(s).")
        if not all(isinstance(arr, np.ndarray) for arr in [beta, se, z, pvalue]):
             raise TypeError(f"{name} logit function did not return numpy arrays.")
        if not all(arr.shape == (X_train.shape[1],) for arr in [beta, se, z, pvalue]):
             raise ValueError(f"{name} logit function returned incorrect shapes.")

        # Predict on test set - Use np.clip to avoid overflow/underflow in exp
        X_test_beta = X_test @ beta
        y_pred_prob = 1.0 / (1.0 + np.exp(-np.clip(X_test_beta, -500, 500))) # Clip linear predictor
        y_pred = (y_pred_prob > 0.5).astype(int)

        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        # Handle cases where roc_curve might return NaN/Inf or empty arrays
        if fpr is None or tpr is None or len(fpr) < 2 or np.isnan(fpr).any() or np.isnan(tpr).any():
             logger.warning(f"Invalid ROC curve values for {name}. Setting AUC to 0.5.")
             roc_auc = 0.5
             fpr, tpr = np.array([0, 1]), np.array([0, 1]) # Default line
        else:
            roc_auc = auc(fpr, tpr)

    except ArithmeticError as e: # Catch Cholesky failure specifically
        logger.error(f"{name} regression failed (likely Cholesky): {e}")
        # Ensure arrays are created with the correct size even on failure
        num_features = X_train.shape[1]
        beta, se, z, pvalue = (np.full(num_features, np.nan),) * 4
        accuracy, roc_auc = np.nan, np.nan
        fpr, tpr = np.array([0, 1]), np.array([0, 1]) # Default line
    except Exception as e:
         logger.error(f"Unexpected error during {name} regression: {e}", exc_info=True)
         num_features = X_train.shape[1]
         beta, se, z, pvalue = (np.full(num_features, np.nan),) * 4
         accuracy, roc_auc = np.nan, np.nan
         fpr, tpr = np.array([0, 1]), np.array([0, 1])

    # Store results
    results[name] = {
        "beta": beta, "se": se, "z": z, "pvalue": pvalue,
        "accuracy": accuracy, "auc": roc_auc, "fpr": fpr, "tpr": tpr
    }

    # Print summary (handle potential NaNs)
    logger.info(f"\n--- {name} Results (PCA Features) ---")
    logger.info("Coefficients (beta):")
    # Check if beta is valid before trying to iterate
    if beta is None or np.isnan(beta).all():
         logger.warning("  Regression failed or returned NaN, coefficients cannot be displayed.")
    else:
         for i in range(len(beta)):
             # Check pvalue validity for stars
             p = pvalue[i] if pvalue is not None and i < len(pvalue) and np.isfinite(pvalue[i]) else 1.0
             beta_val = beta[i] if np.isfinite(beta[i]) else np.nan
             se_val = se[i] if se is not None and i < len(se) and np.isfinite(se[i]) else np.nan
             stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
             logger.info(f"  {feature_names_final[i]:<10}: {beta_val:+.4f} (SE={se_val:.4f}, p={p:.4f}) {stars}")

    logger.info(f"Accuracy on test set: {accuracy:.4f}")
    logger.info(f"AUC on test set: {roc_auc:.4f}")

# --- Visualization (on PCA results) ---
# ROC Curve
plt.figure(figsize=(8, 8))
plot_success_roc = False
colors = {'Standard': 'blue', 'Firth-corrected': 'red'}
for name, res in results.items():
    # Check if required ROC data exists and is valid
    if 'fpr' in res and 'tpr' in res and 'auc' in res and \
       res['fpr'] is not None and res['tpr'] is not None and \
       not np.isnan(res['auc']) and len(res['fpr'])>0 and len(res['tpr'])>0:
        plt.plot(res["fpr"], res["tpr"], color=colors.get(name, 'gray'), lw=2, label=f"{name} (AUC = {res['auc']:.3f})")
        plot_success_roc = True
    else:
        logger.warning(f"Skipping ROC plot for '{name}' due to missing or invalid data.")

if plot_success_roc:
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--'); plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('Logistic Regression ROC Curve (PCA Features)')
    plt.legend(loc="lower right"); plt.grid(True)
    roc_plot_path = OUTPUT_DIR / 'logistic_regression_pca_roc.png'
    try: plt.savefig(roc_plot_path); logger.info(f"\nROC curve plot saved to {roc_plot_path}")
    except Exception as e: logger.error(f"Failed to save ROC plot: {e}")
else: logger.warning("Skipping ROC plot due to missing/invalid results.")
plt.close()

# Coefficient Comparison
beta_std = results.get("Standard", {}).get("beta")
beta_firth = results.get("Firth-corrected", {}).get("beta")
# Check if both results are valid numpy arrays before plotting
if isinstance(beta_std, np.ndarray) and isinstance(beta_firth, np.ndarray) and \
   not np.isnan(beta_std).all() and not np.isnan(beta_firth).all() and \
   len(beta_std) == len(feature_names_final) and len(beta_firth) == len(feature_names_final):
    plt.figure(figsize=(10, 6)); n_plot = len(feature_names_final); x = np.arange(n_plot); width = 0.35
    plt.bar(x - width/2, beta_std, width, label='Standard', color='blue', alpha=0.7)
    plt.bar(x + width/2, beta_firth, width, label='Firth', color='red', alpha=0.7)
    plt.xlabel('Features'); plt.ylabel('Coefficient Value'); plt.title('Comparison of Logistic Regression Coefficients (PCA Features)')
    plt.xticks(x, feature_names_final, rotation=45, ha='right'); plt.legend(); plt.grid(axis='y', linestyle='--', alpha=0.7); plt.tight_layout()
    coef_plot_path = OUTPUT_DIR / 'logistic_regression_pca_coefficients.png'
    try: plt.savefig(coef_plot_path); logger.info(f"Coefficient plot saved to {coef_plot_path}")
    except Exception as e: logger.error(f"Failed to save Coef plot: {e}")
else: logger.warning("Skipping Coefficient plot due to missing/invalid results.")
plt.close()

# Standard Error Comparison
se_std = results.get("Standard", {}).get("se")
se_firth = results.get("Firth-corrected", {}).get("se")
# Check if both results are valid numpy arrays before plotting
if isinstance(se_std, np.ndarray) and isinstance(se_firth, np.ndarray) and \
   not np.isnan(se_std).all() and not np.isnan(se_firth).all() and \
   len(se_std) == len(feature_names_final) and len(se_firth) == len(feature_names_final):
    plt.figure(figsize=(10, 6)); n_plot = len(feature_names_final); x = np.arange(n_plot); width = 0.35
    plt.bar(x - width/2, se_std, width, label='Standard', color='blue', alpha=0.7)
    plt.bar(x + width/2, se_firth, width, label='Firth', color='red', alpha=0.7)
    plt.xlabel('Features'); plt.ylabel('Standard Error'); plt.title('Comparison of Logistic Regression Standard Errors (PCA Features)')
    plt.xticks(x, feature_names_final, rotation=45, ha='right'); plt.legend(); plt.grid(axis='y', linestyle='--', alpha=0.7); plt.tight_layout()
    se_plot_path = OUTPUT_DIR / 'logistic_regression_pca_se.png'
    try: plt.savefig(se_plot_path); logger.info(f"Standard error plot saved to {se_plot_path}")
    except Exception as e: logger.error(f"Failed to save SE plot: {e}")
else: logger.warning("Skipping SE plot due to missing/invalid results.")
plt.close()

logger.info("\nLogistic regression example finished.")