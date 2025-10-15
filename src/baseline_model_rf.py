# Importing necessary libraries
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from scipy import stats
from pathlib import Path

# Configuration
PROJECT_ROOT = Path.cwd()
# Specify year for dataset
YEAR = 2024

# Data configuration
PROCESSED_DIR = PROJECT_ROOT / 'Data' / 'processed' 
DATASET_PATH = PROCESSED_DIR / f'master_feature_dataset_{YEAR}.pkl'

# Output paths
MODELS_DIR = PROJECT_ROOT / 'models'
REPORTS_DIR = PROJECT_ROOT / 'reports' / 'metrics'

# Create directories
PROCESSED_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Validation
if not DATASET_PATH.exists():
    raise FileNotFoundError(f"Master dataset not found: {DATASET_PATH}")


# Cross-validation parameters
SAMPLE_SIZE_PER_FOLD = 1000000  # Sample size for training per fold
RANDOM_STATE = 42  # Random state for reproducibility

# Define the 6 time series folds (expanding window)
cv_folds = [
    {'train_start': '2024-01-01', 'train_end': '2024-02-29', 'test_start': '2024-03-01', 'test_end': '2024-03-31'},
    {'train_start': '2024-01-01', 'train_end': '2024-04-30', 'test_start': '2024-05-01', 'test_end': '2024-05-31'},
    {'train_start': '2024-01-01', 'train_end': '2024-06-30', 'test_start': '2024-07-01', 'test_end': '2024-07-31'},
    {'train_start': '2024-01-01', 'train_end': '2024-08-31', 'test_start': '2024-09-01', 'test_end': '2024-09-30'},
    {'train_start': '2024-01-01', 'train_end': '2024-10-31', 'test_start': '2024-11-01', 'test_end': '2024-11-30'},
    {'train_start': '2024-01-01', 'train_end': '2024-11-30', 'test_start': '2024-12-01', 'test_end': '2024-12-31'},
]

print(f"Configured {len(cv_folds)} cross-validation folds")
print(f"Sample size per fold: {SAMPLE_SIZE_PER_FOLD:,}")


def safe_mape(y_true, y_pred):
    """
    Calculates the Mean Absolute Percentage Error (MAPE) safely.
    Handles cases where the true value is 0.
    """
    mask = y_true != 0
    if np.sum(mask) == 0:
        return np.nan, np.mean(np.abs(y_true - y_pred))
    
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    mae_zeros = np.mean(np.abs(y_true[~mask] - y_pred[~mask]))
    
    return mape, mae_zeros

def evaluate_predictions(y_true, y_pred, fold_name):
    """
    Comprehensive evaluation of the predictions
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape, mae_zeros = safe_mape(y_true, y_pred)
    
    results = {
        'fold': fold_name,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape_non_zero': mape,
        'mae_zeros': mae_zeros,
        'n_samples': len(y_true)
    }
    
    print(f"{fold_name}:")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²: {r2:.4f}")
    print(f"  MAPE (non-zero): {mape:.2f}%" if not np.isnan(mape) else "  MAPE: N/A")
    print(f"  MAE (zeros): {mae_zeros:.4f}")
    print(f"  Samples: {len(y_true):,}")
    print()
    
    return results

# Load dataset. Try pickle first, fallback to CSV
DATASET_PICKLE = PROCESSED_DIR / f'master_feature_dataset_{YEAR}.pkl'
DATASET_CSV = PROCESSED_DIR / f'master_feature_dataset_{YEAR}.csv'

if DATASET_PICKLE.exists():
    DATASET_PATH = DATASET_PICKLE
    print(f"Loading dataset from pickle: {DATASET_PATH}")
    full_df = pd.read_pickle(DATASET_PATH)
elif DATASET_CSV.exists():
    DATASET_PATH = DATASET_CSV
else:
    raise FileNotFoundError(f"No master dataset found in {PROCESSED_DIR}")
    
print(f"Dataset loaded with {full_df.shape[0]:,} rows and {full_df.shape[1]} columns")

# Ensure datetime is a datetime object for filtering
full_df['datetime'] = pd.to_datetime(full_df['datetime'])

# Define features and target
TARGET = 'boardings'
features_to_drop = [TARGET, 'datetime', 'trip_id', 'stop_id']
FEATURES = [col for col in full_df.columns if col not in features_to_drop]

print(f"\nTarget: '{TARGET}'")
print(f"Number of features: {len(FEATURES)}")
print(f"Features: {FEATURES}")

print("Validating fold definitions against available data...")
print("=" * 60)

valid_folds = []
for i, fold in enumerate(cv_folds):
    print(f"\nFold {i+1}:")
    print(f"  Train: {fold['train_start']} to {fold['train_end']}")
    print(f"  Test:  {fold['test_start']} to {fold['test_end']}")
    
    # Check training data availability
    train_mask = (
        (full_df['datetime'] >= fold['train_start']) & 
        (full_df['datetime'] <= fold['train_end'])
    )
    train_count = train_mask.sum()
    
    # Check test data availability
    test_mask = (
        (full_df['datetime'] >= fold['test_start']) & 
        (full_df['datetime'] <= fold['test_end'])
    )
    test_count = test_mask.sum()
    
    print(f"  Train data: {train_count:,} rows")
    print(f"  Test data:  {test_count:,} rows")
    
    if train_count == 0:
        print(f"  ❌ INVALID: No training data available")
    elif test_count == 0:
        print(f"  ❌ INVALID: No test data available")
    else:
        print(f"  ✅ VALID")
        valid_folds.append(fold)

print(f"\n{'='*60}")
print(f"Valid folds: {len(valid_folds)} out of {len(cv_folds)}")

if len(valid_folds) == 0:
    raise ValueError("No valid folds found! Check your date ranges against available data.")

# Use only valid folds
cv_folds = valid_folds

print(f"Starting {len(cv_folds)}-fold time series cross-validation...")
print("=" * 70)

all_results = []
fold_predictions = []  # Store predictions for analysis

for fold_idx, fold in enumerate(cv_folds):
    print(f"\n{'='*20} FOLD {fold_idx + 1}/{len(cv_folds)} {'='*20}")
    print(f"Training: {fold['train_start']} to {fold['train_end']}")
    print(f"Testing:  {fold['test_start']} to {fold['test_end']}")
    
    # Create train/test masks
    train_mask = (
        (full_df['datetime'] >= fold['train_start']) & 
        (full_df['datetime'] <= fold['train_end'])
    )
    test_mask = (
        (full_df['datetime'] >= fold['test_start']) & 
        (full_df['datetime'] <= fold['test_end'])
    )
    
    # Get fold data
    train_data = full_df[train_mask].copy()
    test_data = full_df[test_mask].copy()
    
    print(f"Raw training data: {len(train_data):,} rows")
    print(f"Raw test data: {len(test_data):,} rows")
    
    # Sample training data if needed
    if SAMPLE_SIZE_PER_FOLD and len(train_data) > SAMPLE_SIZE_PER_FOLD:
        train_data_sampled = train_data.sample(n=SAMPLE_SIZE_PER_FOLD, random_state=RANDOM_STATE)
        print(f"Sampled training data: {len(train_data_sampled):,} rows")
    else:
        train_data_sampled = train_data
        print(f"Using all training data: {len(train_data_sampled):,} rows")
    
    # Prepare features and target
    X_train_fold = train_data_sampled[FEATURES]
    y_train_fold = train_data_sampled[TARGET]
    X_test_fold = test_data[FEATURES]
    y_test_fold = test_data[TARGET]
    
    print(f"Training features shape: {X_train_fold.shape}")
    print(f"Test features shape: {X_test_fold.shape}")
    
    # Train Random Forest model with the following hyperparameters
    print("\nTraining Random Forest...")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        min_samples_leaf=5,
        max_depth=15,
        oob_score=True
    )
    # Fit model
    rf_model.fit(X_train_fold, y_train_fold)
    print(f"OOB Score: {rf_model.oob_score_:.4f}")
    
    # Make predictions
    print("Making predictions...")
    y_pred_fold = rf_model.predict(X_test_fold)
    
    # Evaluate fold performance
    fold_name = f"Fold {fold_idx + 1}"
    fold_results = evaluate_predictions(y_test_fold, y_pred_fold, fold_name)
    all_results.append(fold_results)
    
    # Store predictions for later analysis
    fold_predictions.append({
        'fold': fold_idx + 1,
        'y_true': y_test_fold.values,
        'y_pred': y_pred_fold,
        'test_dates': test_data['datetime'].values
    })

print("\n" + "="*70)
print("CROSS-VALIDATION COMPLETED")
print("="*70)

# Convert results to DataFrame for analysis
results_df = pd.DataFrame(all_results)

print("\n" + "="*50)
print("CROSS-VALIDATION RESULTS SUMMARY")
print("="*50)

# Calculate summary statistics
metrics = ['mae', 'rmse', 'r2', 'mape_non_zero']
summary_stats = {}

for metric in metrics:
    values = results_df[metric].dropna()  # Remove NaN values
    if len(values) > 0:
        summary_stats[metric] = {
            'mean': values.mean(),
            'std': values.std(),
            'min': values.min(),
            'max': values.max()
        }

# Print summary
print(f"\nNumber of valid folds: {len(results_df)}")
print(f"Total test samples: {results_df['n_samples'].sum():,}")

print("\nMETRIC SUMMARIES:")
print("-" * 40)
for metric, stats in summary_stats.items():
    metric_name = metric.upper().replace('_', ' ')
    print(f"{metric_name}:")
    print(f"  Mean ± Std: {stats['mean']:.4f} ± {stats['std']:.4f}")
    print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
    print()

# Display detailed results table
print("DETAILED FOLD RESULTS:")
print("-" * 40)
display_df = results_df[['fold', 'mae', 'rmse', 'r2', 'mape_non_zero', 'n_samples']].copy()
display_df.columns = ['Fold', 'MAE', 'RMSE', 'R²', 'MAPE (%)', 'Samples']
print(display_df.to_string(index=False, float_format='%.4f'))

# Create comprehensive visualization of CV results
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Time Series Cross-Validation Results', fontsize=16, fontweight='bold')

# MAE across folds
axes[0, 0].bar(results_df['fold'], results_df['mae'], color='skyblue', alpha=0.7)
axes[0, 0].axhline(y=results_df['mae'].mean(), color='red', linestyle='--', label=f'Mean: {results_df["mae"].mean():.4f}')
axes[0, 0].set_title('Mean Absolute Error by Fold')
axes[0, 0].set_xlabel('Fold')
axes[0, 0].set_ylabel('MAE')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# R² across folds
axes[0, 1].bar(results_df['fold'], results_df['r2'], color='lightgreen', alpha=0.7)
axes[0, 1].axhline(y=results_df['r2'].mean(), color='red', linestyle='--', label=f'Mean: {results_df["r2"].mean():.4f}')
axes[0, 1].set_title('R² Score by Fold')
axes[0, 1].set_xlabel('Fold')
axes[0, 1].set_ylabel('R²')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# MAPE across folds (excluding NaN values)
mape_data = results_df.dropna(subset=['mape_non_zero'])
if len(mape_data) > 0:
    axes[1, 0].bar(mape_data['fold'], mape_data['mape_non_zero'], color='orange', alpha=0.7)
    axes[1, 0].axhline(y=mape_data['mape_non_zero'].mean(), color='red', linestyle='--', 
                      label=f'Mean: {mape_data["mape_non_zero"].mean():.2f}%')
    axes[1, 0].set_title('MAPE (Non-zero Boardings) by Fold')
    axes[1, 0].set_xlabel('Fold')
    axes[1, 0].set_ylabel('MAPE (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
else:
    axes[1, 0].text(0.5, 0.5, 'No valid MAPE data', ha='center', va='center', transform=axes[1, 0].transAxes)
    axes[1, 0].set_title('MAPE (Non-zero Boardings) by Fold')

# Sample sizes
axes[1, 1].bar(results_df['fold'], results_df['n_samples'], color='plum', alpha=0.7)
axes[1, 1].set_title('Test Sample Size by Fold')
axes[1, 1].set_xlabel('Fold')
axes[1, 1].set_ylabel('Number of Samples')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
# Save the figure to a file
plt.savefig('time_series_CV_RandomForest.png', dpi=300)
plt.show()

# Additional metric distribution plots
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

metrics_to_plot = ['mae', 'rmse', 'r2']
x_pos = np.arange(len(metrics_to_plot))
means = [results_df[metric].mean() for metric in metrics_to_plot]
stds = [results_df[metric].std() for metric in metrics_to_plot]

bars = ax.bar(x_pos, means, yerr=stds, capsize=5, color=['skyblue', 'lightcoral', 'lightgreen'], alpha=0.7)
ax.set_xlabel('Metrics')
ax.set_ylabel('Values')
ax.set_title('Cross-Validation Metrics (Mean ± Std)')
ax.set_xticks(x_pos)
ax.set_xticklabels([m.upper() for m in metrics_to_plot])
ax.grid(True, alpha=0.3)

# Add value labels on bars
for bar, mean, std in zip(bars, means, stds):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
            f'{mean:.4f}\n±{std:.4f}',
            ha='center', va='bottom', fontsize=9)

# Save the figure to a file
plt.savefig('cross_validation_metrics_random_forest.png', dpi=300)
plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("FINAL CROSS-VALIDATION SUMMARY")
print("="*60)

# Calculate confidence intervals (assuming normal distribution)
def calculate_confidence_interval(data, confidence=0.95):
    """
    Calculate confidence interval for the mean
    """
    data_clean = data.dropna()
    if len(data_clean) < 2:
        return None, None
    
    mean = data_clean.mean()
    sem = stats.sem(data_clean)  # Standard error of the mean
    ci = stats.t.interval(confidence, len(data_clean)-1, loc=mean, scale=sem)
    return ci

print(f"Model Performance across {len(cv_folds)} folds:")
print("-" * 50)

# MAE Summary
mae_mean = results_df['mae'].mean()
mae_std = results_df['mae'].std()
mae_ci = calculate_confidence_interval(results_df['mae'])
print(f"Mean Absolute Error:")
print(f"  Mean: {mae_mean:.4f} ± {mae_std:.4f}")
if mae_ci[0] is not None:
    print(f"  95% CI: [{mae_ci[0]:.4f}, {mae_ci[1]:.4f}]")

# R² Summary
r2_mean = results_df['r2'].mean()
r2_std = results_df['r2'].std()
r2_ci = calculate_confidence_interval(results_df['r2'])
print(f"\nR² Score:")
print(f"  Mean: {r2_mean:.4f} ± {r2_std:.4f}")
if r2_ci[0] is not None:
    print(f"  Mean: {r2_mean:.4f} ± {r2_std:.4f}")
if r2_ci[0] is not None:
    print(f"  95% CI: [{r2_ci[0]:.4f}, {r2_ci[1]:.4f}]")

# MAPE Summary
mape_mean = results_df['mape_non_zero'].mean()
mape_std = results_df['mape_non_zero'].std()
mape_ci = calculate_confidence_interval(results_df['mape_non_zero'])
print(f"\nMAPE (on non-zero boardings):")
print(f"  Mean: {mape_mean:.2f}% ± {mape_std:.2f}%")
if mape_ci[0] is not None:
    print(f"  95% CI: [{mape_ci[0]:.2f}%, {mape_ci[1]:.2f}%]")

print("\nEnd of summary")