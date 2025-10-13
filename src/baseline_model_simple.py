# --- 0. IMPORTS ---
import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path

# --- 1. CONFIGURATION ---
PROJECT_ROOT = Path.cwd()
YEAR = 2024

DATA_DIR = PROJECT_ROOT / 'Data'
PROCESSED_DIR = DATA_DIR / 'processed'
REPORTS_DIR = PROJECT_ROOT / 'reports' / 'metrics'

# Create output directory if it doesn't exist
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# --- Input File ---
DATASET_PATH = PROCESSED_DIR / f'master_feature_dataset_{YEAR}.pkl'

# --- 2. DATA LOADING ---
try:
    print(f"Loading dataset from: {DATASET_PATH}...")
    full_df = pd.read_pickle(DATASET_PATH)
    print(f"Successfully loaded {len(full_df):,} records.")
    
    if 'datetime' not in full_df.columns:
        raise KeyError("'datetime' column not found in the dataset.")
    full_df['datetime'] = pd.to_datetime(full_df['datetime'])
    print("Converted 'datetime' column to datetime objects.")

except FileNotFoundError:
    print(f"ERROR: The dataset file was not found at {DATASET_PATH}.")
    exit()
except KeyError as e:
    print(f"ERROR: {e}")
    exit()

# --- 3. BASELINE MODEL LOGIC ---
def predict_baseline(train_df, test_df):
    """
    Predicts ridership based on the average of the same hour and day of the week from the previous four weeks.
    """
    # Calculate the rolling average on the training data
    train_df['day_of_week'] = train_df['datetime'].dt.dayofweek
    train_df['hour'] = train_df['datetime'].dt.hour
    
    # Create a lookup table from the training data
    lookup = train_df.groupby(['day_of_week', 'hour', 'stop_id'])['boardings'].mean().reset_index()
    lookup.rename(columns={'boardings': 'predicted_boardings'}, inplace=True)
    
    # Prepare the test data
    test_df['day_of_week'] = test_df['datetime'].dt.dayofweek
    test_df['hour'] = test_df['datetime'].dt.hour
    
    # Merge the lookup table with the test data
    test_df_with_predictions = pd.merge(test_df, lookup, on=['day_of_week', 'hour', 'stop_id'], how='left')
    
    # Fill missing predictions with the global average from the training data
    global_average = train_df['boardings'].mean()
    test_df_with_predictions['predicted_boardings'].fillna(global_average, inplace=True)
    
    return test_df_with_predictions['predicted_boardings']

# --- 4. CROSS-VALIDATION ---
cv_folds = [
    {'train_start': '2024-01-01', 'train_end': '2024-02-29', 'test_start': '2024-03-01', 'test_end': '2024-03-31'},
    {'train_start': '2024-01-01', 'train_end': '2024-04-30', 'test_start': '2024-05-01', 'test_end': '2024-05-31'},
    {'train_start': '2024-01-01', 'train_end': '2024-06-30', 'test_start': '2024-07-01', 'test_end': '2024-07-31'},
    {'train_start': '2024-01-01', 'train_end': '2024-08-31', 'test_start': '2024-09-01', 'test_end': '2024-09-30'},
    {'train_start': '2024-01-01', 'train_end': '2024-10-31', 'test_start': '2024-11-01', 'test_end': '2024-11-30'},
    {'train_start': '2024-01-01', 'train_end': '2024-11-30', 'test_start': '2024-12-01', 'test_end': '2024-12-31'},
]

all_results = []

for i, fold in enumerate(cv_folds):
    print(f"\n--- FOLD {i+1}/{len(cv_folds)} ---")
    train_df = full_df[(full_df['datetime'] >= fold['train_start']) & (full_df['datetime'] <= fold['train_end'])].copy()
    test_df = full_df[(full_df['datetime'] >= fold['test_start']) & (full_df['datetime'] <= fold['test_end'])].copy()
    
    print(f"Training data: {len(train_df):,} rows")
    print(f"Test data: {len(test_df):,} rows")
    
    y_pred = predict_baseline(train_df, test_df)
    y_true = test_df['boardings']
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    results = {
        'fold': i + 1,
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }
    all_results.append(results)
    
    print(f"Fold {i+1} Results:")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R2: {r2:.4f}")

# --- 5. FINAL RESULTS ---
results_df = pd.DataFrame(all_results)
print("\n--- Overall Baseline Performance ---")
print(results_df.mean())
