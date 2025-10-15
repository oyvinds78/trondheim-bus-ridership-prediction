# --- 0. IMPORTS ---
import os
import pandas as pd
import numpy as np
import time
import gc
import random
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# --- 1. CONFIGURATION ---
PROJECT_ROOT = Path.cwd()
YEAR = 2024

DATA_DIR = PROJECT_ROOT / 'Data'
PROCESSED_DIR = DATA_DIR / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models'

# Create output directory if it doesn't exist
MODELS_DIR.mkdir(exist_ok=True)

# --- Express Tune Configuration ---
TRAINING_SAMPLE_SIZE = 500000
N_PARAM_SETS_TO_TEST = 8
RANDOM_STATE = 42

# --- 2. DATA LOADING ---
ENGINEERED_DATASET_PATH = PROCESSED_DIR / 'master_feature_dataset_2024_with_advanced_features.pkl'

if not ENGINEERED_DATASET_PATH.exists():
    raise FileNotFoundError(f"Engineered dataset not found: {ENGINEERED_DATASET_PATH}")

print("Loading pre-engineered dataset (Jan-Sep)...")
full_df_temp = pd.read_pickle(ENGINEERED_DATASET_PATH)
# Use only Jan-Sep for hyperparameter tuning validation
full_df = full_df_temp[full_df_temp['datetime'].dt.month <= 9].copy()
del full_df_temp
gc.collect()

print(f"Dataset loaded with {full_df.shape[0]:,} rows.")

# --- 3. DEFINE FEATURES, FOLDS, AND PARAMETER GRID ---
TARGET = 'boardings'
FEATURES = [
    'route_id', 'stop_sequence', 'temperature_celsius', 'wind_gust_m_s',
    'min_temp_day', 'relative_humidity_percent', 'precipitation_mm_hr',
    'wind_speed_m_s', 'is_rbk_home_game', 'is_major_event', 'is_special_event',
    'is_other_holiday', 'is_summer_holiday', 'is_e_scooter_season',
    'is_student_semester', 'day_of_year', 'month', 'is_weekend',
    'is_rush_hour', 'is_first_stop', 'is_last_stop',
    'stop_sequence_normalized', 'hour_of_day', 'day_of_week',
    'boardings_same_stop_last_hour', 'boardings_same_hour_last_week',
    'boardings_rolling_mean_3h', 'boardings_rolling_max_3h',
    'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos'
]

# Use the first 3 custom folds for a quick validation
cv_folds_subset = [
    {'train_start': '2024-01-01', 'train_end': '2024-02-28', 'test_start': '2024-03-01', 'test_end': '2024-03-30'},
    {'train_start': '2024-01-01', 'train_end': '2024-04-29', 'test_start': '2024-05-01', 'test_end': '2024-05-31'},
    {'train_start': '2024-01-01', 'train_end': '2024-06-29', 'test_start': '2024-07-01', 'test_end': '2024-07-31'}
]

# Define parameter grid and select random combinations
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [8, 12, 18], # Adjusted to include 18 from your other script
    'min_samples_leaf': [6, 10, 20, 50], # Adjusted to include 6
    'max_features': ['sqrt', 'log2']
}

from itertools import product
all_params = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]
random.seed(RANDOM_STATE)
param_sets_to_test = random.sample(all_params, min(N_PARAM_SETS_TO_TEST, len(all_params)))
print(f"Created {len(param_sets_to_test)} random parameter sets to test.")

# --- 4. EXPRESS TUNING LOOP ---
tuning_results = []
total_start_time = time.time()
print(f"\n--- Starting Express Tuning for {len(param_sets_to_test)} Parameter Sets ---")

for i, params in enumerate(param_sets_to_test):
    set_start_time = time.time()
    print(f"\n[Set {i+1}/{len(param_sets_to_test)}] Testing params: {params}")
    
    fold_maes = []
    fold_r2s = []

    for j, fold in enumerate(cv_folds_subset):
        print(f"  -> Fitting Fold {j+1}/{len(cv_folds_subset)}...")
        train_df_fold = full_df[(full_df['datetime'] >= fold['train_start']) & (full_df['datetime'] <= fold['train_end'])]
        test_df_fold = full_df[(full_df['datetime'] >= fold['test_start']) & (full_df['datetime'] <= fold['test_end'])]
        
        train_sample_df = train_df_fold.tail(TRAINING_SAMPLE_SIZE)

        X_train, y_train = train_sample_df[FEATURES], train_sample_df[TARGET]
        X_test, y_test = test_df_fold[FEATURES], test_df_fold[TARGET]

        model = RandomForestRegressor(**params, random_state=RANDOM_STATE, n_jobs=-1)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        fold_maes.append(mae)
        fold_r2s.append(r2)
        print(f"     Fold {j+1} Result -> MAE: {mae:.4f}, R²: {r2:.4f}")

    avg_mae = np.mean(fold_maes)
    std_mae = np.std(fold_maes)
    avg_r2 = np.mean(fold_r2s)
    std_r2 = np.std(fold_r2s)
    
    tuning_results.append({
        'params': params, 
        'avg_mae': avg_mae, 'std_mae': std_mae,
        'avg_r2': avg_r2, 'std_r2': std_r2
    })
    
    set_end_time = time.time()
    print(f"--> Avg MAE: {avg_mae:.4f} (± {std_mae:.4f}) | Avg R²: {avg_r2:.4f} (± {std_r2:.4f}) | Time: {(set_end_time - set_start_time)/60:.2f} mins")

total_end_time = time.time()
print(f"\n--- Express Tuning Complete ---")
print(f"Total time taken: {(total_end_time - total_start_time)/60:.2f} minutes")

# --- 5. ANALYZE RESULTS AND SAVE BEST MODEL ---
if tuning_results:
    results_df = pd.DataFrame(tuning_results)
    best_result = results_df.loc[results_df['avg_mae'].idxmin()]
    
    best_params = best_result['params']
    best_mae = best_result['avg_mae']
    best_mae_std = best_result['std_mae']
    best_r2 = best_result['avg_r2']
    best_r2_std = best_result['std_r2']

    print("\n" + "="*50)
    print("--- Best Hyperparameter Set Found ---")
    print(f"Average MAE: {best_mae:.4f} (± {best_mae_std:.4f})")
    print(f"Average R²:  {best_r2:.4f} (± {best_r2_std:.4f})")
    print("\nParameters:")
    print(best_params)
    print("="*50)

    # Train final model on full Jan-Sep data with best params
    print("\n--- Training Final Model on Full Jan-Sep Data ---")
    final_model = RandomForestRegressor(**best_params, random_state=RANDOM_STATE, n_jobs=-1)
    final_model.fit(full_df[FEATURES], full_df[TARGET])
    print("Final model training complete.")

    # Save the final model artifact
    model_artifact = {
        'model': final_model,
        'features': FEATURES,
        'best_params': best_params,
        'cv_mae': best_mae
    }
    
    MODEL_FILENAME = 'final_bus_ridership_model.joblib'
    MODEL_PATH = MODELS_DIR / MODEL_FILENAME
    joblib.dump(model_artifact, MODEL_PATH)
    print(f"\nFinal tuned model and info saved to: {MODEL_PATH}")

else:
    print("\n--- No tuning results to analyze. ---")
