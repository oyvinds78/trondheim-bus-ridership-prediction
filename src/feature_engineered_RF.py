# --- 0. IMPORTS ---
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import gc
import joblib
from pathlib import Path
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# --- HELPER FUNCTION ---

def evaluate_predictions(y_true, y_pred, fold_name):
    """
    Calculates key performance metrics and returns them in a dictionary
    with standardized, uppercase keys for consistent analysis.
    """
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # This dictionary uses the exact keys the analysis code expects ('MAE', 'R2')
    results = {
        'Fold': fold_name,
        'MAE': mae,
        'R2': r2
    }
    
    return results

# --- 1. CONFIGURATION ---
PROJECT_ROOT = Path.cwd()
# Specify year for dataset
YEAR = 2024

# Data configuration
PROCESSED_DIR = PROJECT_ROOT / 'Data' / 'processed' 
DATASET_PATH = PROCESSED_DIR / f'master_feature_dataset_{YEAR}.pkl'

# Output paths
MODELS_DIR = PROJECT_ROOT / 'code' / 'modeling'
REPORTS_DIR = PROJECT_ROOT / 'reports' / 'metrics'

# Cross-validation parameters (identical to the comparative notebook)
SAMPLE_SIZE_PER_FOLD = 800000
RANDOM_STATE = 42
BASELINE_RF_MAE = 1.5987 # From the comparative analysis

# Define the exact same 6 time-series folds as basline and comparative analysis
cv_folds = [
    {'train_start': '2024-01-01', 'train_end': '2024-02-28', 'test_start': '2024-03-01', 'test_end': '2024-03-30'},
    {'train_start': '2024-01-01', 'train_end': '2024-04-29', 'test_start': '2024-05-01', 'test_end': '2024-05-31'},
    {'train_start': '2024-01-01', 'train_end': '2024-06-29', 'test_start': '2024-07-01', 'test_end': '2024-07-31'},
    {'train_start': '2024-01-01', 'train_end': '2024-08-30', 'test_start': '2024-09-01', 'test_end': '2024-09-30'},
    {'train_start': '2024-01-01', 'train_end': '2024-10-30', 'test_start': '2024-11-01', 'test_end': '2024-11-30'},
    {'train_start': '2024-01-01', 'train_end': '2024-11-29', 'test_start': '2024-12-01', 'test_end': '2024-12-31'}
]

print(f"Configured {len(cv_folds)} cross-validation folds for Random Forest.")

# Define data types for memory optimization
dtype_spec = {
    'route_id': 'category', 'trip_id': 'int32', 'stop_id': 'int32',
    'stop_sequence': 'int16', 'boardings': 'int16', 'temperature_celsius': 'float32',
    'wind_gust_m_s': 'float32', 'min_temp_day': 'float32', 'relative_humidity_percent': 'float32',
    'precipitation_mm_hr': 'float32', 'wind_speed_m_s': 'float32', 'is_rbk_home_game': 'bool',
    'is_major_event': 'bool', 'is_special_event': 'bool', 'is_other_holiday': 'bool',
    'is_summer_holiday': 'bool', 'is_e_scooter_season': 'bool', 'is_student_semester': 'bool',
    'hour_of_day': 'int8', 'day_of_week': 'int8', 'day_of_year': 'int16', 'month': 'int8',
    'is_weekend': 'bool', 'is_rush_hour': 'bool', 'is_first_stop': 'bool', 'is_last_stop': 'bool',
    'stop_sequence_normalized': 'float32'
}

# Load dataset - try pickle first, fallback on csv
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

print("Engineering advanced features...")
start_fe_time = time.time()

# Sort values correctly to ensure accurate lag/rolling calculations
full_df.sort_values(['stop_id', 'route_id', 'datetime'], inplace=True)

# --- Lag Features ---
print("  Creating lag features...")
# Group by stop and route to calculate previous hour's boardings
full_df['boardings_same_stop_last_hour'] = full_df.groupby(['stop_id', 'route_id'], observed=True)['boardings'].shift(1).fillna(0)
# Group by stop, route, and hour to get boardings from the same hour a week ago
full_df['boardings_same_hour_last_week'] = full_df.groupby(['stop_id', 'route_id', 'hour_of_day'], observed=True)['boardings'].shift(168).fillna(0)

# --- Rolling Window Features ---
print("  Creating rolling window features...")
# Group for rolling calculations
grouped = full_df.groupby(['stop_id', 'route_id'], observed=True)['boardings']
# Calculate rolling mean and max over a 3-hour window from the past
full_df['boardings_rolling_mean_3h'] = grouped.transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()).fillna(0)
full_df['boardings_rolling_max_3h'] = grouped.transform(lambda x: x.shift(1).rolling(3, min_periods=1).max()).fillna(0)

# --- Cyclical Time Features ---
print("  Creating cyclical time features...")
full_df['hour_sin'] = np.sin(2 * np.pi * full_df['hour_of_day'] / 24)
full_df['hour_cos'] = np.cos(2 * np.pi * full_df['hour_of_day'] / 24)
full_df['day_of_week_sin'] = np.sin(2 * np.pi * full_df['day_of_week'] / 7)
full_df['day_of_week_cos'] = np.cos(2 * np.pi * full_df['day_of_week'] / 7)

end_fe_time = time.time()
print(f"Feature engineering complete. Time taken: {end_fe_time - start_fe_time:.2f} seconds.")

# Define the final feature set for the model
TARGET = 'boardings'
features_to_drop = [TARGET, 'datetime', 'trip_id', 'stop_id']
FEATURES = [col for col in full_df.columns if col not in features_to_drop]

print(f"\nTotal number of features for modeling: {len(FEATURES)}")

# --- SAVE COMPLETE ENGINEERED MASTER DATASET ---
print("\n--- Saving Complete Engineered Master Dataset ---")

# Save the full engineered dataset (not just training portion)
COMPLETE_ENGINEERED_PICKLE = 'master_feature_dataset_2024_with_advanced_features.pkl'
COMPLETE_ENGINEERED_PATH = os.path.join(PROCESSED_DIR /  COMPLETE_ENGINEERED_PICKLE)

try:
    full_df.to_pickle(COMPLETE_ENGINEERED_PATH)
    print(f" Successfully saved complete engineered master dataset to: {COMPLETE_ENGINEERED_PATH}")
    print(f"   Dataset shape: {full_df.shape}")
    print(f"   Date range: {full_df['datetime'].min()} to {full_df['datetime'].max()}")
except Exception as e:
    print(f" Error saving complete engineered dataset: {e}")

print(f"Starting {len(cv_folds)}-fold cross-validation for Random Forest with advanced features...")

# --- 1. INITIALIZE CONTAINERS ---
# Using containers from the template
all_results = []
feature_importances_per_fold = {}
fold_predictions = []
final_model = None # This will hold the model from the last fold

# The complete list of 32 features for the advanced model
FEATURES = [
    # Base Features
    'route_id', 'stop_sequence', 'temperature_celsius', 'wind_gust_m_s',
    'min_temp_day', 'relative_humidity_percent', 'precipitation_mm_hr',
    'wind_speed_m_s', 'is_rbk_home_game', 'is_major_event', 'is_special_event',
    'is_other_holiday', 'is_summer_holiday', 'is_e_scooter_season',
    'is_student_semester', 'day_of_year', 'month', 'is_weekend',
    'is_rush_hour', 'is_first_stop', 'is_last_stop',
    'stop_sequence_normalized',
    
    # Original Time Features (kept for tree-based models)
    'hour_of_day', 'day_of_week',

    # Engineered Lag Features
    'boardings_same_stop_last_hour',
    'boardings_same_hour_last_week',

    # Engineered Rolling Window Features
    'boardings_rolling_mean_3h',
    'boardings_rolling_max_3h',

    # Engineered Cyclical Time Features
    'hour_sin', 'hour_cos',
    'day_of_week_sin', 'day_of_week_cos'
]

# Initialize the model once before the loop
rf_model = RandomForestRegressor(
    n_estimators=150,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    min_samples_leaf=6,
    max_depth=18,
    max_features='sqrt'
)

# --- 2. CROSS-VALIDATION LOOP ---
for fold_idx, fold in enumerate(cv_folds):
    print(f"\n--- FOLD {fold_idx + 1}/{len(cv_folds)} ---")
    print(f"Training: {fold['train_start']} to {fold['train_end']}")
    print(f"Testing:  {fold['test_start']} to {fold['test_end']}")
    
    # Create train/test masks and data for the fold
    train_mask = (full_df['datetime'] >= fold['train_start']) & (full_df['datetime'] <= fold['train_end'])
    test_mask = (full_df['datetime'] >= fold['test_start']) & (full_df['datetime'] <= fold['test_end'])
    train_data = full_df[train_mask]
    test_data = full_df[test_mask]
    
    # Sample training data
    train_data_sampled = train_data.sample(n=SAMPLE_SIZE_PER_FOLD, random_state=RANDOM_STATE) if len(train_data) > SAMPLE_SIZE_PER_FOLD else train_data
    
    X_train_fold, y_train_fold = train_data_sampled[FEATURES], train_data_sampled[TARGET]
    X_test_fold, y_test_fold = test_data[FEATURES], test_data[TARGET]
    
    # Train model
    print(f"Training model on {len(X_train_fold):,} samples...")
    start_train_time = time.time()
    rf_model.fit(X_train_fold, y_train_fold)
    print(f"Training time: {time.time() - start_train_time:.2f}s")

    # Predict
    y_pred_fold = rf_model.predict(X_test_fold)
    
    # --- 3. COLLECT ARTIFACTS ---
    fold_name = f"Fold {fold_idx + 1}"
    
    # a. Evaluate and store performance metrics
    fold_results = evaluate_predictions(y_test_fold, y_pred_fold, fold_name)
    all_results.append(fold_results)
    print(f"Fold Result -> MAE: {fold_results['MAE']:.4f}, R²: {fold_results['R2']:.4f}")

    # b. Store feature importances
    feature_importances_per_fold[fold_name] = rf_model.feature_importances_

    # c. Store raw predictions for later analysis
    prediction_data = {
        'fold': fold_name,
        'y_true': y_test_fold.values,
        'y_pred': y_pred_fold
    }
    fold_predictions.append(prediction_data)
    
    # d. Keep the model from the final fold as the "champion"
    if fold_idx == len(cv_folds) - 1:
        final_model = rf_model

# Clean up memory after the loop
del full_df
gc.collect()

# Create train_data_engineered from the last fold's training data
train_data_engineered = train_data_sampled.copy()

print("\n" + "=" * 50)
print("Cross-validation and artifact collection complete.")
print("=" * 50)

try:
    # 1. Convert the list of dictionaries into a DataFrame for easy analysis
    results_df = pd.DataFrame(all_results)
    
    print("--- Cross-Validation Performance Summary (All Folds) ---")
    display(results_df)

    # 2. Calculate the mean and standard deviation for the MAE
    mae_mean = results_df['MAE'].mean()
    mae_std = results_df['MAE'].std()
    
    # 3. Calculate for other key metrics as well
    r2_mean = results_df['R2'].mean()
    r2_std = results_df['R2'].std()

    # 4. Print the final, summarized results in the format for your report
    print("\n--- Final Model Performance Estimate ---")
    print(f"Based on 6-fold cross-validation:")
    print(f"Mean Absolute Error (MAE): {mae_mean:.4f} ± {mae_std:.4f}")
    print(f"R-squared (R²):              {r2_mean:.4f} ± {r2_std:.4f}")
    print("\nThe standard deviation indicates the model's stability across different time periods.")

except NameError:
    print("ERROR: The 'all_results' list was not found. Please ensure your cross-validation loop ran successfully and that the results were appended to a list named 'all_results'.")
except KeyError as e:
    print(f"KEY ERROR: Could not find the column {e} in the results. Please check that your evaluation function returns a dictionary with the correct keys (e.g., 'MAE', 'R2').")
    
mae_scores = [result['MAE'] for result in all_results]

# Then use mae_scores for calculations
advanced_mae_mean = np.mean(mae_scores)
advanced_mae_std = np.std(mae_scores)
improvement = BASELINE_RF_MAE - advanced_mae_mean
improvement_pct = (improvement / BASELINE_RF_MAE) * 100

print("--- FINAL PERFORMANCE ANALYSIS ---")
print(f"Baseline Random Forest MAE (Tier 1 Features): {BASELINE_RF_MAE:.4f}")
print(f"Random Forest MAE (Advanced Features):         {advanced_mae_mean:.4f} (± {advanced_mae_std:.4f})")
print("-" * 40)
print(f"Absolute Improvement: {improvement:.4f} MAE")
print(f"Percentage Improvement: {improvement_pct:.2f}%")

# Create a DataFrame for plotting
results_plot_df = pd.DataFrame({'MAE': mae_scores})

try:
    # Visualize the distribution of MAE scores
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=results_plot_df, y='MAE', palette='viridis', ax=ax, width=0.3)
    sns.stripplot(data=results_plot_df, y='MAE', color='black', size=8, jitter=0.1, ax=ax)

    ax.set_title('Distribution of MAE Scores for Enhanced RF Model', fontsize=16, pad=20)
    ax.set_ylabel('Mean Absolute Error (MAE)', fontsize=12)
    ax.set_xlabel('Random Forest with Advanced Features', fontsize=12)
    plt.tight_layout()
    plt.savefig('mae_distribution_engineeredRF_model.png')
    plt.show()
except Exception as e:
    print(f"ERROR: Could not generate the MAE distribution plot. Error details: {e}")

print("--- Starting Artifact Saving Process ---")

# --- 1. Save the Final Champion Model and its Feature List ---
# This is the model trained on the largest dataset (the final fold).
CHAMPION_MODEL_NAME = 'adv_engineered_RF_model.joblib'
CHAMPION_MODEL_PATH = os.path.join(PROJECT_ROOT, 'code', 'models', CHAMPION_MODEL_NAME)

model_artifact = {
    'model': rf_model,
    'features': FEATURES
}

try:
    joblib.dump(model_artifact, CHAMPION_MODEL_PATH)
    print(f"1/5: Successfully saved the champion model to: {CHAMPION_MODEL_PATH}")
except Exception as e:
    print(f"Error saving champion model: {e}")


# --- 2. Save the Engineered Training DataFrame ---
# This saves the result of your time-consuming feature engineering.
ENGINEERED_DATA_NAME = 'engineered_training_dataset_rf_advanced.csv'
ENGINEERED_DATA_PATH = os.path.join(PROCESSED_DIR, ENGINEERED_DATA_NAME)
# Also save as pickle for faster future loading
ENGINEERED_DATA_PICKLE = 'engineered_training_dataset_rf_advanced.pkl'
ENGINEERED_DATA_PICKLE_PATH = os.path.join(PROCESSED_DIR, ENGINEERED_DATA_PICKLE)

try:
    train_data_engineered.to_pickle(ENGINEERED_DATA_PICKLE_PATH)
    print(f"2b/5: Successfully saved the engineered training data (pickle) to: {ENGINEERED_DATA_PICKLE_PATH}")
except Exception as e:
    print(f"Error saving engineered data (pickle): {e}")

try:
    train_data_engineered.to_csv(ENGINEERED_DATA_PATH, index=False)
    print(f"2/5: Successfully saved the engineered training data to: {ENGINEERED_DATA_PATH}")
except NameError:
    print("Variable `train_data_engineered` not found. Skipping engineered data save.")
except Exception as e:
    print(f"Error saving engineered data: {e}")


# --- 3. Save the Cross-Validation Performance Metrics ---
CV_METRICS_NAME = 'cv_performance_metrics_rf_advanced.csv'
CV_METRICS_PATH = os.path.join(REPORTS_DIR / CV_METRICS_NAME)

try:
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(CV_METRICS_PATH, index=False)
    print(f"3/5: Successfully saved CV performance metrics to: {CV_METRICS_PATH}")
except NameError:
    print("Variable `all_results` not found. Skipping metrics save.")
except Exception as e:
    print(f"Error saving metrics: {e}")


# --- 4. Save the Feature Importances from Each Fold ---
CV_IMPORTANCES_NAME = 'cv_feature_importances_rf_advanced.csv'
CV_IMPORTANCES_PATH = os.path.join(REPORTS_DIR / CV_METRICS_NAME)

try:
    # Convert the dictionary of importances into a DataFrame
    importances_df = pd.DataFrame(feature_importances_per_fold, index=FEATURES)
    importances_df.to_csv(CV_IMPORTANCES_PATH)
    print(f"4/5: Successfully saved feature importances to: {CV_IMPORTANCES_PATH}")
except NameError:
    print("Variable `feature_importances_per_fold` not found. Skipping importances save.")
except Exception as e:
    print(f"Error saving feature importances: {e}")


# Create necessary directories
#os.makedirs(os.path.join(PROJECT_ROOT, 'models'), exist_ok=True)
#os.makedirs(os.path.join(PROJECT_ROOT, 'Data', 'processed'), exist_ok=True)
#os.makedirs(os.path.join(PROJECT_ROOT, 'reports', 'metrics'), exist_ok=True 

print("--- Artifact Saving Process Complete ---")