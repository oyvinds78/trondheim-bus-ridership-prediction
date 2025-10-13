# --- 0. IMPORTS ---
import pandas as pd
import numpy as np
import joblib
import os
import time
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
import pickle

# --- 1. CONFIGURATION ---
PROJECT_ROOT = Path.cwd()

YEAR = 2024
DATA_DIR = PROJECT_ROOT / 'Data'
PROCESSED_DIR = DATA_DIR / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models'

# Create output directory if it doesn't exist
MODELS_DIR.mkdir(exist_ok=True)

# --- Input File ---
FINAL_TRAIN_PATH = PROCESSED_DIR / 'master_feature_dataset_2024_with_advanced_features.pkl'

# --- Output File ---
FINAL_MODEL_PATH = MODELS_DIR / 'final_bus_ridership_model.joblib'

# --- 2. DATA LOADING ---
try:
    print(f"Loading feature-engineered dataset from: {FINAL_TRAIN_PATH}...")
    full_df = pd.read_pickle(FINAL_TRAIN_PATH)
    print(f"Successfully loaded {len(full_df):,} records.")
    
    # Ensure 'datetime' column is in datetime format for splitting
    if 'datetime' not in full_df.columns:
        raise KeyError("'datetime' column not found in the dataset. Cannot perform time-based split.")
    full_df['datetime'] = pd.to_datetime(full_df['datetime'])
    print("Converted 'datetime' column to datetime objects.")

except FileNotFoundError:
    print(f"ERROR: The feature-engineered file was not found at {FINAL_TRAIN_PATH}. Please run the preceding pipeline steps first.")
    exit()
except KeyError as e:
    print(f"ERROR: {e}")
    exit()

# --- 3. SPLIT DATA INTO TRAINING AND HOLDOUT SETS ---
print(f"\nSplitting data into training and holdout sets based on the date 2024-09-30...")
split_date = pd.to_datetime(f'{YEAR}-09-30')

train_df = full_df[full_df['datetime'] <= split_date].copy()
holdout_df = full_df[full_df['datetime'] > split_date].copy()

print(f"Training set (until 2024-09-30): {len(train_df):,} records")
print(f"Holdout set (after 2024-09-30):  {len(holdout_df):,} records")

# --- 4. DEFINE FINAL MODEL AND FEATURES ---
# These are the optimal parameters discovered during hyperparameter tuning.
best_params = {
    'n_estimators': 150,
    'max_depth': 18,
    'min_samples_leaf': 6,
    'max_features': 'sqrt',
    'random_state': 42,
    'n_jobs': -1
}

# Initialize the final model with these parameters
final_model = RandomForestRegressor(**best_params)

# Define the final feature set and target variable
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

print("\nFinal model initialized with optimal hyperparameters.")
print(best_params)

# --- 5. TRAIN THE FINAL MODEL ---
if not train_df.empty:
    print(f"\nPreparing to train the final model on the training set ({len(train_df):,} records)...")
    
    # Prepare the training data
    X_train = train_df[FEATURES]
    y_train = np.log1p(train_df[TARGET])
    
    # Start the training process
    print("Training started... This will take a significant amount of time.")
    start_time = time.time()
    
    final_model.fit(X_train, y_train)
    
    end_time = time.time()
    training_duration_minutes = (end_time - start_time) / 60
    
    print(f"\nFinal model training complete!")
    print(f"Total training time: {training_duration_minutes:.2f} minutes.")

    # --- 6. SAVE THE FINAL MODEL ARTIFACT AND DATASETS ---
    # Bundle the model and its feature list together for robust deployment
    model_artifact = {
        'model': final_model,
        'features': FEATURES
    }
    
    try:
        joblib.dump(model_artifact, FINAL_MODEL_PATH)
        print(f"Successfully saved the final model artifact to: {FINAL_MODEL_PATH}")
    except Exception as e:
        print(f"Error saving the final model artifact: {e}")

    # Save the training and holdout sets
    try:
        # Output paths for the datasets
        final_training_set_path = PROCESSED_DIR / 'final_training_set_advanced.pkl'
        final_holdout_test_set_path = PROCESSED_DIR / 'final_holdout_test_set_advanced.pkl'

        train_df.to_pickle(final_training_set_path)
        print(f"Successfully saved the final training set to: {final_training_set_path}")
        
        holdout_df.to_pickle(final_holdout_test_set_path)
        print(f"Successfully saved the final holdout test set to: {final_holdout_test_set_path}")

    except Exception as e:
        print(f"Error saving the datasets: {e}")
else:
    print("Training data is empty. Cannot train the final model.")
