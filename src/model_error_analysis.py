# --- 0. IMPORTS ---
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- 1. CONFIGURATION AND PATHS ---
PROJECT_ROOT = Path.cwd().parent

YEAR = 2024
DATA_DIR = PROJECT_ROOT / 'Data'
PROCESSED_DIR = DATA_DIR / 'processed'
MODELS_DIR = PROJECT_ROOT /  'models'
REPORTS_DIR = PROJECT_ROOT / 'reports'
FIGURES_OUTPUT_DIR = REPORTS_DIR / 'figures'

# Create output directory for figures if it doesn't exist
FIGURES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Input Files ---
CHAMPION_MODEL_PATH = MODELS_DIR / 'final_bus_ridership_model.joblib'
HOLDOUT_TEST_PATH = PROCESSED_DIR / 'final_holdout_test_set_advanced.pkl'
STOP_LIST_PATH = PROCESSED_DIR / 'stop_list_atb.csv'

# --- 2. ASSET LOADING ---
print("--- Starting Model Error Analysis ---")
analysis_df = None

try:
    # Load Model
    print(f"Loading final tuned model from: {CHAMPION_MODEL_PATH}")
    artifact = joblib.load(CHAMPION_MODEL_PATH)
    final_model = artifact['model']
    FINAL_MODEL_FEATURES = artifact['features']
    print(f" Model loaded. Expecting {len(FINAL_MODEL_FEATURES)} features.")

    # Load Hold-Out Test Data
    print(f"Loading PRE-ENGINEERED hold-out test data from: {HOLDOUT_TEST_PATH}")
    test_df = pd.read_pickle(HOLDOUT_TEST_PATH)
    print(f" Loaded {len(test_df):,} records for the test period.")

    # Load Stop List for Name Mapping
    print(f"Loading stop list from: {STOP_LIST_PATH}")
    stop_list_df = pd.read_csv(STOP_LIST_PATH)
    print(f" Loaded {len(stop_list_df)} stop names.")

except Exception as e:
    print(f"❌ An error occurred during asset loading: {e}")
    exit()

# --- 3. MAKE PREDICTIONS & CREATE ANALYSIS DATAFRAME ---
try:
    print("\nMaking predictions on the engineered test set...")
    X_test = test_df[FINAL_MODEL_FEATURES]
    predictions = final_model.predict(X_test)

    # Create the base analysis DataFrame
    analysis_df = test_df.copy()
    analysis_df['predictions'] = predictions
    analysis_df['residual'] = analysis_df['boardings'] - analysis_df['predictions']
    analysis_df['abs_error'] = np.abs(analysis_df['residual'])

    # Enrich with stop names for better analysis
    if 'stop_name' not in analysis_df.columns:
        analysis_df = pd.merge(analysis_df, stop_list_df, on='stop_id', how='left')
    
    print("✅ Predictions generated and analysis DataFrame is ready.")

except Exception as e:
    print(f" An error occurred during prediction: {e}")
    exit()


# --- 4. ANALYSIS FUNCTIONS ---

def analyze_error_by_hour(df):
    """Analyzes and plots model error by hour of the day."""
    print("\n--- Analyzing Error by Hour of Day ---")
    error_by_hour = df.groupby('hour_of_day')['abs_error'].mean()
    
    plt.figure(figsize=(12, 6))
    error_by_hour.plot(kind='bar', color='skyblue')
    plt.title('Average Model Error (MAE) by Hour of Day', fontsize=16)
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Mean Absolute Error', fontsize=12)
    plt.xticks(rotation=0)
    plt.savefig(FIGURES_OUTPUT_DIR / 'error_by_hour.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_error_by_route(df):
    """Analyzes and plots model error by bus route."""
    print("\n--- Analyzing Error by Route ID ---")
    error_by_route = df.groupby('route_id')['abs_error'].mean().sort_values(ascending=False)
    
    print("Top 5 Routes with Highest MAE:")
    print(error_by_route.head(5))
    
    plt.figure(figsize=(14, 7))
    error_by_route.head(20).plot(kind='bar', color='salmon')
    plt.title('Top 20 Routes by Average Model Error (MAE)', fontsize=16)
    plt.xlabel('Route ID', fontsize=12)
    plt.ylabel('Mean Absolute Error', fontsize=12)
    plt.xticks(rotation=45)
    plt.savefig(FIGURES_OUTPUT_DIR / 'error_by_route.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_error_by_stop(df):
    """Identifies the stops with the highest and lowest prediction errors."""
    print("\n--- Analyzing Error by Stop ID ---")
    error_by_stop = df.groupby(['stop_id', 'stop_name'])['abs_error'].agg(['mean', 'count']).rename(columns={'mean': 'MAE'})
    
    # Filter for stops with a significant number of observations for reliable stats
    reliable_stops = error_by_stop[error_by_stop['count'] > 100]
    
    print("\nTop 10 Hardest-to-Predict Stops (Highest MAE):")
    print(reliable_stops.sort_values('MAE', ascending=False).head(10))
    
    print("\nTop 10 Easiest-to-Predict Stops (Lowest MAE):")
    print(reliable_stops.sort_values('MAE', ascending=True).head(10))

def analyze_residuals(df):
    """Creates a residual plot to check for bias."""
    print("\n--- Analyzing Residuals ---")
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='predictions', y='residual', data=df.sample(50000, random_state=42), alpha=0.3)
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Residuals vs. Predicted Values', fontsize=16)
    plt.xlabel('Predicted Boardings', fontsize=12)
    plt.ylabel('Residual (Actual - Predicted)', fontsize=12)
    plt.savefig(FIGURES_OUTPUT_DIR / 'residual_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

# --- 5. MAIN EXECUTION ---
if __name__ == '__main__':
    if analysis_df is not None:
        sns.set_theme(style="whitegrid")
        
        # Run all analysis functions
        analyze_error_by_hour(analysis_df)
        analyze_error_by_route(analysis_df)
        analyze_error_by_stop(analysis_df)
        analyze_residuals(analysis_df)
        
        print("\n\n --- Model Error Analysis Complete ---")
        print(f"All plots have been saved to: {FIGURES_OUTPUT_DIR}")
    else:
        print("\n--- Analysis Skipped: `analysis_df` was not created due to an earlier error. ---")
