# --- 0. IMPORTS ---
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score

# --- 1. CONFIGURATION AND PATHS ---
PROJECT_ROOT = Path.cwd()
YEAR = 2024

DATA_DIR = PROJECT_ROOT / 'Data'
PROCESSED_DIR = DATA_DIR / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models'
REPORTS_DIR = PROJECT_ROOT / 'reports'
FIGURES_OUTPUT_DIR = REPORTS_DIR / 'figures'

# Create output directory for figures if it doesn't exist
FIGURES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Input Files ---
CHAMPION_MODEL_PATH = MODELS_DIR / 'final_bus_ridership_model.joblib'
HOLDOUT_TEST_PATH = PROCESSED_DIR / 'final_holdout_test_set_advanced.pkl'
STOP_LIST_PATH = PROCESSED_DIR / 'stop_list_atb.csv'


print("--- Starting Business Impact Analysis ---")
analysis_df = None

try:
    # Load Model
    print(f"Loading final tuned model from: {CHAMPION_MODEL_PATH}")
    artifact = joblib.load(CHAMPION_MODEL_PATH)
    final_model = artifact['model']
    FINAL_MODEL_FEATURES = artifact['features']
    print(f"Model loaded. Expecting {len(FINAL_MODEL_FEATURES)} features.")

    # Load Hold-Out Test Data
    print(f"\nLoading PRE-ENGINEERED hold-out test data from: {HOLDOUT_TEST_PATH}")
    test_df_engineered = pd.read_pickle(HOLDOUT_TEST_PATH)
    print(f"Loaded {len(test_df_engineered):,} records for the test period.")

    # Load Stop List for Name Mapping
    print(f"\nLoading stop list from: {STOP_LIST_PATH}")
    stop_list_df = pd.read_csv(STOP_LIST_PATH)
    print(f"Loaded {len(stop_list_df)} stop names.")

except Exception as e:
    print(f"An error occurred during asset loading: {e}")
    exit()

# --- 3. MAKE PREDICTIONS & CREATE ANALYSIS DATAFRAME ---
try:
    print("\nMaking predictions on the engineered test set...")
    X_test = test_df_engineered[FINAL_MODEL_FEATURES]
    predictions = np.expm1(final_model.predict(X_test))

    # Create Initial Analysis DataFrame
    analysis_df = test_df_engineered.copy()
    analysis_df['predictions'] = predictions
    print("Predictions generated.")

    # Enrich with stop names for better analysis
    if 'stop_name' not in analysis_df.columns:
        analysis_df = pd.merge(analysis_df, stop_list_df, on='stop_id', how='left')
    print("`analysis_df` is ready for business scenarios.")

except Exception as e:
    print(f"An error occurred during prediction: {e}")
    exit()


# --- 4. BUSINESS SCENARIO ANALYSIS ---

def scenario_1_high_demand_events(df):
    """Identifies high-demand events for a specific stop."""
    print("\n--- Scenario 1: Identifying High-Demand Events (Overcrowding Indicator) ---")
    stop_id_scenario = 71181  # Prinsens gate P2
    
    stop_analysis_df = df[df['stop_id'] == stop_id_scenario].copy()
    if stop_analysis_df.empty:
        print(f"Warning: No data for stop_id {stop_id_scenario}. Skipping scenario.")
        return

    high_demand_threshold = stop_analysis_df['predictions'].quantile(0.99)
    print(f"Data-driven threshold (99th percentile) for stop {stop_id_scenario}: {high_demand_threshold:.2f} boardings\n")
    
    high_demand_events = stop_analysis_df[stop_analysis_df['predictions'] > high_demand_threshold]
    print(f"Found {len(high_demand_events)} instances of predicted high-demand events.")
    print("\nTop 10 Predicted High-Demand Events (sorted by prediction):")
    print(high_demand_events[['datetime', 'route_id', 'boardings', 'predictions']].sort_values('predictions', ascending=False).head(10))

def scenario_2_rbk_game_day(df):
    """Analyzes the ridership uplift during post-game exodus hours."""
    print("\n--- Scenario 2: RBK Game Day Service Optimization (Targeted Analysis) ---")
    
    all_game_day_hours = df[df['is_rbk_home_game'] == True]['hour_of_day'].unique()
    if len(all_game_day_hours) == 0:
        print("No RBK game day hours found in the test period. Skipping analysis.")
        return

    post_game_exodus_hour = sorted(all_game_day_hours)[-1:]
    print(f"Analysis focusing on post-game exodus hour: {post_game_exodus_hour}\n")

    peak_hour_df = df[df['hour_of_day'].isin(post_game_exodus_hour)]
    routes_to_analyze = [1, 2, 10, 11]
    
    game_day_impact = peak_hour_df[peak_hour_df['route_id'].isin(routes_to_analyze)] \
        .groupby(['route_id', 'is_rbk_home_game'])['predictions'].mean().unstack()

    if True not in game_day_impact.columns: game_day_impact[True] = np.nan
    if False not in game_day_impact.columns: game_day_impact[False] = np.nan
        
    game_day_impact.columns = ['Avg Pred (Peak Hour, Non-Game Day)', 'Avg Pred (Peak Hour, Game Day)']
    game_day_impact['Uplift (%)'] = ((game_day_impact['Avg Pred (Peak Hour, Game Day)'] - game_day_impact['Avg Pred (Peak Hour, Non-Game Day)']) / game_day_impact['Avg Pred (Peak Hour, Non-Game Day)']) * 100

    print("Average Predicted Boardings and Uplift during the Post-Game Exodus Hour:")
    print(game_day_impact.sort_values('Uplift (%)', ascending=False))

    plot_data = game_day_impact.dropna(subset=['Uplift (%)']).sort_values('Uplift (%)', ascending=False)
    if not plot_data.empty:
        plt.figure(figsize=(10, 6))
        plot_data[['Uplift (%)']].plot(kind='bar', legend=None, ax=plt.gca())
        plt.title('Predicted Ridership Uplift (%) on RBK Game Days (Post-Game Hour)')
        plt.ylabel('Uplift (%)')
        plt.xlabel('Route ID')
        plt.xticks(rotation=0)
        plt.savefig(FIGURES_OUTPUT_DIR / 'rbk_game_day_uplift.png', dpi=300, bbox_inches='tight')
        plt.show()

def scenario_3_underutilized_stops(df):
    """Identifies consistently underutilized stops."""
    print("\n--- Scenario 3: Network Optimization through Underutilized Stop Identification ---")
    stop_performance = df.groupby(['stop_id', 'stop_name']).agg(
        avg_prediction=('predictions', 'mean'),
        avg_actual=('boardings', 'mean'),
        total_observations=('datetime', 'count')
    ).reset_index()

    underutilized_threshold = 0.5
    underutilized_stops = stop_performance[
        (stop_performance['avg_prediction'] < underutilized_threshold) &
        (stop_performance['avg_actual'] < underutilized_threshold) &
        (stop_performance['total_observations'] > 100)
    ].sort_values('avg_prediction')

    print(f"Found {len(underutilized_stops)} stops consistently predicted as underutilized.")
    print("\nTop 10 Most Underutilized Stops (based on model predictions):")
    print(underutilized_stops.head(10))

def final_performance_evaluation(df):
    """Calculates and prints the final model performance on the hold-out set."""
    print("\n--- Final Champion Model Performance on Unseen Hold-Out Test Set (Oct-Dec 2024) ---")
    y_true = df['boardings']
    y_pred = df['predictions']
    final_mae = mean_absolute_error(y_true, y_pred)
    final_r2 = r2_score(y_true, y_pred)
    print(f"Mean Absolute Error (MAE): {final_mae:.4f}")
    print(f"R-squared (R²):              {final_r2:.4f}")
    print("\nThis represents the final, most realistic estimate of the model's performance.")

# --- 5. MAIN EXECUTION ---
if __name__ == '__main__':
    if analysis_df is not None:
        sns.set_theme(style="whitegrid")
        
        # Run all analysis functions
        scenario_1_high_demand_events(analysis_df)
        scenario_2_rbk_game_day(analysis_df)
        scenario_3_underutilized_stops(analysis_df)
        final_performance_evaluation(analysis_df)
        
        print("\n\n--- Business Impact Analysis Complete ---")
        print(f"All plots have been saved to: {FIGURES_OUTPUT_DIR}")
    else:
        print("\n--- Analysis Skipped: `analysis_df` was not created due to an earlier error. ---")
