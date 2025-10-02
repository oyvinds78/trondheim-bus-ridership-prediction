# --- 0. IMPORTS ---
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# --- 1. CONFIGURATION 
PROJECT_ROOT = Path.cwd()

YEAR = 2024
DATA_DIR = PROJECT_ROOT / 'Data'
PROCESSED_DIR = DATA_DIR / 'processed'
REPORTS_DIR = PROJECT_ROOT / 'reports'
FIGURES_OUTPUT_DIR = REPORTS_DIR / 'figures'

# Create output directory for figures if it doesn't exist
FIGURES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(" --- Loading dataset ---..")
DATASET_PICKLE = PROCESSED_DIR / f'master_feature_dataset_{YEAR}_with_advanced_features.pkl'
DATASET_CSV = PROCESSED_DIR / f'master_feature_dataset_{YEAR}_with_advanced_features.csv'

print("--- Starting Exploratory Data Analysis ---")

if DATASET_PICKLE.exists():
    print(f"Loading dataset from pickle: {DATASET_PICKLE}")
    df = pd.read_pickle(DATASET_PICKLE)
elif DATASET_CSV.exists():
    print(f"Loading dataset from CSV: {DATASET_CSV}")
    df = pd.read_csv(DATASET_CSV, parse_dates=['datetime'], low_memory=False)
else:
    raise FileNotFoundError(f"No master dataset found in {PROCESSED_DIR}")

print(f"✅ Dataset loaded with {df.shape[0]:,} rows and {df.shape[1]} columns")

# Ensure datetime is a datetime object
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)

# --- 3. ANALYSIS FUNCTIONS ---

def analyze_daily_patterns(data):
    """Analyzes and plots the average daily ridership pattern."""
    print("\n--- Analyzing Daily Ridership Patterns ---")
    hourly_boardings = data.groupby(data.index.hour)['boardings'].mean()
    
    plt.figure(figsize=(12, 6))
    hourly_boardings.plot(kind='line', marker='o', linestyle='-')
    plt.title('Average Daily Ridership Pattern (Diurnal Cycle)', fontsize=16)
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Average Boardings', fontsize=12)
    plt.xticks(np.arange(0, 24, 1))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Save the figure
    plt.savefig(FIGURES_OUTPUT_DIR / 'daily_ridership_pattern.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_weekly_patterns(data):
    """Analyzes and plots ridership patterns across the week."""
    print("\n--- Analyzing Weekly Ridership Patterns ---")
    daily_avg = data.groupby(data.index.day_name())['boardings'].mean()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_avg = daily_avg.reindex(day_order)

    plt.figure(figsize=(12, 6))
    daily_avg.plot(kind='bar', color=sns.color_palette('viridis', 7))
    plt.title('Average Ridership by Day of the Week', fontsize=16)
    plt.xlabel('Day of the Week', fontsize=12)
    plt.ylabel('Average Boardings', fontsize=12)
    plt.xticks(rotation=45)
    
    plt.savefig(FIGURES_OUTPUT_DIR / 'weekly_ridership_pattern.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_weather_impact(data):
    """Analyzes and plots the correlation of weather features with ridership."""
    print("\n--- Analyzing Weather Impact ---")
    weather_features = [
        'boardings', 'temperature_celsius', 'precipitation_mm_hr',
        'wind_speed_m_s', 'snow_depth_cm', 'wind_gust_m_s'
    ]
    # Check for snow_depth_cm as it might not be in all datasets
    if 'snow_depth_cm' not in data.columns:
        weather_features.remove('snow_depth_cm')
        
    weather_corr = data[weather_features].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(weather_corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
    plt.title('Weather Feature Correlation with Boardings', fontsize=16)
    
    plt.savefig(FIGURES_OUTPUT_DIR / 'weather_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_event_impact(data, event_col, event_name):
    """Analyzes and plots the impact of a specific event type."""
    print(f"\n--- Analyzing Impact of {event_name} ---")
    if event_col not in data.columns:
        print(f"Event column '{event_col}' not found. Skipping analysis.")
        return
        
    event_impact = data.groupby(event_col)['boardings'].mean()
    print(event_impact)

    # Perform t-test for statistical significance
    event_days = data[data[event_col] == True]['boardings']
    non_event_days = data[data[event_col] == False]['boardings']
    
    if len(event_days) > 1 and len(non_event_days) > 1:
        t_stat, p_value = stats.ttest_ind(event_days, non_event_days, equal_var=False)
        print(f"T-test p-value: {p_value:.3e}")
        
        plt.figure(figsize=(8, 5))
        event_impact.plot(kind='bar', color=['skyblue', 'salmon'])
        plt.title(f'Average Boardings on {event_name} vs. Non-Event Days', fontsize=14)
        plt.xticks([0, 1], ['Non-Event Days', f'{event_name} Days'], rotation=0)
        plt.ylabel('Average Boardings')
        
        plt.savefig(FIGURES_OUTPUT_DIR / f'{event_col}_impact.png', dpi=300, bbox_inches='tight')
        plt.show()
    else:
        print("Not enough data to perform statistical test or plot for this event.")


# --- 4. MAIN EXECUTION ---
if __name__ == '__main__':
    sns.set_theme(style="whitegrid")

    # Run all analysis functions
    analyze_daily_patterns(df)
    analyze_weekly_patterns(df)
    analyze_weather_impact(df)
    
    # Analyze different event types
    analyze_event_impact(df, 'is_rbk_home_game', 'RBK Home Game')
    analyze_event_impact(df, 'is_major_event', 'Major Event')
    analyze_event_impact(df, 'is_student_semester', 'Student Semester')
    analyze_event_impact(df, 'is_summer_holiday', 'Summer Holiday')
    
    print("\n\n✅ --- EDA ANALYSIS COMPLETE ---")
    print(f"All plots have been saved to: {FIGURES_OUTPUT_DIR}")
