import os
import json
import pandas as pd
from datetime import timedelta
from pathlib import Path

# Configuration
# Defining the year to generate features for.
YEAR = 2024

# Defining root directory for project - use current working directory
PROJECT_ROOT = Path.cwd()

# Define paths for input and output files, relative to the project root.
CONFIG_PATH = PROJECT_ROOT / 'events_config.json'
OUTPUT_DIR = PROJECT_ROOT / 'Data' / 'processed'
OUTPUT_FILENAME = f'event_features_{YEAR}.csv'
FULL_OUTPUT_PATH = OUTPUT_DIR / OUTPUT_FILENAME


def build_event_features(config_path: str, output_path: str, year: int):
    """
    Loads an event configuration JSON and creates a feature dataset with
    boolean flags for every hour of the specified year.
    """
    print(f"--- Starting Event Feature Generation for {year} ---")

    # --- Creating Master Hourly Timeline ---
    print("Creating hourly timeline...")
    start_date = f'{year}-01-01 00:00:00'
    end_date = f'{year}-12-31 23:00:00'
    hourly_range = pd.date_range(start=start_date, end=end_date, freq='h')
    df = pd.DataFrame(hourly_range, columns=['datetime'])
    print(f"Timeline created with {len(df)} hours.")

    # ---Loading Event Configuration from JSON file ---
    print(f"Loading event configuration from {config_path}")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"--- ERROR: Configuration file not found at {config_path} ---")
        return

    # --- Initializing Feature Columns ---
    # These column names must match the keys in the JSON and the feature plan.
    feature_columns = [
        'is_rbk_home_game', 'is_major_event', 'is_special_event',
        'is_other_holiday', 'is_summer_holiday', 'is_e_scooter_season',
        'is_student_semester'
    ]
    for col in feature_columns:
        df[col] = False

    # --- Processing Events and Applying Features ---
    print("Processing events and applying features...")

    # Helper function to handle date ranges with optional times
    def apply_event_mask(events, column_name):
        for event in events:
            start = event['start']
            end = event['end']
            start_time = event.get('start_time', '00:00')
            end_time = event.get('end_time', '23:59')

            start_dt = pd.to_datetime(f"{start} {start_time}")
            end_dt = pd.to_datetime(f"{end} {end_time}")
            
            # Handling cases like an event ending at 03:00 the next day
            if end_dt < start_dt:
                 end_dt += timedelta(days=1)

            mask = (df['datetime'] >= start_dt) & (df['datetime'] <= end_dt)
            df.loc[mask, column_name] = True
        print(f"  - Applied {len(events)} periods for '{column_name}'")

    # Processing events that are lists of dictionaries (games, major events, etc.)
    apply_event_mask(config.get('rbk_home_games', []), 'is_rbk_home_game')
    apply_event_mask(config.get('major_events', []), 'is_major_event')
    apply_event_mask(config.get('special_events', []), 'is_special_event')
    apply_event_mask(config.get('student_semester', []), 'is_student_semester')

    # Processing events that are single dictionaries (summer holiday, scooter season)
    apply_event_mask([config['summer_holiday']], 'is_summer_holiday')
    apply_event_mask([config['e_scooter_season']], 'is_e_scooter_season')

    # Processing list of single-day holidays
    holidays = config.get('other_holidays', [])
    holiday_dates = pd.to_datetime(holidays).date
    mask = df['datetime'].dt.date.isin(holiday_dates)
    df.loc[mask, 'is_other_holiday'] = True
    print(f"  - Applied {len(holidays)} single-day holidays for 'is_other_holiday'")

    # --- Saving the Final Dataset ---
    print(f"Saving feature set to {output_path}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print("\n--- Successfully created the event features dataset! ---")
    print(f"Final DataFrame shape: {df.shape}")
    print("\nPreview of generated features for a sample date:")
    print(df[df['datetime'].dt.date == pd.to_datetime('2024-08-15').date()].tail())


if __name__ == '__main__':
    # Make sure to set the correct year if not running for 2024
    build_event_features(config_path=CONFIG_PATH, output_path=FULL_OUTPUT_PATH, year=YEAR)
