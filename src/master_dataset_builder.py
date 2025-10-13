# ---IMPORTS---
import os
import glob
import pandas as pd
import time
import pickle
import os
from pathlib import Path

# --- Configuration 
PROJECT_ROOT = Path.cwd()

YEAR = 2024

# --- Define paths for input and output files ---
DATA_DIR = PROJECT_ROOT / 'Data'
RIDERSHIP_DIR = DATA_DIR / 'atb_data' / str(YEAR)
WEATHER_PATH = DATA_DIR / 'weather' / f'weather_data_{YEAR}.csv'
EVENTS_PATH = DATA_DIR / 'processed' / f'event_features_{YEAR}.csv'

# --- Define output paths ---
OUTPUT_DIR = DATA_DIR / 'processed'
OUTPUT_FILENAME_CSV = f'master_feature_dataset_{YEAR}.csv'
OUTPUT_FILENAME_PKL = f'master_feature_dataset_{YEAR}.pkl'
FULL_OUTPUT_PATH_CSV = OUTPUT_DIR / OUTPUT_FILENAME_CSV
FULL_OUTPUT_PATH_PKL = OUTPUT_DIR / OUTPUT_FILENAME_PKL

# --- Validation  ---
print(f"Project Root determined to be: {PROJECT_ROOT}")
print(f"Ridership data will be read from: {RIDERSHIP_DIR}")
print(f"Master dataset will be saved to: {FULL_OUTPUT_PATH_CSV}")
FULL_OUTPUT_PATH_CSV = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME_CSV)
FULL_OUTPUT_PATH_PKL = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME_PKL)


def engineer_features_for_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers the set of chosen features for a given DataFrame chunk.
    """
    print("    - Engineering features for the current month...")
    
    # Temporal Features
    df['hour_of_day'] = df['datetime'].dt.hour.astype('int8')
    df['day_of_week'] = df['datetime'].dt.dayofweek.astype('int8')
    df['day_of_year'] = df['datetime'].dt.dayofyear.astype('int16')
    df['month'] = df['datetime'].dt.month.astype('int8')
    df['is_weekend'] = df['day_of_week'].isin([5, 6])
    df['is_rush_hour'] = (df['hour_of_day'].isin([7, 8, 15, 16])) & (~df['is_weekend'])

    # Route-specific Features
    # Memory-intensive step, performed on a smaller chunk
    max_sequence_per_trip = df.groupby(['route_id', 'trip_id'], observed=False)['stop_sequence'].transform('max')
    df['is_first_stop'] = df['stop_sequence'] == 0
    df['is_last_stop'] = df['stop_sequence'] == max_sequence_per_trip
    df['stop_sequence_normalized'] = (df['stop_sequence'] / max_sequence_per_trip).fillna(0).astype('float32')

    return df


def main():
    """Main function, orchestrating the entire data building process."""
    
    # Loading and Optimizing Shared Datasets
    print("\\n--- Loading Weather and Event Feature Sets ---")
    try:
        weather_df = pd.read_csv(WEATHER_PATH)
        events_df = pd.read_csv(EVENTS_PATH)
    except FileNotFoundError as e:
        print(f"--- ERROR: Prerequisite file not found. {e} ---")
        return

    if 'datetime' not in weather_df.columns and 'referenceTime' in weather_df.columns:
        weather_df.rename(columns={'referenceTime': 'datetime'}, inplace=True)
    
    weather_df['datetime'] = pd.to_datetime(weather_df['datetime']).dt.tz_localize(None)
    events_df['datetime'] = pd.to_datetime(events_df['datetime']).dt.tz_localize(None)
    
    print("Optimizing weather data types for memory efficiency...")
    for col in weather_df.columns:
        if weather_df[col].dtype == 'float64':
            weather_df[col] = weather_df[col].astype('float32')
        if weather_df[col].dtype == 'int64':
            weather_df[col] = weather_df[col].astype('int32')

    print("Weather and event data loaded and optimized successfully.")

    # ---Processing Raw Ridership Data and Writing to CSV in Chunks ---
    all_files = sorted(glob.glob(os.path.join(RIDERSHIP_DIR, "*.csv"))) # Sort files to process chronologically
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in directory: {RIDERSHIP_DIR}")

    print(f"\\n--- Step 2: Found {len(all_files)} monthly files to process ---")
    
    column_map = {
        'Dato': 'date', 'Linje': 'route_id', 'Tur': 'trip_id',
        'Holdeplassnummer': 'stop_id', 'Holdeplassekvens': 'stop_sequence',
        'Time': 'hour', 'Påstigende': 'boardings'
    }
    required_raw_cols = list(column_map.keys())
    
    # Store processed chunks for pickle creation 
    processed_chunks = []
    total_rows_processed = 0

    # Processing the the first file with a header, then appending the rest
    for i, filename in enumerate(all_files):
        print(f"  - Reading and processing {os.path.basename(filename)}...")
        try:
            # reading a single month into temporary DataFrame. 
            df = pd.read_csv(filename, sep=';', encoding='utf-8', low_memory=False)
            # Making sure all required columns are present
            if not all(col in df.columns for col in required_raw_cols):
                print(f"    Warning: Skipping file {os.path.basename(filename)} due to missing columns.")
                continue
            # Renaming columns to English. 
            df = df.rename(columns=column_map)
            df = df[list(column_map.values())]
            # Converting text columns to numberic, coercing errors to NaN.
            for col in ['trip_id', 'stop_id', 'stop_sequence', 'hour', 'boardings']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            # Dropping all rows missing essential information. 
            df.dropna(subset=['trip_id', 'stop_id', 'hour'], inplace=True)
            
            # Converting all clean columns to the most memory-efficient data types.
            df['boardings'] = df['boardings'].fillna(0).astype('int16')
            df['stop_sequence'] = df['stop_sequence'].fillna(0).astype('int16')
            df['trip_id'] = df['trip_id'].astype('int32')
            df['stop_id'] = df['stop_id'].astype('int32')
            df['hour'] = df['hour'].astype('int8')
            df['route_id'] = df['route_id'].astype('category')

            # Creating a unified datetime column. 
            df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['hour'].astype(str) + ':00:00', errors='coerce')
            df.dropna(subset=['datetime'], inplace=True)
            df.drop(columns=['date', 'hour'], inplace=True)

            # Merging and then engineering features for this month only
            merged_df = pd.merge(df, weather_df, on='datetime', how='left')
            merged_df = pd.merge(merged_df, events_df, on='datetime', how='left')
            # Engineering features for this month only. 
            featured_df = engineer_features_for_chunk(merged_df)

            # Impute weather NaNs for this chunk, using backward and forward fill. 
            weather_cols = [
                'temperature_celsius', 'wind_gust_m_s', 'min_temp_day',
                'relative_humidity_percent', 'precipitation_mm_hr', 'wind_speed_m_s'
            ]
            cols_to_fill = [col for col in weather_cols if col in featured_df.columns]
            for col in cols_to_fill:
                featured_df[col] = featured_df[col].bfill().ffill()
                
            # Store chunk for pickle creation later
            processed_chunks.append(featured_df.copy())
            total_rows_processed += len(featured_df)
            
            # Writing the month to the final CSV
            write_mode = 'w' if i == 0 else 'a'
            write_header = True if i == 0 else False
            
            print(f"    - Writing {len(featured_df)} processed rows to CSV...")
            featured_df.to_csv(FULL_OUTPUT_PATH_CSV, index=False, mode=write_mode, header=write_header)

        except Exception as e:
            print(f"    Warning: Could not process file {os.path.basename(filename)}. Error: {e}")
            continue
        
    # Create pickle file from all processed chunks
    if processed_chunks:
        print(f"\n--- Creating pickle from {len(processed_chunks)} processed chunks ---")
        print(f"Total rows to combine: {total_rows_processed:,}")
        
        # Combine all chunks into a single DataFrame
        start_time = time.time()
        print("Combining all processed chunks...")
        complete_df = pd.concat(processed_chunks, ignore_index=True)
        combine_time = time.time() - start_time
        print(f"  - Chunks combined in {combine_time:.2f} seconds.")
        
        # Sort by datetime for consistency
        print(" Sorting by datetime..")
        complete_df.sort_values('datetime', inplace=True)
        
        # Save as pickle
        print(f" Saving complete dataset as pickle: {FULL_OUTPUT_PATH_PKL}")
        start_time = time.time()
        complete_df.to_pickle(FULL_OUTPUT_PATH_PKL)
        pickle_time = time.time() - start_time
        print(f"  - Pickle saved in {pickle_time:.2f} seconds.")
        
        # File size comparison
        csv_size_mb = os.path.getsize(FULL_OUTPUT_PATH_CSV) / (1024**2) if os.path.exists(FULL_OUTPUT_PATH_CSV) else 0
        pkl_size_mb = os.path.getsize(FULL_OUTPUT_PATH_PKL) / (1024**2)
        
        print(f"\n--- File Size Comparison ---")
        print(f"CSV file:    {csv_size_mb:.1f} MB")
        print(f"Pickle file: {pkl_size_mb:.1f} MB")
        print(f"Pickle is {pkl_size_mb/csv_size_mb:.2f}x the size of CSV" if csv_size_mb > 0 else "")
        
        # Clean up memory
        del complete_df
        del processed_chunks
        
    print(f"\n--- SUCCESS! Master feature dataset created ---")
    print(f"CSV:    {FULL_OUTPUT_PATH_CSV}")
    print(f"Pickle: {FULL_OUTPUT_PATH_PKL}")
    print(f"Total rows processed: {total_rows_processed:,}")


if __name__ == '__main__':
    main()

