
# Imports
import requests
import pandas as pd
import argparse
from pathlib import Path
from dotenv import load_dotenv

# --- 1. CONFIGURATION AND AUTHENTICATION ---

# Set up argument parser to accept CLIENTID from the command line
parser = argparse.ArgumentParser(description="Retrieve hourly weather data from the Frost API.")
parser.add_argument(
    '--client_id', 
    type=str, 
    help="Your Frost API Client ID. If not provided, it will be read from a .env file."
)
args = parser.parse_args()

# Load API key from command line or .env file
if args.client_id:
    client_id = args.client_id
    print("Using CLIENTID provided from command-line argument.")
else:
    load_dotenv()
    client_id = os.getenv('CLIENTID')
    if client_id:
        print("Using CLIENTID found in .env file.")

if not client_id:
    raise ValueError("CLIENTID not found. Please provide it as a command-line argument (`--client_id YOUR_ID`) or set it in a .env file.")

# --- 2. API REQUEST PARAMETERS ---
FROST_ELEMENTS = (
    'air_temperature,'
    'sum(precipitation_amount PT1H),'
    'wind_speed,'
    'max(wind_speed_of_gust PT1H),'
    'relative_humidity,'
    'surface_snow_thickness,'
    'min(air_temperature P1D)'
)

endpoint = 'https://frost.met.no/observations/v0.jsonld'
parameters = {
    'sources': 'SN68860', # Trondheim (Voll)
    'elements': FROST_ELEMENTS,
    'referencetime': '2024-01-01/2024-12-31',
    'timeoffsets': 'default',
    'timeresolutions': 'PT1H,P1D,PT6H',
}

# --- 3. MAKING THE API REQUEST ---
print("\nRequesting weather data from Frost API...")
try:
    r = requests.get(endpoint, parameters, auth=(client_id, ''))
    r.raise_for_status() # This will raise an exception for bad status codes (4xx or 5xx)
    json_data = r.json()
except requests.exceptions.RequestException as e:
    print(f" API Request Failed: {e}")
    exit()

# --- 4. DATA VALIDATION AND PROCESSING ---
data = json_data.get('data', [])
if not data:
    print(" Warning: API request was successful but returned no data.")
    exit()

print(f"✅ Data retrieved successfully! Total records: {len(data)}")

# Process into a "long" DataFrame
df_list = []
for entry in data:
    for obs in entry['observations']:
        row = {
            'referenceTime': entry['referenceTime'],
            'elementId': obs['elementId'],
            'value': obs.get('value'),
            'unit': obs.get('unit'),
        }
        df_list.append(row)

if not df_list:
    print("No observations found in the API response.")
    exit()

long_df = pd.DataFrame(df_list)

# Pivot to "wide" format
print("\nPivoting data to 'wide' format...")
wide_df = long_df.pivot_table(
    index='referenceTime', columns='elementId', values='value', aggfunc='first'
)

# Clean column names
column_rename_map = {
    'air_temperature': 'temperature_celsius',
    'max(wind_speed_of_gust PT1H)': 'wind_gust_m_s',
    'min(air_temperature P1D)': 'min_temp_day',
    'relative_humidity': 'relative_humidity_percent',
    'sum(precipitation_amount PT1H)': 'precipitation_mm_hr',
    'surface_snow_thickness': 'snow_depth_cm',
    'wind_speed': 'wind_speed_m_s'
}
wide_df = wide_df.rename(columns=column_rename_map)

# Tidy up the DataFrame
wide_df.index = pd.to_datetime(wide_df.index)
for col in wide_df.columns:
    wide_df[col] = pd.to_numeric(wide_df[col], errors='coerce')

# Forward-fill data for columns with lower resolutions
fill_cols = ['snow_depth_cm', 'min_temp_day', 'relative_humidity_percent']
existing_fill_cols = [col for col in fill_cols if col in wide_df.columns]
if existing_fill_cols:
    print(f"\nForward-filling data for: {existing_fill_cols}")
    wide_df[existing_fill_cols] = wide_df[existing_fill_cols].fillna(method='ffill')

print("\n--- Sample of Cleaned 'Wide' Format Data ---")
print(wide_df.head())

# --- 5. SAVING THE FINAL DATAFRAME (ROBUST PATHS) ---
try:
    # Use the location of this script to find the project root
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    # Fallback for interactive environments
    PROJECT_ROOT = Path.cwd().parent

OUTPUT_DIR = PROJECT_ROOT / 'Data' / 'weather'
OUTPUT_FILENAME = 'weather_data_2024.csv'
full_path = OUTPUT_DIR / OUTPUT_FILENAME

# Create directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Save the file
wide_df.to_csv(full_path)
print(f"\n✅ Successfully processed and saved data to '{full_path}'")
