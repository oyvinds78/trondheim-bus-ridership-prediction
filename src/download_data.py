"""
This script downloads the raw bus ridership data from a public Google Drive folder.
"""
import os
import gdown

def download_data():
    """
    Downloads all necessary data from Google Drive if it doesn't already exist.
    """
    # --- 1. Download Weather Data ---
    weather_file_path = 'Data/weather/weather_data_2024.csv'
    weather_dir = 'Data/weather'
    
    if not os.path.exists(weather_dir):
        os.makedirs(weather_dir)

    if not os.path.exists(weather_file_path):
        print("Downloading weather data...")
        weather_url = 'https://drive.google.com/uc?id=1zaigqqX3QkUE5Z9Xy7qfK4UT5ShBErTS'
        gdown.download(url=weather_url, output=weather_file_path, quiet=False)
    else:
        print("Weather data already exists locally.")

    # --- 2. Download Bus Ridership Data ---
    print("\nChecking for raw bus ridership files...")
    atb_folder_url = "https://drive.google.com/drive/folders/1HufdGE0mBCMklXfOiKGSUPEqGYPQW_bW?usp=sharing"
    atb_output_folder = 'Data/atb_data/2024'
    # gdown.download_folder will create the output folder and skip existing files
    gdown.download_folder(url=atb_folder_url, output=atb_output_folder, quiet=False, use_cookies=False)

    print("\nData download check complete.")

if __name__ == '__main__':
    download_data()
