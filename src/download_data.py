"""
This script downloads the raw bus ridership data from a public Google Drive folder.
"""
import os
import gdown

def download_data():
    """
    Downloads data from Google Drive if it doesn't already exist.
    """
    output_folder = 'Data/atb_data/2024'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # URL of the public Google Drive folder
    folder_url = "https://drive.google.com/drive/folders/1HufdGE0mBCMklXfOiKGSUPEqGYPQW_bW?usp=sharing"

    print("Checking for raw data files...")
    
    # Get list of files already in the output folder
    existing_files = os.listdir(output_folder)
    
    # Get list of files in the google drive folder
    files = gdown.list_files(folder_url)

    for file in files:
        file_path = os.path.join(output_folder, file['name'])
        if file['name'] not in existing_files:
            print(f"Downloading {file['name']}...")
            gdown.download(id=file['id'], output=file_path, quiet=False)
        else:
            print(f"Found {file['name']} locally.")

    print("Data download check complete.")

if __name__ == '__main__':
    download_data()
