"""
Download Medical MNIST dataset from Kaggle using credentials from .env file
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

# Load environment variables from .env file
from dotenv import load_dotenv

# Try to load from .env in parent directories
env_paths = [
    Path(__file__).parent / '.env',
    Path(__file__).parent.parent / '.env',
    Path(__file__).parent.parent.parent / '.env',
]

env_loaded = False
for env_path in env_paths:
    if env_path.exists():
        load_dotenv(env_path)
        env_loaded = True
        print(f"Loaded environment from: {env_path}")
        break

if not env_loaded:
    print("Warning: No .env file found")

# Set Kaggle credentials from environment
if os.getenv('KAGGLE_USERNAME'):
    os.environ['KAGGLE_USERNAME'] = os.getenv('KAGGLE_USERNAME')
if os.getenv('KAGGLE_KEY'):
    os.environ['KAGGLE_KEY'] = os.getenv('KAGGLE_KEY')

from config.paths import Paths
from src.data_loader import download_medical_mnist_kaggle, check_data_exists


def main():
    print("=" * 60)
    print("Medical MNIST Data Download")
    print("=" * 60)
    
    # Ensure directories exist
    Paths.ensure_directories()
    
    raw_data_dir = Paths.RAW_DATA_DIR
    
    # Check if data already exists
    if check_data_exists(raw_data_dir):
        print(f"\nData already exists in: {raw_data_dir}")
        print("Skipping download.")
        return
    
    print(f"\nDownloading Medical MNIST dataset to:")
    print(f"  {raw_data_dir}")
    print("\nThis may take a few minutes depending on your connection...")
    print("-" * 60)
    
    try:
        download_medical_mnist_kaggle(raw_data_dir)
        
        print("\n" + "=" * 60)
        print("Download complete!")
        print("=" * 60)
        
        if check_data_exists(raw_data_dir):
            print("\nData verified successfully!")
            print("You can now run: python main.py --running_mode debug")
        else:
            print("\nWarning: Data structure not as expected.")
            print("Please check the contents of:", raw_data_dir)
            
    except Exception as e:
        print(f"\nError: {e}")
        print("\n" + "=" * 60)
        print("Setup Instructions:")
        print("=" * 60)
        print("""
1. Get your Kaggle API credentials:
   - Go to https://www.kaggle.com/account
   - Click 'Create New API Token'
   - This downloads a kaggle.json file

2. Set up credentials:
   Windows: Copy kaggle.json to:
       C:/Users/<YourUsername>/.kaggle/kaggle.json
   
   Linux/Mac: Copy kaggle.json to:
       ~/.kaggle/kaggle.json

3. Authenticate:
   kaggle auth login

4. Run this script again:
   python download_data.py

Alternative: Download manually from:
https://www.kaggle.com/datasets/andrewmvd/medical-mnist
And extract the files to: data/raw/
""")
        raise


if __name__ == "__main__":
    main()
