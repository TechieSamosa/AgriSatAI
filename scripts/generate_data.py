import os
import sys

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from src.utils.synthetic_data import generate_dataset

if __name__ == "__main__":
    print("Generating synthetic satellite imagery data...")
    generate_dataset()
    print(f"Data generated in {os.path.join(settings.RAW_DATA_DIR, 'satellite_imagery')}")
