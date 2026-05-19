import os
import sys
import pandas as pd
import glob

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings
from src.ml.train import train_ensemble_classifier, train_yield_regressor

def get_latest_csv(base_dir):
    """Helper to read the processed CSV output"""
    csv_file = os.path.join(base_dir, 'processed_features_csv.csv')
    if not os.path.exists(csv_file):
        raise FileNotFoundError("No processed CSV data found. Run ETL first.")
    return pd.read_csv(csv_file)

if __name__ == "__main__":
    print("--- STARTING MODEL TRAINING PIPELINE ---")
    
    try:
        df = get_latest_csv(os.path.join(settings.BASE_DIR, 'output'))
        
        # Train Models
        clf_model, importances = train_ensemble_classifier(df)
        print("\nFeature Importances:")
        for feat, imp in importances.items():
            print(f"  {feat}: {imp:.4f}")
            
        reg_model = train_yield_regressor(df)
        
        print("\nTraining Pipeline completed successfully!")
    except Exception as e:
        print(f"Error: {e}")
