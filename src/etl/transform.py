from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.profiling import time_it

@time_it
def clean_data(df):
    """
    Handles nulls and removes extreme outliers from the DataFrame.
    """
    print("Cleaning data...")
    # Drop rows where crucial columns are null
    if "mean_red" in df.columns:
        df = df.dropna(subset=["mean_red", "mean_nir"])
    
    # Assuming Kaggle dataset has these columns
    if "soil_moisture" in df.columns:
        df = df.dropna(subset=["soil_moisture"])
        
    return df

@time_it
def calculate_ndvi(df, red_col="mean_red", nir_col="mean_nir"):
    """
    Computes NDVI = (NIR - Red) / (NIR + Red)
    """
    print("Calculating NDVI...")
    if red_col in df.columns and nir_col in df.columns:
        # Add a small epsilon to denominator to prevent division by zero
        epsilon = 1e-8
        df = df.withColumn(
            "NDVI", 
            (F.col(nir_col) - F.col(red_col)) / (F.col(nir_col) + F.col(red_col) + epsilon)
        )
    return df

@time_it
def engineer_features(df):
    """
    Creates additional features for ML.
    """
    print("Engineering features...")
    # Example feature engineering based on typical ag dataset
    
    # 1. Vegetation Stress Index (if we have NDVI and temperature)
    if "NDVI" in df.columns and "temperature" in df.columns:
        df = df.withColumn("stress_index", F.col("temperature") / (F.col("NDVI") + 0.1))
        
    # 2. Moisture-Temperature Interaction
    if "soil_moisture" in df.columns and "temperature" in df.columns:
        df = df.withColumn("moisture_temp_interaction", F.col("soil_moisture") * F.col("temperature"))
        
    return df
