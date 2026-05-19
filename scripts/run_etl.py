import os
import sys
import subprocess

# 1. Fix paths for PySpark on Windows before any imports!
# PySpark workers communicate over binary stdin/stdout, so a .bat wrapper corrupts the pipe!
# To bypass space-in-path issues, we create a Windows Directory Junction without spaces.
junction_path = r"C:\AgriSatVenv"
if not os.path.exists(junction_path):
    subprocess.run(["cmd.exe", "/c", "mklink", "/J", junction_path, sys.prefix], capture_output=True)

os.environ['PYSPARK_PYTHON'] = os.path.join(junction_path, 'Scripts', 'python.exe')
os.environ['PYSPARK_DRIVER_PYTHON'] = os.path.join(junction_path, 'Scripts', 'python.exe')
os.environ['SPARK_HOME'] = os.path.join(sys.prefix, 'Lib', 'site-packages', 'pyspark')
os.environ['SPARK_LOCAL_IP'] = '127.0.0.1'

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from src.etl.extract import create_spark_session, extract_geotiff_metadata_local
from src.etl.transform import clean_data, calculate_ndvi, engineer_features
from src.etl.load import load_to_parquet, load_to_csv_local
from src.utils.profiling import time_it

@time_it
def run_pipeline():
    """Executes the full ETL pipeline."""
    # 1. Initialize Spark
    spark = create_spark_session()
    
    # 2. Extract
    raw_img_dir = os.path.join(settings.RAW_DATA_DIR, 'satellite_imagery')
    
    print("--- STARTING EXTRACTION ---")
    df = extract_geotiff_metadata_local(spark, raw_img_dir)
    
    if df.count() == 0:
        print("No data found to process. Please run scripts/generate_data.py first.")
        spark.stop()
        return
    
    # 3. Transform
    print("--- STARTING TRANSFORMATION ---")
    # For MVP, we mock some weather/soil data as if joined from a Kaggle CSV
    from pyspark.sql.functions import rand
    
    # Simulate joining weather data
    df = df.withColumn("temperature", 20.0 + (rand() * 15.0)) # 20-35 C
    df = df.withColumn("soil_moisture", 0.1 + (rand() * 0.4)) # 10-50%
    # Simulate yield
    # Higher NDVI generally means higher yield (mock relation)
    # We will compute NDVI first
    
    df = clean_data(df)
    df = calculate_ndvi(df)
    
    # Simulate yield based on NDVI
    df = df.withColumn("yield_tons_ha", df.NDVI * 10.0 + (rand() * 2.0))
    
    df = engineer_features(df)
    
    # 4. Load
    print("--- STARTING LOAD ---")
    output_dir = os.path.join(settings.BASE_DIR, 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    parquet_path = os.path.join(output_dir, 'processed_features.parquet')
    csv_path = os.path.join(output_dir, 'processed_features_csv')
    
    load_to_parquet(df, parquet_path)
    load_to_csv_local(df, csv_path)
    
    print("ETL Pipeline completed successfully!")
    spark.stop()

if __name__ == "__main__":
    run_pipeline()
