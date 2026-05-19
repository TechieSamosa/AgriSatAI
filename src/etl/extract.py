from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, StringType
import os
import sys

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import settings
from src.utils.profiling import time_it

def create_spark_session():
    """Initializes and returns a SparkSession with MinIO/S3 configurations."""
    builder = SparkSession.builder \
        .appName(settings.SPARK_APP_NAME) \
        .master(settings.SPARK_MASTER)
    
    # Apply S3/MinIO configurations
    spark_conf = settings.get_spark_config()
    for key, value in spark_conf.items():
        builder = builder.config(key, value)
        
    return builder.getOrCreate()

@time_it
def extract_csv(spark, path):
    """Extracts CSV data from S3/MinIO or local path into a Spark DataFrame."""
    print(f"Extracting CSV from {path}...")
    
    # Since we are using synthetic data and downloading Kaggle locally, 
    # we'll read from local file system for now if it doesn't start with s3a://
    
    df = spark.read.csv(path, header=True, inferSchema=True)
    print(f"Extracted {df.count()} records.")
    return df

@time_it
def extract_geotiff_metadata_local(spark, directory_path):
    """
    Extracts metadata and band statistics from local GeoTIFF files using rasterio,
    then converts to a Spark DataFrame.
    """
    import rasterio
    import numpy as np
    
    print(f"Extracting GeoTIFF data from {directory_path}...")
    
    data = []
    if not os.path.exists(directory_path):
        print(f"Directory {directory_path} not found.")
        # Return empty DF
        schema = StructType([
            StructField("field_id", StringType(), True),
            StructField("mean_red", DoubleType(), True),
            StructField("mean_nir", DoubleType(), True),
            StructField("status_label", StringType(), True)
        ])
        return spark.createDataFrame([], schema)

    for filename in os.listdir(directory_path):
        if filename.endswith(".tif"):
            file_path = os.path.join(directory_path, filename)
            try:
                with rasterio.open(file_path) as src:
                    # Read bands (1: Red, 4: NIR based on our generation script)
                    red_band = src.read(1)
                    nir_band = src.read(4)
                    
                    mean_red = float(np.mean(red_band))
                    mean_nir = float(np.mean(nir_band))
                    
                    # Extract label from filename (e.g. field_1_healthy.tif)
                    parts = filename.split('_')
                    label = parts[2].split('.')[0] if len(parts) >= 3 else "unknown"
                    
                    data.append((filename, mean_red, mean_nir, label))
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    schema = StructType([
        StructField("field_id", StringType(), True),
        StructField("mean_red", DoubleType(), True),
        StructField("mean_nir", DoubleType(), True),
        StructField("status_label", StringType(), True)
    ])
    
    if not data:
        return spark.createDataFrame([], schema)
        
    import tempfile
    import csv
    
    # Bypass PySpark daemon issues on Windows by writing to CSV and reading via JVM
    fd, temp_path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    
    try:
        with open(temp_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["field_id", "mean_red", "mean_nir", "status_label"])
            writer.writerows(data)
            
        df = spark.read.csv(temp_path, header=True, schema=schema)
        # Force an action to materialize, but count doesn't trigger python daemon for pure JVM ops
        print(f"Extracted {df.count()} GeoTIFF records.")
    except Exception as e:
        print(f"Error reading temp CSV: {e}")
        df = spark.createDataFrame([], schema)
        
    return df
