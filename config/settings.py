import os
from dotenv import load_dotenv

load_dotenv()

# S3/MinIO Configuration
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")

BUCKET_RAW = "raw"
BUCKET_PROCESSED = "processed"
BUCKET_MODELS = "models"

# Local Data Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
SAMPLE_DATA_DIR = os.path.join(DATA_DIR, "sample")

# Spark Configuration
SPARK_APP_NAME = "AgriSatAI_ETL"
SPARK_MASTER = "local[*]"

def get_spark_config():
    """Returns the dictionary of Spark configurations needed for MinIO / S3."""
    return {
        "spark.hadoop.fs.s3a.endpoint": MINIO_ENDPOINT,
        "spark.hadoop.fs.s3a.access.key": MINIO_ACCESS_KEY,
        "spark.hadoop.fs.s3a.secret.key": MINIO_SECRET_KEY,
        "spark.hadoop.fs.s3a.path.style.access": "true",
        "spark.hadoop.fs.s3a.impl": "org.apache.hadoop.fs.s3a.S3AFileSystem",
        "spark.hadoop.fs.s3a.connection.ssl.enabled": "false",
        # Note: In a real environment, you need the aws-java-sdk and hadoop-aws jars
        # compatible with your Spark version for s3a:// to work.
    }
