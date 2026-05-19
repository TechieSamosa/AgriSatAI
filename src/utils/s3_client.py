import boto3
from botocore.client import Config
import os
import sys

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import settings

def get_s3_client():
    """Returns a boto3 client configured for MinIO / S3."""
    return boto3.client(
        's3',
        endpoint_url=settings.MINIO_ENDPOINT,
        aws_access_key_id=settings.MINIO_ACCESS_KEY,
        aws_secret_access_key=settings.MINIO_SECRET_KEY,
        config=Config(signature_version='s3v4'),
        region_name='us-east-1' # Default region for MinIO
    )

def create_buckets_if_not_exist():
    """Creates the necessary buckets if they don't already exist."""
    s3 = get_s3_client()
    existing_buckets = [b['Name'] for b in s3.list_buckets().get('Buckets', [])]
    
    buckets_to_create = [
        settings.BUCKET_RAW,
        settings.BUCKET_PROCESSED,
        settings.BUCKET_MODELS
    ]
    
    for bucket in buckets_to_create:
        if bucket not in existing_buckets:
            print(f"Creating bucket: {bucket}")
            s3.create_bucket(Bucket=bucket)
        else:
            print(f"Bucket {bucket} already exists.")

def upload_file(local_path, bucket, key):
    """Uploads a local file to the specified bucket and key."""
    s3 = get_s3_client()
    try:
        s3.upload_file(local_path, bucket, key)
        print(f"Successfully uploaded {local_path} to s3://{bucket}/{key}")
    except Exception as e:
        print(f"Error uploading {local_path}: {str(e)}")

def download_file(bucket, key, local_path):
    """Downloads a file from the specified bucket and key to a local path."""
    s3 = get_s3_client()
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3.download_file(bucket, key, local_path)
        print(f"Successfully downloaded s3://{bucket}/{key} to {local_path}")
    except Exception as e:
        print(f"Error downloading s3://{bucket}/{key}: {str(e)}")
