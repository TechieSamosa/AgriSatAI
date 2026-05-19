import os
import sys

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.profiling import time_it
from config import settings

@time_it
def load_to_parquet(df, path):
    """
    Writes the DataFrame to Parquet format using pandas to bypass Hadoop/winutils on Windows.
    """
    print(f"Loading data to Parquet at {path}...")
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pdf = df.toPandas()
    # Write to a single file rather than a directory for pandas
    if path.endswith('/'):
        path = path[:-1] + ".parquet"
    if not path.endswith('.parquet'):
        path = path + ".parquet"
    pdf.to_parquet(path, index=False)
    print("Load complete.")

@time_it
def load_to_csv_local(df, path):
    """
    Writes the DataFrame to local CSV for easy dashboard reading (pandas).
    """
    print(f"Loading data to CSV at {path}...")
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pdf = df.toPandas()
    if not path.endswith('.csv'):
        path = path + ".csv"
    pdf.to_csv(path, index=False)
    print("Load complete.")
