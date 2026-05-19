import time
from functools import wraps
import pandas as pd

def time_it(func):
    """Decorator to measure and print the execution time of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"[PROFILER] Function '{func.__name__}' executed in {execution_time:.4f} seconds.")
        
        # Save profiling info to a file to be read by the dashboard
        log_profiling_data(func.__name__, execution_time)
        return result
    return wrapper

def log_profiling_data(func_name, execution_time):
    """Appends profiling data to a local CSV for the dashboard to read."""
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from config import settings
    
    log_file = os.path.join(settings.BASE_DIR, "output", "profiling_log.csv")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Determine type of task
    task_type = 'Pandas (Single-thread)' if 'pandas' in func_name.lower() else 'Spark (Distributed)'
    
    data = {
        'timestamp': [pd.Timestamp.now()],
        'function_name': [func_name],
        'execution_time_sec': [execution_time],
        'task_type': [task_type]
    }
    df = pd.DataFrame(data)
    
    if os.path.exists(log_file):
        df.to_csv(log_file, mode='a', header=False, index=False)
    else:
        df.to_csv(log_file, mode='w', header=True, index=False)
