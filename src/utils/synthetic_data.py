import numpy as np
import rasterio
from rasterio.transform import from_origin
import os
import sys

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import settings

def generate_synthetic_geotiff(output_path, width=256, height=256, health_status='healthy'):
    """
    Generates a synthetic 4-band GeoTIFF image (R, G, B, NIR) mimicking satellite data.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Base reflectance ranges depending on health status
    if health_status == 'healthy':
        nir_range = (0.6, 0.9)
        red_range = (0.05, 0.15)
        green_range = (0.1, 0.2)
        blue_range = (0.02, 0.08)
    elif health_status == 'stressed':
        nir_range = (0.4, 0.6)
        red_range = (0.15, 0.25)
        green_range = (0.1, 0.15)
        blue_range = (0.05, 0.1)
    else: # critical
        nir_range = (0.2, 0.4)
        red_range = (0.25, 0.4)
        green_range = (0.05, 0.1)
        blue_range = (0.05, 0.15)

    # Generate random arrays for each band within the specified ranges
    red_band = np.random.uniform(red_range[0], red_range[1], (height, width)).astype(np.float32)
    green_band = np.random.uniform(green_range[0], green_range[1], (height, width)).astype(np.float32)
    blue_band = np.random.uniform(blue_range[0], blue_range[1], (height, width)).astype(np.float32)
    nir_band = np.random.uniform(nir_range[0], nir_range[1], (height, width)).astype(np.float32)

    # Add some spatial noise/patterns (simulating field variations)
    noise = np.random.normal(0, 0.02, (height, width)).astype(np.float32)
    red_band = np.clip(red_band + noise, 0, 1)
    nir_band = np.clip(nir_band - noise, 0, 1) # inversely correlated noise for NIR

    # Define geospatial metadata
    # Origin coordinates (e.g., somewhere in India) and pixel size (10m)
    transform = from_origin(78.9629, 20.5937, 10.0, 10.0)
    
    # Write the synthetic image
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=4,
        dtype=str(red_band.dtype),
        crs='+proj=latlong',
        transform=transform,
    ) as dest:
        dest.write(red_band, 1)
        dest.write(green_band, 2)
        dest.write(blue_band, 3)
        dest.write(nir_band, 4)
        
        # Add band descriptions
        dest.set_band_description(1, 'Red')
        dest.set_band_description(2, 'Green')
        dest.set_band_description(3, 'Blue')
        dest.set_band_description(4, 'NIR')

    print(f"Generated synthetic GeoTIFF ({health_status}): {output_path}")

def generate_dataset():
    """Generates a small dataset of synthetic images."""
    data_dir = os.path.join(settings.RAW_DATA_DIR, 'satellite_imagery')
    
    statuses = ['healthy', 'stressed', 'critical']
    for i in range(10):
        status = statuses[i % 3]
        filename = f"field_{i+1}_{status}.tif"
        output_path = os.path.join(data_dir, filename)
        generate_synthetic_geotiff(output_path, health_status=status)

if __name__ == "__main__":
    generate_dataset()
