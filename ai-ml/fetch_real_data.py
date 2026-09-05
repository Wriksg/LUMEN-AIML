import os
import requests
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.env import Env

WAC_URL = "https://planetarymaps.usgs.gov/mosaic/Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif"

TILE_IMG_URL = "https://imbrium.mit.edu/DATA/SLDEM2015/TILES/FLOAT_IMG/SLDEM2015_512_00N_30N_000_045_FLOAT.IMG"
TILE_LBL_URL = "https://imbrium.mit.edu/DATA/SLDEM2015/TILES/FLOAT_IMG/SLDEM2015_512_00N_30N_000_045_FLOAT.LBL"
EXPECTED_IMG_SIZE = 1415577600

# Target: Sinus Medii
TARGET_LON = 2.0
TARGET_LAT = 1.3
TARGET_FOOTPRINT_M = 50000.0
MOON_RADIUS_M = 1737400.0

def download_file(url, dest_path, expected_size=None, chunk_size=8192):
    if os.path.exists(dest_path):
        if expected_size and os.path.getsize(dest_path) != expected_size:
            print(f"[!] {dest_path} size mismatch. Deleting and redownloading...")
            os.remove(dest_path)
        else:
            print(f"{dest_path} already exists and size matches, skipping download.")
            return

    print(f"Downloading {url.split('/')[-1]}...")
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(f"\r  {downloaded/1e6:.1f} / {total/1e6:.1f} MB", end="")
    print("\nDownload complete.")

def fetch_and_crop_wac(out_path):
    print("\nConnecting to USGS Astrogeology WAC...")
    env_kwargs = {'GDAL_HTTP_MAX_RETRY': 5, 'GDAL_HTTP_RETRY_DELAY': 3, 'GDAL_HTTP_TIMEOUT': 120}
    with Env(**env_kwargs):
        with rasterio.open(WAC_URL) as src:
            # WAC CRS is in meters. Convert our degrees to lunar meters.
            tgt_x_m = TARGET_LON * (np.pi / 180.0) * MOON_RADIUS_M
            tgt_y_m = TARGET_LAT * (np.pi / 180.0) * MOON_RADIUS_M
            
            row_off, col_off = src.index(tgt_x_m, tgt_y_m)
            pixel_scale_m = abs(src.transform.a)
            size_px = int(TARGET_FOOTPRINT_M / pixel_scale_m)
            
            window = Window(col_off - size_px//2, row_off - size_px//2, size_px, size_px)
            print(f"WAC Streaming pixel window {window}...")
            data = src.read(1, window=window)
            
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff", "height": window.height, "width": window.width,
                "transform": src.window_transform(window), "compress": "lzw"
            })
            print(f"Saving to {out_path}")
            with rasterio.open(out_path, "w", **out_meta) as dest:
                dest.write(data, 1)

def crop_local_dem(local_lbl_path, out_path):
    print(f"\nCropping DEM from local PDS3 file {local_lbl_path}...")
    with rasterio.open(local_lbl_path) as src:
        
        # FIX: Bypass GDAL's buggy PDS3 bounds parser with exact pixel math
        # Tile is 512 ppd. Top-Left is 0E, 30N.
        col_off = int((TARGET_LON - 0.0) * 512.0)
        row_off = int((30.0 - TARGET_LAT) * 512.0)
        
        pixel_scale_deg = 1.0 / 512.0
        pixel_scale_m = pixel_scale_deg * (2 * np.pi * MOON_RADIUS_M) / 360.0
        
        size_px = int(TARGET_FOOTPRINT_M / pixel_scale_m)
        window = Window(col_off - size_px//2, row_off - size_px//2, size_px, size_px)
        
        print(f"DEM Reading pixel window {window} (Scale: {pixel_scale_m:.2f} m/px)...")
        data = src.read(1, window=window).astype("float32")
        
        scale = src.scales[0] if src.scales and src.scales[0] else 1.0
        offset = src.offsets[0] if src.offsets and src.offsets[0] else 0.0
        elevation_m = data * scale + offset
        
        if src.nodata is not None:
            elevation_m[data == src.nodata] = np.nan
            
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff", "height": window.height, "width": window.width,
            "transform": src.window_transform(window), "compress": "lzw", 
            "dtype": "float32", "nodata": np.nan
        })
        print(f"Saving to {out_path}")
        with rasterio.open(out_path, "w", **out_meta) as dest:
            dest.write(elevation_m, 1)

if __name__ == "__main__":
    os.makedirs("local_data", exist_ok=True)
    img_path = os.path.join("local_data", "SLDEM2015_512_00N_30N_000_045_FLOAT.IMG")
    lbl_path = os.path.join("local_data", "SLDEM2015_512_00N_30N_000_045_FLOAT.LBL")
    
    download_file(TILE_IMG_URL, img_path, expected_size=EXPECTED_IMG_SIZE)
    download_file(TILE_LBL_URL, lbl_path)
    
    fetch_and_crop_wac("sinus_medii_wac.tif")
    crop_local_dem(lbl_path, "sinus_medii_dem.tif")