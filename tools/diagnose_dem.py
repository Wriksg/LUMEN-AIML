import sys
import os
import rasterio
import numpy as np

def diagnose_dem(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing DEM file: {file_path}")

    print(f"=== READING: {file_path} ===\n")

    with rasterio.open(file_path) as src:
        # 1. Geometry & CRS
        crs = src.crs
        is_geographic = crs.is_geographic if crs else "Unknown"
        is_projected = crs.is_projected if crs else "Unknown"
        res = src.res
        
        print("--- GEOMETRY ---")
        print(f"CRS              : {crs}")
        print(f"Is Geographic    : {is_geographic} (Degrees)")
        print(f"Is Projected     : {is_projected} (Meters)")
        print(f"Transform        : \n{src.transform}")
        print(f"Pixel Resolution : {res}")

        # 2. Metadata
        dtype = src.dtypes[0]
        scale = src.scales[0] if src.scales and src.scales[0] is not None else 1.0
        offset = src.offsets[0] if src.offsets and src.offsets[0] is not None else 0.0
        nodata = src.nodata
        tags = src.tags()

        print("\n--- METADATA ---")
        print(f"Dtype            : {dtype}")
        print(f"Scale Factor     : {scale}")
        print(f"Offset           : {offset}")
        print(f"NoData Value     : {nodata}")
        print(f"Tags             : {tags}")

        # 3. Read and process data
        z_raw = src.read(1)
        z = z_raw.astype(np.float64)

        if nodata is not None:
            z[z == nodata] = np.nan

        z = z * scale + offset

        # 4. Elevation Stats
        z_min = np.nanmin(z)
        z_max = np.nanmax(z)
        z_ptp = z_max - z_min
        z_mean = np.nanmean(z)

        print("\n--- ELEVATION STATS (After Scale/Offset) ---")
        print(f"Min Elevation    : {z_min:.4f}")
        print(f"Max Elevation    : {z_max:.4f}")
        print(f"Peak-to-Peak     : {z_ptp:.4f}")
        print(f"Mean Elevation   : {z_mean:.4f}")

        # 5. Slope Stats
        dx, dy = res
        dzdy, dzdx = np.gradient(z, dy, dx)
        slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
        
        p50, p90, p99 = np.nanpercentile(slope, [50, 90, 99])

        print("\n--- SLOPE HISTOGRAM (Using Raw Spacing) ---")
        print(f"50th Percentile  : {p50:.4f} degrees")
        print(f"90th Percentile  : {p90:.4f} degrees")
        print(f"99th Percentile  : {p99:.4f} degrees")

    # 6. Diagnosis
    print("\n=== DIAGNOSIS ===")
    print("REFERENCE FACTS:")
    print("- SLDEM2015 elevations are normally METERS, often int16 at 0.5 m/DN")
    print("- Lunar relief over a few-hundred-pixel crop at ~60 m/px is typically 50-2000 m")
    print("- Real lunar slopes at that baseline: median 2°-8°, 99th percentile under ~25°")
    print("- Moon radius = 1737400.0 m")

    print("\nINTERPRETATION:")
    
    # Horizontal
    if res[0] < 1.0:
        print("[HORIZONTAL] WARNING: Pixel resolution is < 1.0. XY is in DEGREES.")
    elif 10 <= res[0] <= 500:
        print("[HORIZONTAL] PASS: Pixel resolution looks metric (Meters).")
    else:
        print(f"[HORIZONTAL] UNKNOWN: Pixel resolution {res[0]} is unusual.")

    # Vertical
    if z_ptp < 5.0:
        print("[VERTICAL] WARNING: Peak-to-Peak relief is < 5m. Z is likely in KILOMETERS.")
    elif 50 <= z_ptp <= 3000:
        print("[VERTICAL] PASS: Peak-to-Peak relief aligns with expected METERS.")
    else:
        print(f"[VERTICAL] UNKNOWN: Peak-to-Peak relief is {z_ptp:.2f}. Check scaling.")

    # Absolute Radius Check
    if z_mean > 1700000:
        print("[VERTICAL] WARNING: Z values are > 1.7 million. Data represents ABSOLUTE RADIUS from lunar core, not topographic relief.")

    # Slope
    if p50 < 0.1:
        print("[SLOPE] WARNING: Median slope is near zero. Z units are heavily underscaled relative to XY units.")
    elif p50 > 45:
        print("[SLOPE] WARNING: Median slope is extreme. Z units are heavily overscaled relative to XY units.")
    elif 2 <= p50 <= 8:
        print("[SLOPE] PASS: Median slope perfectly aligns with lunar reality. No correction needed.")
    else:
        print(f"[SLOPE] ANOMALY: Median slope {p50:.2f}° is outside the 2°-8° expected range.")

if __name__ == '__main__':
    target_file = sys.argv[1] if len(sys.argv) > 1 else "sldem_crop.tif"
    diagnose_dem(target_file)