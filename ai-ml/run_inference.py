import os
import json
import uuid
import rasterio
import numpy as np

# Our custom modules
from prism import compute_surface_normals, get_sun_vector, apply_lunar_lambert
from match_loftr import LoFTRMatcher
from modality_bridge import bridge_iirs_to_grayscale
from backend_client_stub import get_dem, get_products_for_site, get_spice_kernels, register_run_in_db

def run_pipeline(site_id):
    run_id = f"run_{uuid.uuid4().hex[:8]}"
    print(f"\n🚀 STARTING PIPELINE FOR SITE: {site_id} | RUN ID: {run_id}")
    
    # 1. Setup Backend Folder Structure
    base_dir = f"../data/sites/{site_id}"
    json_dir = os.path.join(base_dir, "matched_points")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # 2. Fetch Data via Backend Client
    print("Fetching data (Mock)...")
    products = get_products_for_site(site_id)
    dem, pixel_scale = get_dem(site_id)
    spice_data = get_spice_kernels(site_id)

    # 3. Modality Bridge & Reference Load
    if products.get("type") == "iirs_wac":
        source_image = bridge_iirs_to_grayscale(products.get("source_cube_path"))
    else:
        # Fallback/Mock payload handling
        source_image = np.ones((512, 512), dtype=np.uint8) * 150
        
    # PRODUCTION UPGRADE: Ingest reference via rasterio to retain spatial profile
    ref_path = products.get("reference_image_path", "mock_ref.tif")
    try:
        with rasterio.open(ref_path) as src:
            ref_image = src.read(1)
            profile = src.profile
    except (rasterio.errors.RasterioIOError, FileNotFoundError):
        print("⚠️ Mock mode: creating dummy profile for reference image.")
        ref_image = np.ones((512, 512), dtype=np.float32)
        profile = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': None, 'width': 512, 'height': 512, 'count': 1, 'crs': None, 'transform': rasterio.transform.from_origin(0, 0, 1, 1)}

    # 4. PRISM Relighting
    print("Running PRISM Relighting...")
    normals = compute_surface_normals(dem, pixel_scale)
    sun_vector = get_sun_vector(spice_data["sun_azimuth"], spice_data["sun_elevation"])
    relit_reference = apply_lunar_lambert(ref_image, normals, sun_vector)

    # 5. Matching
    print("Running LoFTR...")
    matcher = LoFTRMatcher()
    mkpts0, mkpts1, conf = matcher.match(source_image, relit_reference)
    
    # 6. Format Handoff JSON (Strict schema alignment)
    matches_list = [
        {
            "src_x": float(mkpts0[i][0]),
            "src_y": float(mkpts0[i][1]),
            "ref_x": float(mkpts1[i][0]),
            "ref_y": float(mkpts1[i][1]),
            "confidence": float(conf[i])
        }
        for i in range(len(mkpts0))
    ]
        
    handoff_payload = {
        "site_id": site_id,
        "source_image": os.path.basename(products.get("source_cube_path", "source_mock.cub")),
        "reference_image": os.path.basename(ref_path),
        "matches": matches_list
    }

    # 7. Write Files & Register in DB
    json_path = os.path.join(json_dir, f"{run_id}.json")
    relit_path = os.path.join(results_dir, f"{run_id}_relit_reference.tif")
    
    with open(json_path, "w") as f:
        json.dump(handoff_payload, f, indent=2)
        
    # PRODUCTION UPGRADE: Geospatial Metadata-Preserving Write
    profile.update(dtype=relit_reference.dtype, compress='lzw')
    with rasterio.open(relit_path, 'w', **profile) as dst:
        dst.write(relit_reference, 1)
    
    print("\n✅ HANDOFF COMPLETE!")
    register_run_in_db(run_id, site_id, json_path, relit_path)

if __name__ == "__main__":
    run_pipeline("test_ohrc_site")