import urllib.request
import urllib.parse
import json
import time
import numpy as np
from pathlib import Path

def download_gebco_offline():
    print("Starting commercial offline bathymetry download (GEBCO 2020)")
    lats = np.arange(5, 36, 0.5)
    lons = np.arange(120, 176, 0.5)
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
    
    flat_lats = lat_grid.flatten()
    flat_lons = lon_grid.flatten()
    total_pts = len(flat_lats)
    print(f"Grid: {len(lats)}x{len(lons)} = {total_pts} points.")
    
    bathy_flat = np.zeros(total_pts, dtype=np.float32)
    
    chunk_size = 100
    base_url = "https://api.opentopodata.org/v1/gebco2020?locations="
    
    for i in range(0, total_pts, chunk_size):
        chunk_lats = flat_lats[i:i+chunk_size]
        chunk_lons = flat_lons[i:i+chunk_size]
        
        locs = "|".join([f"{lat},{lon}" for lat, lon in zip(chunk_lats, chunk_lons)])
        url = base_url + urllib.parse.quote(locs)
        
        success = False
        retries = 3
        while not success and retries > 0:
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'scientific-data-client/1.0'})
                with urllib.request.urlopen(req, timeout=15) as response:
                    data = json.loads(response.read().decode())
                    if "results" in data:
                        for j, res in enumerate(data["results"]):
                            bathy_flat[i+j] = res["elevation"]
                        success = True
                    else:
                        print(f"Unexpected response at {i}: {data}")
                        retries -= 1
                        time.sleep(2)
            except Exception as e:
                print(f"Chunk {i} failed: {e}. Retries left: {retries-1}")
                retries -= 1
                time.sleep(3)
                
        if not success:
            print(f"FATAL: Could not download chunk starting at {i}.")
            return
            
        print(f"Progress: {min(i+chunk_size, total_pts)} / {total_pts} ({(min(i+chunk_size, total_pts)/total_pts)*100:.1f}%)")
        time.sleep(1.1)  # Rate limit: 1 call per second
        
    bathy_grid = bathy_flat.reshape(lat_grid.shape)
    
    # Save to offline path
    out_path = Path("data/bathymetry/etopo_15min_wpac.npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, bathy=bathy_grid, lats=lats, lons=lons)
    print(f"\\n✅ Successfully saved genuine GEBCO 2020 bathymetry to {out_path}!")

if __name__ == '__main__':
    download_gebco_offline()
