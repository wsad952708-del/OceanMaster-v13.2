"""
Standalone Copernicus Marine Data Fetcher
This module isolates all `copernicusmarine` library usage to avoid GPL/EUPL licensing contagion
to the core OceanMaster engine.

Data is downloaded as NetCDF files to `C:/tmp/L2_cache/` or a specified directory,
which the core engine will then read without directly importing this library.
"""

import os
import logging
from datetime import datetime, timezone, timedelta

try:
    import copernicusmarine as cm
    _HAS_CMEMS = True
except ImportError:
    _HAS_CMEMS = False

log = logging.getLogger("OceanMaster.ExternalFetcher")

def download_cmems_sst(lat_min, lat_max, lon_min, lon_max, output_dir="C:/tmp/L2_cache"):
    if not _HAS_CMEMS:
        log.warning("copernicusmarine is not installed.")
        return False
        
    user = os.environ.get("CMEMS_USER", "")
    pwd = os.environ.get("CMEMS_PASS", "")
    if not user:
        return False

    now = datetime.now(timezone.utc)
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "cmems_sst.nc")
    
    try:
        cm.subset(
            dataset_id="cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m",
            variables=["thetao"],
            minimum_latitude=lat_min, maximum_latitude=lat_max,
            minimum_longitude=lon_min, maximum_longitude=lon_max,
            minimum_depth=0.0, maximum_depth=1.0,
            start_datetime=(now - timedelta(days=3)).isoformat(),
            end_datetime=now.isoformat(),
            username=user, password=pwd,
            output_filename="cmems_sst.nc",
            output_directory=output_dir,
            force_download=True
        )
        return True
    except Exception as e:
        log.error(f"CMEMS SST download failed: {e}")
        return False

def download_cmems_chl(lat_min, lat_max, lon_min, lon_max, days_lag=0, output_dir="C:/tmp/L2_cache"):
    if not _HAS_CMEMS:
        return False
        
    user = os.environ.get("CMEMS_USER", "")
    pwd = os.environ.get("CMEMS_PASS", "")
    if not user:
        return False

    now = datetime.now(timezone.utc) - timedelta(days=days_lag)
    os.makedirs(output_dir, exist_ok=True)
    out_name = f"cmems_chl_lag{days_lag}.nc" if days_lag > 0 else "cmems_chl.nc"
    
    try:
        cm.subset(
            dataset_id="cmems_mod_glo_bgc-pft_anfc_0.25deg_P1D-m",
            variables=["chl"],
            minimum_latitude=lat_min, maximum_latitude=lat_max,
            minimum_longitude=lon_min, maximum_longitude=lon_max,
            minimum_depth=0.0, maximum_depth=10.0,
            start_datetime=(now - timedelta(days=5)).isoformat(),
            end_datetime=now.isoformat(),
            username=user, password=pwd,
            output_filename=out_name,
            output_directory=output_dir,
            force_download=True
        )
        return True
    except Exception as e:
        log.error(f"CMEMS CHL download failed: {e}")
        return False

def download_cmems_currents(lat_min, lat_max, lon_min, lon_max, output_dir="C:/tmp/L2_cache"):
    if not _HAS_CMEMS:
        return False
        
    user = os.environ.get("CMEMS_USER", "")
    pwd = os.environ.get("CMEMS_PASS", "")
    if not user:
        return False

    now = datetime.now(timezone.utc)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        cm.subset(
            dataset_id="cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m",
            variables=["uo", "vo"],
            minimum_latitude=lat_min, maximum_latitude=lat_max,
            minimum_longitude=lon_min, maximum_longitude=lon_max,
            minimum_depth=0.0, maximum_depth=5.0,
            start_datetime=(now - timedelta(days=3)).isoformat(),
            end_datetime=now.isoformat(),
            username=user, password=pwd,
            output_filename="cmems_cur.nc",
            output_directory=output_dir,
            force_download=True
        )
        return True
    except Exception as e:
        log.error(f"CMEMS currents download failed: {e}")
        return False

def download_cmems_ssh(lat_min, lat_max, lon_min, lon_max, output_dir="C:/tmp/L2_cache"):
    if not _HAS_CMEMS:
        return False
        
    user = os.environ.get("CMEMS_USER", "")
    pwd = os.environ.get("CMEMS_PASS", "")
    if not user:
        return False

    now = datetime.now(timezone.utc)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        cm.subset(
            dataset_id="cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
            variables=["zos"],
            minimum_latitude=lat_min, maximum_latitude=lat_max,
            minimum_longitude=lon_min, maximum_longitude=lon_max,
            start_datetime=(now - timedelta(days=3)).isoformat(),
            end_datetime=now.isoformat(),
            username=user, password=pwd,
            output_filename="cmems_ssh.nc",
            output_directory=output_dir,
            force_download=True
        )
        return True
    except Exception as e:
        log.error(f"CMEMS SSH download failed: {e}")
        return False

def download_cmems_glorys12(lat_min, lat_max, lon_min, lon_max, output_dir="C:/tmp/L2_cache"):
    if not _HAS_CMEMS:
        return False
        
    user = os.environ.get("CMEMS_USER", "")
    pwd = os.environ.get("CMEMS_PASS", "")
    if not user:
        return False

    now = datetime.now(timezone.utc)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        cm.subset(
            dataset_id="cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m",
            variables=["thetao"],
            minimum_latitude=lat_min, maximum_latitude=lat_max,
            minimum_longitude=lon_min, maximum_longitude=lon_max,
            minimum_depth=0.0, maximum_depth=1000.0,
            start_datetime=(now - timedelta(days=5)).isoformat(),
            end_datetime=now.isoformat(),
            username=user, password=pwd,
            output_filename="cmems_glorys12.nc",
            output_directory=output_dir,
            force_download=True
        )
        return True
    except Exception as e:
        log.error(f"CMEMS GLORYS12 download failed: {e}")
        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lat-min", type=float, required=True)
    parser.add_argument("--lat-max", type=float, required=True)
    parser.add_argument("--lon-min", type=float, required=True)
    parser.add_argument("--lon-max", type=float, required=True)
    parser.add_argument("--out-dir", default="C:/tmp/L2_cache")
    parser.add_argument("--target", choices=["sst", "chl", "cur", "ssh", "glorys12", "all"], default="all")
    args = parser.parse_args()
    
    if args.target in ["sst", "all"]: download_cmems_sst(args.lat_min, args.lat_max, args.lon_min, args.lon_max, args.out_dir)
    if args.target in ["chl", "all"]: download_cmems_chl(args.lat_min, args.lat_max, args.lon_min, args.lon_max, 0, args.out_dir)
    if args.target in ["cur", "all"]: download_cmems_currents(args.lat_min, args.lat_max, args.lon_min, args.lon_max, args.out_dir)
    if args.target in ["ssh", "all"]: download_cmems_ssh(args.lat_min, args.lat_max, args.lon_min, args.lon_max, args.out_dir)
    if args.target in ["glorys12", "all"]: download_cmems_glorys12(args.lat_min, args.lat_max, args.lon_min, args.lon_max, args.out_dir)
