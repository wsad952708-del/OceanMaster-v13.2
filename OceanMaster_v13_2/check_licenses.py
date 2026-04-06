import urllib.request
import json
import re

packages = [
    "numpy", "scipy", "pandas", "httpx", "fastapi", "uvicorn",
    "scikit-learn", "joblib", "lightgbm", "xgboost", "shap",
    "netCDF4", "xarray", "zarr", "matplotlib", "folium",
    "simplekml", "geojson", "requests", "aiohttp", "apscheduler",
    "python-dotenv", "tenacity", "copernicusmarine", "anthropic"
]

results = []

for pkg in packages:
    url = f"https://pypi.org/pypi/{pkg}/json"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            info = data.get("info", {})
            
            # extract version
            version = info.get("version", "unknown")
            
            # extract license field
            lic = info.get("license", "")
            
            # extract classifiers
            classifiers = info.get("classifiers", [])
            license_classifiers = [c.split("::")[-1].strip() for c in classifiers if "License" in c]
            
            results.append({
                "package": pkg,
                "version": version,
                "license_field": lic,
                "classifiers": license_classifiers
            })
    except Exception as e:
        results.append({
            "package": pkg,
            "error": str(e)
        })

print(json.dumps(results, indent=2))
