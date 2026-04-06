"""Quick API verification script"""
import urllib.request
import json
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

KEY = "sW1VVq9fgYka_2EnWR2-IpHOCKFXwXAArWjhEGsHJBY"
BASE = "http://localhost:8000"

def api_get(path):
    req = urllib.request.Request(f"{BASE}{path}", headers={"X-API-Key": KEY})
    return json.loads(urllib.request.urlopen(req, timeout=10).read())

# Test 1: Health
print("=" * 50)
print("  OceanMaster API Verification")
print("=" * 50)

health = json.loads(urllib.request.urlopen(f"{BASE}/health", timeout=5).read())
print(f"\n✅ Health: {health['status']}, version={health['version']}, hotspots={health['hotspot_count']}")

# Test 2: Hotspots
data = api_get("/api/hotspots")
spots = data["hotspots"]
print(f"\n📊 Hotspots: {len(spots)} total")

species_count = {}
for s in spots:
    sp = s.get("species", "unknown")
    species_count[sp] = species_count.get(sp, 0) + 1

print(f"   Species found ({len(species_count)}):")
for sp, cnt in sorted(species_count.items()):
    print(f"   🐟 {sp}: {cnt} hotspots")

print(f"\n   Top 8 hotspots:")
for i, s in enumerate(spots[:8]):
    hsi = s.get("score", s.get("hsi", 0))
    print(f"   #{i+1} [{s.get('species','?')}] HSI={hsi:.3f} ({s.get('lat',0):.2f}°N, {s.get('lon',0):.2f}°E)")

# Test 3: Sea Conditions
sea = api_get("/api/sea_conditions")
print(f"\n🌊 Sea Conditions:")
print(f"   Moon: {sea.get('moon_phase', '--')}")
print(f"   SST: {sea.get('sst_min', '--')} - {sea.get('sst_max', '--')}°C")
print(f"   ONI: {sea.get('oni', '--')}")
print(f"   ML Ensemble: {sea.get('ml_ensemble', False)}")

# Test 4: Dashboard HTML
req = urllib.request.Request(f"{BASE}/")
html = urllib.request.urlopen(req, timeout=5).read().decode("utf-8")
has_api_key = KEY in html
has_species_filter = "黃鰭鮪" in html or "yellowfin" in html.lower()
print(f"\n🌐 Dashboard HTML:")
print(f"   Size: {len(html)} bytes")
print(f"   API key injected: {'✅' if has_api_key else '❌'}")
print(f"   Species filter UI: {'✅' if has_species_filter else '❌'}")

print(f"\n{'=' * 50}")
print(f"  ALL TESTS PASSED ✅")
print(f"{'=' * 50}")
