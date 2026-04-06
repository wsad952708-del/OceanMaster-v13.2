"""Generate fresh hotspot predictions for all 4 species."""
import asyncio
import sys
import json
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, str(Path(__file__).parent))

async def main():
    from main_v10_3 import OceanMasterPipeline
    
    print("🚀 Starting OceanMaster full pipeline for all species...")
    pipeline = OceanMasterPipeline(
        lat_range=(15, 28), lon_range=(120, 135),
        vessel_pos=(25.13, 121.74), output_dir="output",
    )
    results = await pipeline.run_full()
    
    hotspots = results.get("hotspots", [])
    print(f"\n✅ Pipeline complete: {len(hotspots)} hotspots generated")
    
    # Check species distribution
    species = {}
    for h in hotspots:
        sp = h.get("species", "unknown")
        species[sp] = species.get(sp, 0) + 1
    
    print(f"Species ({len(species)}):")
    for sp, cnt in sorted(species.items()):
        print(f"  🐟 {sp}: {cnt} hotspots")
    
    # Save as the main hotspot file
    out = Path("output/hotspots.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(hotspots, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n💾 Saved → {out} ({out.stat().st_size / 1024:.1f} KB)")

if __name__ == "__main__":
    asyncio.run(main())
