"""Test WaveFetcher cascade"""
import sys, os, asyncio
os.chdir(r'c:\Users\user\Desktop\好像快好了\OceanMaster_v13_2')
sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from engine.wave_fetcher import WaveFetcher

async def main():
    wf = WaveFetcher()
    lats = np.arange(20, 26, 1.0)
    lons = np.arange(125, 136, 1.0)
    r = await wf.fetch_wave_height((20, 25), (125, 135), lats, lons)
    print(f"Source: {r.get('source', '?')}")
    wh = r['wave_height']
    print(f"Wave height: shape={wh.shape}, range={np.nanmin(wh):.1f}~{np.nanmax(wh):.1f}m")
    print("PASS")

asyncio.run(main())
