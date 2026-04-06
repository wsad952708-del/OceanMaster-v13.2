"""Verify all bug fixes"""
import sys, numpy as np
sys.stdout.reconfigure(encoding='utf-8')

# 1. ConvLSTMCell callable
from engine.ml.convlstm_predictor import ConvLSTMCell
cell = ConvLSTMCell(input_channels=3, hidden_channels=8)
x = np.random.randn(3, 16, 16).astype(np.float32)
h = np.zeros((8, 16, 16), dtype=np.float32)
c = np.zeros((8, 16, 16), dtype=np.float32)
h1, c1 = cell.forward(x, h, c)
print(f'[PASS] ConvLSTMCell.forward(): h={h1.shape}')
h2, c2 = cell(x, (h, c))
print(f'[PASS] ConvLSTMCell.__call__(): h={h2.shape}')

# 2. SSTChangeDetector with real data
import xarray as xr
ds = xr.open_dataset('C:/tmp/L2_cache/cmems_sst.nc')
sst = ds['thetao'].isel(depth=0)
sst_t0 = sst.isel(time=0).values.astype(np.float32)
sst_t1 = sst.isel(time=-1).values.astype(np.float32)
from engine.sst_change_detector import SSTChangeDetector
scd = SSTChangeDetector()
r = scd.compute(sst_t1, sst_t0, dt_days=2.0)
print(f'[PASS] SSTChangeDetector: upwelling={r["upwelling_pct"]}%')
ds.close()

# 3. SRGAN
from engine.ocean_srgan import OceanSRGAN
srgan = OceanSRGAN()
lats = np.linspace(20, 25, 32)
lons = np.linspace(120, 125, 32)
sst_small = np.random.randn(32, 32).astype(np.float32) * 5 + 25
r = srgan.super_resolve(sst_small, lats, lons)
print(f'[PASS] SRGAN: {sst_small.shape} -> {r["sst_hr"].shape} (mode={r["mode"]})')

print('\nAll 3 verified OK')
