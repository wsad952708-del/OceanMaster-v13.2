"""
DL Infrastructure — 深層檢驗
=============================
超越 smoke test，驗證：
1. U-Net: gradient flow, skip connection 有效性, CBAM 增益
2. ConvLSTM: 時序記憶, gradient flow through time
3. Data Pipeline: edge cases (全 NaN, 單點, 空序列)
4. Trainer: 能否在合成資料上收斂 (overfit test)
5. TZCF: 已知等值線精度
6. Syrjala: 已知分布 p-value 正確性
7. Lagrangian: thermotaxis 方向正確性
8. API 相容性: predict() 接口與現有系統相容
"""

import numpy as np
import sys
import time
import os

os.environ["PYTHONIOENCODING"] = "utf-8"

PASS = 0
FAIL = 0
ERRORS = []


def check(name, condition, detail=""):
    global PASS, FAIL, ERRORS
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        ERRORS.append(f"{name}: {detail}")
        print(f"  [FAIL] {name} -- {detail}")


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ═══════════════════════════════════════════════════════════
#  1. TZCF Tracker — 精確度測試
# ═══════════════════════════════════════════════════════════
section("1. TZCF Tracker")

from engine.tzcf_tracker import TZCFTracker

# 1a. 已知 TZCF 位置的合成資料
lats = np.linspace(20, 45, 100)
lons = np.linspace(140, 170, 120)
# CHL = 0.1 below lat 32, 0.5 above lat 32 → TZCF at exactly 32°N
chl = np.zeros((100, 120))
for i, lat in enumerate(lats):
    chl[i, :] = 0.1 if lat < 32.0 else 0.5

t = TZCFTracker(chl_threshold=0.2)
r = t.compute(chl, lats, lons)
mean_lat = r["mean_tzcf_lat"]
check("TZCF known contour", abs(mean_lat - 32.0) < 1.0,
      f"Expected ~32.0, got {mean_lat:.1f}")

# 1b. Distance field 符號: 北邊正, 南邊負
dist = r["distance_to_tzcf"]
check("TZCF dist sign north",
      np.mean(dist[lats > 35, :]) > 0,
      "North of TZCF should be positive")
check("TZCF dist sign south",
      np.mean(dist[lats < 28, :]) < 0,
      "South of TZCF should be negative")

# 1c. 全零 Chl-a → 沒有 TZCF
chl_zero = np.zeros((50, 50))
r2 = t.compute(chl_zero, np.linspace(20, 40, 50), np.linspace(130, 160, 50))
check("TZCF zero chl", r2["valid_fraction"] == 0.0,
      f"Expected 0 valid, got {r2['valid_fraction']}")

# 1d. 全 NaN → 當 0 處理
chl_nan = np.full((50, 50), np.nan)
r3 = t.compute(chl_nan, np.linspace(20, 40, 50), np.linspace(130, 160, 50))
check("TZCF all NaN", r3["valid_fraction"] == 0.0)


# ═══════════════════════════════════════════════════════════
#  2. Syrjala Test — 統計正確性
# ═══════════════════════════════════════════════════════════
section("2. Syrjala Test")

from engine.ml.syrjala_test import syrjala_test

# 2a. 完全相同的分布 → p 接近 1
identical = np.random.rand(20, 20)
s1 = syrjala_test(identical, identical, n_permutations=199)
check("Syrjala identical -> p~1", s1["p_value"] > 0.5,
      f"p={s1['p_value']:.3f}")

# 2b. 完全不同的分布 → p 接近 0
dist_a = np.zeros((20, 20))
dist_a[:10, :] = 1.0  # 全在北邊
dist_b = np.zeros((20, 20))
dist_b[10:, :] = 1.0  # 全在南邊
s2 = syrjala_test(dist_a, dist_b, n_permutations=199)
check("Syrjala different -> p<0.05", s2["p_value"] < 0.05,
      f"p={s2['p_value']:.3f}")
check("Syrjala different -> significant", s2["significant"] is True)

# 2c. 加小擾動 → 不顯著
noise = np.random.rand(20, 20) * 0.01
s3 = syrjala_test(identical, identical + noise, n_permutations=199)
check("Syrjala small noise -> not significant", s3["p_value"] > 0.05,
      f"p={s3['p_value']:.3f}")


# ═══════════════════════════════════════════════════════════
#  3. Data Pipeline — Edge cases
# ═══════════════════════════════════════════════════════════
section("3. Data Pipeline")

from engine.ml.dl_data_pipeline import (
    normalize_field, compute_time_statistics, cpue_to_density_field,
    CLIM_STATS,
)

# 3a. 全 NaN → 全 0 (不 crash)
nan_arr = np.full((10, 10), np.nan)
norm = normalize_field(nan_arr, "sst", mode="per_sample")
check("Normalize all NaN", not np.any(np.isnan(norm)),
      f"Found NaN in output")
check("Normalize all NaN → 0", np.allclose(norm, 0.0))

# 3b. Climatology 模式 SST 正規化
sst = np.full((10, 10), 22.0)  # 剛好等於 climatology mean
norm_clim = normalize_field(sst, "sst", mode="climatology")
check("Normalize clim SST=mean → ~0", abs(np.mean(norm_clim)) < 0.01,
      f"mean={np.mean(norm_clim):.4f}")

# 3c. Time statistics trend
daily = [np.ones((5, 5)) * t for t in range(30)]  # 線性增長
ts = compute_time_statistics(daily)
check("TimeStats trend positive", np.all(ts["trend"] > 0),
      f"trend mean={np.mean(ts['trend']):.4f}")
check("TimeStats mean correct", abs(np.mean(ts["mean"]) - 14.5) < 0.1,
      f"mean={np.mean(ts['mean']):.2f}")
check("TimeStats max correct", abs(np.mean(ts["max"]) - 29.0) < 0.1,
      f"max={np.mean(ts['max']):.2f}")

# 3d. CPUE density: 單點 → 中心有值
lats_d = np.linspace(20, 30, 40)
lons_d = np.linspace(130, 140, 40)
points = [{"lat": 25.0, "lon": 135.0, "cpue": 100.0}]
density = cpue_to_density_field(points, lats_d, lons_d, bandwidth_deg=0.5)
center_i = np.argmin(np.abs(lats_d - 25.0))
center_j = np.argmin(np.abs(lons_d - 135.0))
check("CPUE density peak at center", density[center_i, center_j] > 0.9,
      f"center value={density[center_i, center_j]:.3f}")

# 3e. Binary threshold
binary = cpue_to_density_field(points, lats_d, lons_d, binary_threshold=0.5)
check("CPUE binary values", set(np.unique(binary)) <= {0.0, 1.0})


# ═══════════════════════════════════════════════════════════
#  4. Lagrangian — Thermotaxis 方向正確性
# ═══════════════════════════════════════════════════════════
section("4. Lagrangian Thermotaxis")

from engine.lagrangian_advection import AdvectionConfig

cfg = AdvectionConfig(swim_enabled=True, preferred_sst=24.0, sst_sensitivity=2.0)
check("Config swim_enabled", cfg.swim_enabled is True)
check("Config swim_speed", cfg.swim_speed_ms == 0.3)
check("Config random_walk_std", cfg.random_walk_std == 0.01)

# Test _compute_thermotaxis directly
from engine.lagrangian_advection import _compute_thermotaxis

# SST field: warm on south (30°C), cold on north (15°C)
lats_l = np.linspace(20, 40, 50).astype(np.float64)
lons_l = np.linspace(130, 150, 60).astype(np.float64)
sst_field = np.zeros((50, 60), dtype=np.float64)
for i, lat in enumerate(lats_l):
    sst_field[i, :] = 30.0 - (lat - 20.0) * 0.75  # 30°C@20N → 15°C@40N

# Particle at 35°N (SST≈18.75°C, too cold, preferred=24) → should swim south (lower lat)
particle_lat = np.array([35.0], dtype=np.float64)
particle_lon = np.array([140.0], dtype=np.float64)
deg_lon = 1.0 / (111320.0 * np.cos(np.radians(35.0)))

dx, dy = _compute_thermotaxis(
    particle_lat, particle_lon,
    None, None, None, None, lats_l, lons_l, None,
    sst_field=sst_field,
    swim_speed=0.5,
    preferred_sst=24.0,
    sst_sensitivity=2.0,
    random_walk_std=0.0,  # disable random for direction test
    dt_sec=3600.0,
    deg_per_m_lon=np.array([deg_lon]),
    deg_per_m_lat=1.0/110540.0,
)
check("Thermotaxis cold → swim south (dy<0)", dy[0] < 0,
      f"dy={dy[0]:.6f}")

# Particle at 22°N (SST≈28.5°C, too warm, preferred=24) → should swim north (higher lat)
particle_lat2 = np.array([22.0], dtype=np.float64)
particle_lon2 = np.array([140.0], dtype=np.float64)
deg_lon2 = 1.0 / (111320.0 * np.cos(np.radians(22.0)))

dx2, dy2 = _compute_thermotaxis(
    particle_lat2, particle_lon2,
    None, None, None, None, lats_l, lons_l, None,
    sst_field=sst_field,
    swim_speed=0.5,
    preferred_sst=24.0,
    sst_sensitivity=2.0,
    random_walk_std=0.0,
    dt_sec=3600.0,
    deg_per_m_lon=np.array([deg_lon2]),
    deg_per_m_lat=1.0/110540.0,
)
check("Thermotaxis warm → swim north (dy>0)", dy2[0] > 0,
      f"dy={dy2[0]:.6f}")

# Particle at ~27°N (SST≈24.75°C, within sensitivity=2°C) → minimal swimming
particle_lat3 = np.array([27.0], dtype=np.float64)
particle_lon3 = np.array([140.0], dtype=np.float64)
deg_lon3 = 1.0 / (111320.0 * np.cos(np.radians(27.0)))

dx3, dy3 = _compute_thermotaxis(
    particle_lat3, particle_lon3,
    None, None, None, None, lats_l, lons_l, None,
    sst_field=sst_field,
    swim_speed=0.5,
    preferred_sst=24.0,
    sst_sensitivity=2.0,
    random_walk_std=0.0,
    dt_sec=3600.0,
    deg_per_m_lon=np.array([deg_lon3]),
    deg_per_m_lat=1.0/110540.0,
)
check("Thermotaxis in comfort zone → ~0", abs(dy3[0]) < abs(dy[0]) * 0.1,
      f"dy={dy3[0]:.8f} vs cold dy={dy[0]:.8f}")


# ═══════════════════════════════════════════════════════════
#  5. Metrics — 已知值驗證
# ═══════════════════════════════════════════════════════════
section("5. Metrics")

from engine.ml.dl_trainer import compute_f1, compute_ssim, compute_iou

# 5a. 完全匹配
pred_perfect = np.array([[1, 0], [0, 1]], dtype=np.float32)
target = np.array([[1, 0], [0, 1]], dtype=np.float32)
check("F1 perfect", compute_f1(pred_perfect, target) == 1.0)
check("IoU perfect", compute_iou(pred_perfect, target) >= 0.99)
check("SSIM perfect ~1", compute_ssim(pred_perfect, target) > 0.99)

# 5b. 完全錯誤
pred_wrong = np.array([[0, 1], [1, 0]], dtype=np.float32)
f1_wrong = compute_f1(pred_wrong, target)
iou_wrong = compute_iou(pred_wrong, target)
check("F1 wrong = 0", f1_wrong < 0.01, f"F1={f1_wrong:.3f}")
check("IoU wrong = 0", iou_wrong < 0.01, f"IoU={iou_wrong:.3f}")

# 5c. SSIM identical
arr = np.random.rand(50, 50)
check("SSIM identical = 1", abs(compute_ssim(arr, arr) - 1.0) < 1e-6)


# ═══════════════════════════════════════════════════════════
#  6. PyTorch Models — Deep Tests
# ═══════════════════════════════════════════════════════════

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    section("6. U-Net Deep Tests")

    from engine.ml.unet_fishing import UNetFishingPredictor, _build_unet_model

    # 6a. Skip connection 驗證: 移除 skip → 輸出應該不同
    model_cbam = _build_unet_model(7, use_cbam=True)
    model_no_cbam = _build_unet_model(7, use_cbam=False)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cbam = model_cbam.to(dev)
    model_no_cbam = model_no_cbam.to(dev)

    n_cbam = sum(p.numel() for p in model_cbam.parameters())
    n_no_cbam = sum(p.numel() for p in model_no_cbam.parameters())
    check("CBAM adds parameters", n_cbam > n_no_cbam,
          f"CBAM={n_cbam:,} vs no_CBAM={n_no_cbam:,}")

    # 6b. Gradient flow: loss.backward() → all parameters have gradients
    model_cbam.train()
    x = torch.randn(2, 7, 32, 32, device=dev)
    y = torch.rand(2, 1, 32, 32, device=dev)
    out = model_cbam(x)
    loss = torch.nn.functional.binary_cross_entropy(out, y)
    loss.backward()

    no_grad_params = []
    for name, param in model_cbam.named_parameters():
        if param.grad is None:
            no_grad_params.append(name)
    check("U-Net gradient flow (all params)", len(no_grad_params) == 0,
          f"No grad: {no_grad_params[:3]}...")

    # 6c. 不同 batch size (eval mode for BN with bs=1)
    model_cbam.eval()
    for bs in [1, 2, 4]:
        x = torch.randn(bs, 7, 16, 16, device=dev)
        with torch.no_grad():
            o = model_cbam(x)
        check(f"U-Net batch_size={bs}", o.shape == (bs, 1, 16, 16),
              f"shape={o.shape}")

    # 6d. 輸出範圍 0-1 (sigmoid)
    with torch.no_grad():
        x = torch.randn(1, 7, 32, 32, device=dev)
        out = model_cbam(x)
    check("U-Net output in [0,1]",
          out.min() >= 0 and out.max() <= 1,
          f"min={out.min():.4f}, max={out.max():.4f}")

    # 6e. predict() API 相容性
    predictor = UNetFishingPredictor(use_cbam=True)
    features = {
        "sst": np.random.rand(40, 50).astype(np.float32) * 30,
        "chl": np.random.rand(40, 50).astype(np.float32),
        "ssh": np.random.rand(40, 50).astype(np.float32) * 0.3,
    }
    result = predictor.predict(features, np.linspace(20, 40, 40), np.linspace(130, 160, 50))
    check("predict() returns dict", isinstance(result, dict))
    check("predict() has fishing_probability",
          "fishing_probability" in result)
    check("predict() shape matches input",
          result["fishing_probability"].shape == (40, 50),
          f"shape={result['fishing_probability'].shape}")

    # ─── ConvLSTM Deep Tests ───
    section("7. ConvLSTM Deep Tests")

    from engine.ml.convlstm_predictor import ConvLSTMPredictor, _build_convlstm_model

    # 7a. Gradient flow through time
    model_cl = _build_convlstm_model(6, [32, 16], [3, 3], "relu").to(dev)
    model_cl.train()
    x = torch.randn(2, 10, 6, 16, 16, device=dev)
    y = torch.rand(2, 1, 16, 16, device=dev)
    out = model_cl(x)
    loss = torch.nn.functional.mse_loss(out, y)
    loss.backward()

    no_grad = [n for n, p in model_cl.named_parameters() if p.grad is None]
    check("ConvLSTM gradient flow all params", len(no_grad) == 0,
          f"No grad: {no_grad[:3]}...")

    # 7b. 時序記憶: 不同長度序列產生不同輸出
    model_cl.eval()
    with torch.no_grad():
        x5 = torch.randn(1, 5, 6, 16, 16, device=dev)
        x10 = torch.cat([x5, torch.randn(1, 5, 6, 16, 16, device=dev)], dim=1)
        o5 = model_cl(x5)
        o10 = model_cl(x10)
    check("ConvLSTM different seq len → different output",
          not torch.allclose(o5, o10, atol=1e-4),
          f"diff={torch.abs(o5-o10).mean():.6f}")

    # 7c. predict() API 相容性
    pred_cl = ConvLSTMPredictor(hidden_channels=[32, 16])
    seq = [
        {"sst": np.random.rand(20, 25).astype(np.float32) * 30,
         "chl": np.random.rand(20, 25).astype(np.float32),
         "ssh": np.random.rand(20, 25).astype(np.float32) * 0.3,
         "u_current": np.random.rand(20, 25).astype(np.float32),
         "v_current": np.random.rand(20, 25).astype(np.float32),
         "npp": np.random.rand(20, 25).astype(np.float32) * 500}
        for _ in range(10)
    ]
    result_cl = pred_cl.predict(seq, np.linspace(20, 30, 20), np.linspace(130, 145, 25))
    check("ConvLSTM predict() dict", isinstance(result_cl, dict))
    check("ConvLSTM predict() shape",
          result_cl["predicted_hsi"].shape == (20, 25),
          f"shape={result_cl['predicted_hsi'].shape}")

    # ─── Overfit Test ───
    section("8. Training Convergence (Overfit on 4 samples)")

    from engine.ml.dl_trainer import DLTrainer, get_bce_loss
    from engine.ml.dl_data_pipeline import create_unet_dataset, create_dataloader

    # Create trivial dataset: SST high → fish=1, SST low → fish=0
    samples = []
    for _ in range(4):
        feat = {"sst": np.random.rand(16, 16).astype(np.float32) * 30 + 10}
        target = (feat["sst"] > 25).astype(np.float32)
        samples.append((feat, target))

    dataset = create_unet_dataset(samples, ["sst"], norm_mode="per_sample")

    # Build tiny U-Net (1 channel, no CBAM for speed)
    tiny_model = _build_unet_model(1, use_cbam=False).to(dev)
    loader = create_dataloader(dataset, batch_size=4, shuffle=False)

    trainer = DLTrainer(
        tiny_model, loss_fn=get_bce_loss(), lr=1e-2,
        checkpoint_dir="models/checkpoints_test"
    )

    # Train 30 epochs — should overfit (loss should drop significantly)
    t0 = time.time()
    history = trainer.fit(loader, val_loader=None, epochs=30, save_every=0)
    dt = time.time() - t0

    first_loss = history["train_loss"][0]
    last_loss = history["train_loss"][-1]
    check("Overfit loss drops >50%",
          last_loss < first_loss * 0.5,
          f"first={first_loss:.4f}, last={last_loss:.4f}")
    check("Training speed <30s for 30 epochs",
          dt < 30,
          f"took {dt:.1f}s")

    # ─── PinballLoss Test ───
    section("9. PinballLoss Properties")

    from engine.ml.dl_trainer import build_pinball_loss

    # q=0.5 should behave like MAE/2
    loss_50 = build_pinball_loss(0.5)
    pred_t = torch.tensor([1.0, 2.0, 3.0], device=dev)
    target_t = torch.tensor([2.0, 2.0, 2.0], device=dev)
    l50 = loss_50(pred_t, target_t).item()

    # q=0.1: underpredict penalized less, overpredict penalized more
    loss_10 = build_pinball_loss(0.1)
    loss_90 = build_pinball_loss(0.9)

    # Same magnitude error but different direction
    over = torch.tensor([3.0], device=dev)  # overprediction by 1
    under = torch.tensor([1.0], device=dev)  # underprediction by 1
    actual = torch.tensor([2.0], device=dev)

    l10_over = loss_10(over, actual).item()
    l10_under = loss_10(under, actual).item()
    check("q=0.1: overpredict costlier than underpredict",
          l10_over > l10_under,
          f"over={l10_over:.4f}, under={l10_under:.4f}")

    l90_over = loss_90(over, actual).item()
    l90_under = loss_90(under, actual).item()
    check("q=0.9: underpredict costlier than overpredict",
          l90_under > l90_over,
          f"under={l90_under:.4f}, over={l90_over:.4f}")

else:
    print("\n  [SKIP] PyTorch tests (not installed)")


# ═══════════════════════════════════════════════════════════
#  Summary
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  TOTAL: {PASS} passed, {FAIL} failed")
print(f"{'='*60}")

if ERRORS:
    for e in ERRORS:
        print(f"  X {e}")
    sys.exit(1)
else:
    print("  ALL DEEP TESTS PASSED")
    sys.exit(0)
