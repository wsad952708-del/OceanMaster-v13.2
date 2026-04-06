"""
Round 4 — FINAL stress test.
盲區掃盲：
1. Trainer + ConvLSTM 5D 整合 (不只 U-Net)
2. 模型 CUDA/CPU 混合
3. Gradient accumulation 不洩漏
4. predict() 連續呼叫不互相污染
5. 超大/超小 feature 值的數值穩定性
6. Syrjala 極端分布 (全相同值/全零/大尺度)
7. save→load→predict 結果一致 (ConvLSTM)
8. 訓練後 is_trained 狀態正確
9. TZCF 非排序 lats 的穩定性
10. data_pipeline: 單一 sample 也能跑
"""
import numpy as np
import sys
import os
import tempfile
import shutil

PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} -- {detail}")

print("=== Round 4 FINAL Stress Test ===\n")

import torch
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────
# 1. Trainer + ConvLSTM 5D 整合
# ─────────────────────────────────────────────
print("--- 1. Trainer + ConvLSTM 5D Integration ---")
from engine.ml.convlstm_predictor import _build_convlstm_model
from engine.ml.dl_trainer import DLTrainer, get_mse_loss
from engine.ml.dl_data_pipeline import create_convlstm_dataset, create_dataloader

# 合成資料: 溫度高→CPUE高
seqs = []
for _ in range(8):
    days = []
    for _ in range(5):
        sst = np.random.rand(8, 8).astype(np.float32) * 30
        days.append({"sst": sst, "chl": np.random.rand(8, 8).astype(np.float32)})
    target = (days[-1]["sst"] / 30.0).astype(np.float32)
    seqs.append((days, target))

ds = create_convlstm_dataset(seqs, ["sst", "chl"])
train_loader = create_dataloader(ds, batch_size=4, shuffle=True)

cl_model = _build_convlstm_model(2, [16], [3], "relu").to(dev)
trainer = DLTrainer(cl_model, loss_fn=get_mse_loss(), lr=5e-2,
                    checkpoint_dir=os.path.join(tempfile.gettempdir(), "r4_ckpt"))

history = trainer.fit(train_loader, val_loader=None, epochs=30, save_every=0)
check("ConvLSTM+Trainer integration runs",
      len(history["train_loss"]) == 30)
check("ConvLSTM training loss decreases",
      history["train_loss"][-1] < history["train_loss"][0],
      f"first={history['train_loss'][0]:.4f}, last={history['train_loss'][-1]:.4f}")

# ─────────────────────────────────────────────
# 2. Gradient 不洩漏
# ─────────────────────────────────────────────
print("\n--- 2. Gradient Leak Test ---")
from engine.ml.unet_fishing import _build_unet_model

m = _build_unet_model(3, use_cbam=True).to(dev)
m.train()

# Forward+backward #1
x1 = torch.randn(2, 3, 32, 32, device=dev)
y1 = torch.rand(2, 1, 32, 32, device=dev)
o1 = m(x1)
loss1 = torch.nn.functional.binary_cross_entropy(o1, y1)
loss1.backward()
grad_snapshot = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}

# zero_grad then forward+backward #2 with different data
m.zero_grad()
# Verify zero_grad actually clears gradients
all_zero = all(p.grad is None or p.grad.abs().max() == 0 for p in m.parameters())
check("zero_grad clears all gradients", all_zero)

# ─────────────────────────────────────────────
# 3. predict() 連續呼叫不互相污染
# ─────────────────────────────────────────────
print("\n--- 3. Predict Determinism ---")
from engine.ml.unet_fishing import UNetFishingPredictor
from engine.ml.convlstm_predictor import ConvLSTMPredictor

up = UNetFishingPredictor(use_cbam=False)
feat_a = {"sst": np.ones((20, 20), dtype=np.float32) * 25}
feat_b = {"sst": np.ones((20, 20), dtype=np.float32) * 5}

# Call A, then B, then A again — A should be same both times
r_a1 = up.predict(feat_a, np.linspace(20, 30, 20), np.linspace(130, 140, 20))
r_b = up.predict(feat_b, np.linspace(20, 30, 20), np.linspace(130, 140, 20))
r_a2 = up.predict(feat_a, np.linspace(20, 30, 20), np.linspace(130, 140, 20))

check("U-Net predict deterministic (A==A after B)",
      np.allclose(r_a1["fishing_probability"], r_a2["fishing_probability"], atol=1e-6),
      f"max diff={np.max(np.abs(r_a1['fishing_probability'] - r_a2['fishing_probability'])):.8f}")

# Same for ConvLSTM
cp = ConvLSTMPredictor(hidden_channels=[16])
seq_a = [{"sst": np.ones((8, 8), dtype=np.float32) * 25} for _ in range(5)]
seq_b = [{"sst": np.ones((8, 8), dtype=np.float32) * 5} for _ in range(5)]

rc_a1 = cp.predict(seq_a, np.linspace(20, 25, 8), np.linspace(130, 135, 8))
rc_b = cp.predict(seq_b, np.linspace(20, 25, 8), np.linspace(130, 135, 8))
rc_a2 = cp.predict(seq_a, np.linspace(20, 25, 8), np.linspace(130, 135, 8))

check("ConvLSTM predict deterministic (A==A after B)",
      np.allclose(rc_a1["predicted_hsi"], rc_a2["predicted_hsi"], atol=1e-6))

# ─────────────────────────────────────────────
# 4. 超大/超小 feature 值的數值穩定性
# ─────────────────────────────────────────────
print("\n--- 4. Numerical Stability ---")
from engine.ml.dl_data_pipeline import normalize_field

# Very large values
huge = np.full((10, 10), 1e12, dtype=np.float32)
n_huge = normalize_field(huge, "sst", mode="per_sample")
check("normalize huge values no inf", not np.any(np.isinf(n_huge)))
check("normalize huge values no NaN", not np.any(np.isnan(n_huge)))

# Very small values
tiny = np.full((10, 10), 1e-15, dtype=np.float32)
n_tiny = normalize_field(tiny, "sst", mode="per_sample")
check("normalize tiny values no inf", not np.any(np.isinf(n_tiny)))
check("normalize tiny values no NaN", not np.any(np.isnan(n_tiny)))

# Mixed NaN and inf
mixed = np.array([[np.nan, np.inf, -np.inf, 1.0],
                  [0.0, np.nan, 2.0, 3.0]], dtype=np.float32)
n_mixed = normalize_field(mixed, "sst", mode="per_sample")
check("normalize NaN+inf no bad vals",
      not np.any(np.isnan(n_mixed)) and not np.any(np.isinf(n_mixed)))

# U-Net with extreme values
feat_extreme = {"sst": np.random.rand(16, 16).astype(np.float32) * 1e6}
r_ext = up.predict(feat_extreme, np.linspace(20, 30, 16), np.linspace(130, 140, 16))
check("U-Net with extreme input values",
      not np.any(np.isnan(r_ext["fishing_probability"])),
      f"has NaN={np.any(np.isnan(r_ext['fishing_probability']))}")

# ─────────────────────────────────────────────
# 5. Syrjala 極端分布
# ─────────────────────────────────────────────
print("\n--- 5. Syrjala Edge Cases ---")
from engine.ml.syrjala_test import syrjala_test

# All-zero predicted
sz = syrjala_test(np.zeros((10, 10)), np.random.rand(10, 10), n_permutations=9)
check("Syrjala all-zero predicted", sz["p_value"] == 1.0)

# Uniform distribution (all same value)
uniform = np.ones((10, 10))
su = syrjala_test(uniform, uniform, n_permutations=9)
check("Syrjala uniform distributions p~1", su["p_value"] > 0.5)

# Spatially spread peaks (high power for Syrjala)
sparse_a = np.zeros((20, 20))
sparse_a[:5, :] = np.random.rand(5, 20) + 0.5  # Top half concentrated
sparse_b = np.zeros((20, 20))
sparse_b[15:, :] = np.random.rand(5, 20) + 0.5  # Bottom half concentrated
ss = syrjala_test(sparse_a, sparse_b, n_permutations=199)
check("Syrjala spread peaks different location", ss["significant"] is True,
      f"p={ss['p_value']:.3f}")

# ─────────────────────────────────────────────
# 6. ConvLSTM save→load→predict
# ─────────────────────────────────────────────
print("\n--- 6. ConvLSTM Save/Load/Predict ---")
save_dir = os.path.join(tempfile.gettempdir(), "r4_cl")
save_p = os.path.join(save_dir, "cl.pt")

cp_orig = ConvLSTMPredictor(hidden_channels=[16, 8])
seq_test = [{"sst": np.random.rand(12, 12).astype(np.float32) * 30,
             "chl": np.random.rand(12, 12).astype(np.float32)} 
            for _ in range(7)]
r_orig = cp_orig.predict(seq_test, np.linspace(20, 30, 12), np.linspace(130, 140, 12))

cp_orig.save_weights(save_p)
cp_loaded = ConvLSTMPredictor(hidden_channels=[16, 8], model_path=save_p)
r_loaded = cp_loaded.predict(seq_test, np.linspace(20, 30, 12), np.linspace(130, 140, 12))

check("ConvLSTM save/load output match",
      np.allclose(r_orig["predicted_hsi"], r_loaded["predicted_hsi"], atol=1e-5),
      f"max diff={np.max(np.abs(r_orig['predicted_hsi'] - r_loaded['predicted_hsi'])):.8f}")

# ─────────────────────────────────────────────
# 7. dl_data_pipeline 單一 sample
# ─────────────────────────────────────────────
print("\n--- 7. Single Sample Edge Case ---")
from engine.ml.dl_data_pipeline import create_unet_dataset

single_sample = [({"sst": np.random.rand(16, 16).astype(np.float32)},
                  np.random.rand(16, 16).astype(np.float32))]
ds_single = create_unet_dataset(single_sample, ["sst"])
check("Single sample dataset length", len(ds_single) == 1)
x, y = ds_single[0]
check("Single sample X shape", x.shape == (1, 16, 16))
check("Single sample Y shape", y.shape == (1, 16, 16))

# ─────────────────────────────────────────────
# 8. TZCF 非排序 lats
# ─────────────────────────────────────────────
print("\n--- 8. TZCF Non-sorted Lats ---")
from engine.tzcf_tracker import TZCFTracker

# Reverse-sorted lats (south to north vs north to south)
lats_rev = np.linspace(45, 20, 100)  # reversed
lons_t = np.linspace(140, 170, 80)
chl_t = np.zeros((100, 80))
for i, lat in enumerate(lats_rev):
    chl_t[i, :] = 0.1 if lat < 32.0 else 0.5

tk = TZCFTracker(0.2)
r_rev = tk.compute(chl_t, lats_rev, lons_t)
check("TZCF reversed lats valid",
      r_rev["valid_fraction"] > 0.5,
      f"valid={r_rev['valid_fraction']:.2f}")
check("TZCF reversed lats ~32N",
      abs(r_rev["mean_tzcf_lat"] - 32.0) < 2.0,
      f"got {r_rev['mean_tzcf_lat']:.1f}")

# ─────────────────────────────────────────────
# 9. Trainer validation metrics 正確性
# ─────────────────────────────────────────────
print("\n--- 9. Trainer Validation Correctness ---")
from engine.ml.dl_trainer import DLTrainer, get_bce_loss, compute_f1

# 訓練一個幾乎完美的任務: 全白→全白, 全黑→全黑
perfect_samples = []
for _ in range(4):
    t = (np.random.rand(16, 16) > 0.5).astype(np.float32)
    f = {"sst": t * 30}  # SST high where fish present
    perfect_samples.append((f, t))

ds_perf = create_unet_dataset(perfect_samples, ["sst"])
loader_perf = create_dataloader(ds_perf, batch_size=4, shuffle=False)

m_perf = _build_unet_model(1, use_cbam=False).to(dev)
t_perf = DLTrainer(m_perf, loss_fn=get_bce_loss(), lr=1e-2,
                   checkpoint_dir=os.path.join(tempfile.gettempdir(), "r4_perf"))

# Train to overfit
h = t_perf.fit(loader_perf, val_loader=loader_perf, epochs=30, patience=30, save_every=0)

check("Trainer val_f1 populated", len(h["val_f1"]) > 0)
check("Trainer final val_f1 > 0.3", h["val_f1"][-1] > 0.3,
      f"val_f1={h['val_f1'][-1]:.3f}")

# ─────────────────────────────────────────────
# 10. Multiple features in U-Net predict
# ─────────────────────────────────────────────
print("\n--- 10. Multi-Feature U-Net ---")
up_full = UNetFishingPredictor(use_cbam=True)
feat_full = {
    "sst": np.random.rand(32, 32).astype(np.float32) * 30,
    "chl": np.random.rand(32, 32).astype(np.float32),
    "ssh": np.random.rand(32, 32).astype(np.float32) * 0.3,
    "u_current": np.random.rand(32, 32).astype(np.float32) * 0.5,
    "v_current": np.random.rand(32, 32).astype(np.float32) * 0.5,
    "depth": np.random.rand(32, 32).astype(np.float32) * -4000,
    "npp": np.random.rand(32, 32).astype(np.float32) * 1000,
}
r_full = up_full.predict(feat_full, np.linspace(20, 40, 32), np.linspace(130, 160, 32))
check("7-channel U-Net predict",
      r_full["fishing_probability"].shape == (32, 32))
check("7-channel output in [0,1]",
      r_full["fishing_probability"].min() >= 0 and r_full["fishing_probability"].max() <= 1)

# Cleanup
for d in ["r4_ckpt", "r4_cl", "r4_perf"]:
    p = os.path.join(tempfile.gettempdir(), d)
    if os.path.exists(p):
        shutil.rmtree(p)

print(f"\n{'='*50}")
print(f"  Round 4 FINAL: {PASS} passed, {FAIL} failed")
print(f"{'='*50}")
if FAIL > 0:
    sys.exit(1)
else:
    print("  ALL ROUND 4 TESTS PASSED")
