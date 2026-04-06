"""Round 2 regression tests: edge cases, squeeze safety, key errors."""
import numpy as np
import sys

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

print("=== Round 2 Regression Tests ===\n")

import torch
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. U-Net squeeze safety with H=1
from engine.ml.unet_fishing import UNetFishingPredictor, _build_unet_model
p = UNetFishingPredictor(use_cbam=False)
feat = {"sst": np.random.rand(1, 50).astype(np.float32) * 30}
r = p.predict(feat, np.array([25.0]), np.linspace(130, 155, 50))
check("U-Net squeeze safety H=1",
      r["fishing_probability"].shape == (1, 50),
      f"got {r['fishing_probability'].shape}")

# 2. U-Net squeeze safety with W=1
feat2 = {"sst": np.random.rand(50, 1).astype(np.float32) * 30}
r2 = p.predict(feat2, np.linspace(20, 40, 50), np.array([135.0]))
check("U-Net squeeze safety W=1",
      r2["fishing_probability"].shape == (50, 1),
      f"got {r2['fishing_probability'].shape}")

# 3. ConvLSTM squeeze safety H=1
from engine.ml.convlstm_predictor import ConvLSTMPredictor
cl = ConvLSTMPredictor(hidden_channels=[16])
seq = [{"sst": np.random.rand(1, 10).astype(np.float32) * 30} for _ in range(5)]
r3 = cl.predict(seq, np.array([25.0]), np.linspace(130, 140, 10))
check("ConvLSTM squeeze safety H=1",
      r3["predicted_hsi"].shape == (1, 10),
      f"got {r3['predicted_hsi'].shape}")

# 4. ConvLSTM predict with non-first feature key
seq2 = [{"chl": np.random.rand(8, 8).astype(np.float32)} for _ in range(5)]
r4 = cl.predict(seq2, np.linspace(20, 25, 8), np.linspace(130, 135, 8))
check("ConvLSTM non-first key", r4["predicted_hsi"].shape == (8, 8))

# 5. ConvLSTM raises ValueError for no valid keys
try:
    seq3 = [{"garbage": np.random.rand(8, 8)} for _ in range(5)]
    cl.predict(seq3, np.linspace(20, 25, 8), np.linspace(130, 135, 8))
    check("ConvLSTM no valid keys raises", False, "should have raised ValueError")
except ValueError:
    check("ConvLSTM no valid keys raises", True)

# 6. ConvLSTM empty sequence
r5 = cl.predict([], np.array([25.0]), np.array([135.0]))
check("ConvLSTM empty sequence", r5["predicted_hsi"] is None)

# 7. BCELoss correctness with sigmoid output
from engine.ml.dl_trainer import get_bce_loss
loss_fn = get_bce_loss()
pred_t = torch.tensor([0.7, 0.3, 0.9])
target_t = torch.tensor([1.0, 0.0, 1.0])
loss = loss_fn(pred_t, target_t)
check("BCELoss with sigmoid output", loss.item() > 0 and not torch.isnan(loss))

# 8. U-Net non-standard input sizes
m = _build_unet_model(3, use_cbam=False).to(dev).eval()
all_ok = True
for h, w in [(7, 7), (15, 33), (100, 100), (1, 64), (64, 1), (3, 3)]:
    x = torch.randn(1, 3, h, w, device=dev)
    with torch.no_grad():
        o = m(x)
    if o.shape != (1, 1, h, w):
        check(f"U-Net {h}x{w}", False, f"out={o.shape}")
        all_ok = False
    else:
        check(f"U-Net {h}x{w}", True)

# 9. Training convergence with validation (round 2: test val metrics)
from engine.ml.dl_trainer import DLTrainer, get_bce_loss, compute_f1
from engine.ml.dl_data_pipeline import create_unet_dataset, create_dataloader

samples = []
for _ in range(8):
    feat = {"sst": np.random.rand(32, 32).astype(np.float32) * 30 + 10}
    tgt = (feat["sst"] > 25).astype(np.float32)
    samples.append((feat, tgt))

dataset = create_unet_dataset(samples, ["sst"])
train_ds = torch.utils.data.Subset(dataset, [0, 1, 2, 3])
val_ds = torch.utils.data.Subset(dataset, [4, 5, 6, 7])

train_loader = create_dataloader(train_ds, batch_size=2, shuffle=True)
val_loader = create_dataloader(val_ds, batch_size=2, shuffle=False)

tiny = _build_unet_model(1, use_cbam=False).to(dev)
trainer = DLTrainer(tiny, loss_fn=get_bce_loss(), lr=1e-2,
                    checkpoint_dir="models/checkpoints_r2")

history = trainer.fit(train_loader, val_loader, epochs=20, patience=15,
                      save_every=0, model_name="r2test")

check("Trainer val metrics populated",
      len(history["val_f1"]) > 0 and len(history["val_ssim"]) > 0)
check("Trainer val loss populated",
      len(history["val_loss"]) > 0)
check("Trainer LR populated",
      len(history["lr"]) > 0)

# Check best model was loaded back
import os
best_path = os.path.join("models", "checkpoints_r2", "r2test_best.pt")
check("Best model checkpoint exists", os.path.exists(best_path))

# 10. Data pipeline: create_convlstm_dataset
from engine.ml.dl_data_pipeline import create_convlstm_dataset
seqs = []
for _ in range(4):
    seq_days = [
        {"sst": np.random.rand(8, 8).astype(np.float32) * 30,
         "chl": np.random.rand(8, 8).astype(np.float32)}
        for _ in range(10)
    ]
    tgt = np.random.rand(8, 8).astype(np.float32)
    seqs.append((seq_days, tgt))

ds = create_convlstm_dataset(seqs, ["sst", "chl"])
check("ConvLSTM dataset length", len(ds) == 4)
x_sample, y_sample = ds[0]
check("ConvLSTM dataset X shape", x_sample.shape == (10, 2, 8, 8),
      f"got {x_sample.shape}")
check("ConvLSTM dataset Y shape", y_sample.shape == (1, 8, 8),
      f"got {y_sample.shape}")

# 11. TZCF: southern hemisphere
from engine.tzcf_tracker import TZCFTracker
lats_sh = np.linspace(-45, -20, 100)
lons_sh = np.linspace(140, 170, 80)
chl_sh = np.zeros((100, 80))
for i, lat in enumerate(lats_sh):
    chl_sh[i, :] = 0.5 if lat < -32.0 else 0.1
t = TZCFTracker(chl_threshold=0.2)
r_sh = t.compute(chl_sh, lats_sh, lons_sh)
check("TZCF southern hemisphere",
      abs(r_sh["mean_tzcf_lat"] - (-32.0)) < 1.0,
      f"got {r_sh['mean_tzcf_lat']:.1f}")

# 12. Syrjala with large arrays
from engine.ml.syrjala_test import syrjala_test
big_a = np.random.rand(100, 100)
big_b = np.random.rand(100, 100)
s = syrjala_test(big_a, big_b, n_permutations=99)
check("Syrjala large arrays", 0 <= s["p_value"] <= 1)

print(f"\n{'='*50}")
print(f"  Round 2: {PASS} passed, {FAIL} failed")
print(f"{'='*50}")
if FAIL > 0:
    sys.exit(1)
else:
    print("  ALL ROUND 2 TESTS PASSED")
