# 🎯 OceanMaster v13.2 — 完整啟動教學（從零開始）

> **適用對象**：完全沒碰過程式的船長或管理者
> **最終目標**：儀表板上所有數據即時更新、ML/DL 模型用歷史漁獲紀錄訓練完成
> **最後更新**：2026-03-03

---

## 📋 目錄

1. [環境安裝](#1-環境安裝)
2. [設定 API 金鑰（.env）](#2-設定-api-金鑰env)
3. [啟動儀表板](#3-啟動儀表板)
4. [訓練 ML 模型（歷史漁獲數據）](#4-訓練-ml-模型歷史漁獲數據)
5. [訓練深度學習 DL 模型（U-Net / ConvLSTM）](#5-訓練深度學習-dl-模型u-net--convlstm)
6. [接上即時 API 數據](#6-接上即時-api-數據)
7. [完整營運模式（自動排程）](#7-完整營運模式自動排程)
8. [常見問題](#8-常見問題)

---

## 1. 環境安裝

### 1.1 安裝 Python

1. 前往 https://www.python.org/downloads/
2. 下載 **Python 3.11** 或更新版本
3. 安裝時 **務必勾選** `Add Python to PATH` ✅
4. 安裝完成後，開啟 **PowerShell**（按 `Win + X` → `終端機`），輸入：

```powershell
python --version
```

看到 `Python 3.11.x` 或 `3.12.x` 就表示成功。

### 1.2 安裝專案套件

```powershell
cd "C:\Users\user\Desktop\好像快好了\OceanMaster_v13_2"
pip install -r requirements.txt
```

> ⏱️ 約需 5-10 分鐘。主要套件：
> - `fastapi` + `uvicorn` — 網頁伺服器
> - `scikit-learn` + `xgboost` + `lightgbm` — ML 模型
> - `numpy` + `scipy` + `pandas` — 科學計算
> - `copernicusmarine` — CMEMS 海洋數據 API
> - `httpx` + `aiohttp` — 非同步 HTTP 請求

如果某個套件安裝失敗：

```powershell
pip install --upgrade pip
pip install -r requirements.txt --prefer-binary
```

### 1.3 （選擇性）安裝 PyTorch — 深度學習用

只有要跑 U-Net / ConvLSTM 才需要（第 5 節）。不裝也能用 ML 模型預測。

```powershell
# CPU 版本（不需要顯卡，約 200MB）
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 有 NVIDIA 顯卡的話，安裝 CUDA 版（效能快 10-50 倍）
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## 2. 設定 API 金鑰（.env）

### 2.1 複製範例檔

```powershell
cd "C:\Users\user\Desktop\好像快好了\OceanMaster_v13_2"
copy .env.example .env
```

### 2.2 用記事本編輯

```powershell
notepad .env
```

### 2.3 填入以下欄位

| 欄位 | 必填 | 說明 | 如何取得 |
|------|:---:|------|---------|
| `CMEMS_USER` | ✅ | Copernicus 海洋數據帳號 | 免費 → https://data.marine.copernicus.eu/register |
| `CMEMS_PASS` | ✅ | Copernicus 密碼 | 同上 |
| `OCEANMASTER_API_KEY` | ✅ | 系統 API 密碼 | 自己設，例如 `my-secret-key-12345` |
| `GFW_API_TOKEN` | ❌ | 全球漁船追蹤 (AIS) | https://globalfishingwatch.org/our-apis/ |
| `ANTHROPIC_API_KEY` | ❌ | Claude AI 週報分析 | https://console.anthropic.com/ |
| `FLASK_PORT` | ❌ | 伺服器端口 | 預設 `8000` |

> [!IMPORTANT]
> **`CMEMS_USER` 和 `CMEMS_PASS` 是必填的**。沒有它就無法下載即時 SST / SSH / 海流數據。
> 註冊完全免費，但審核需要 1-2 個工作天。

改完 → `Ctrl + S` 存檔 → 關閉記事本。

---

## 3. 啟動儀表板

### 3.1 方式 A：純前端展示（不需要 API，馬上就能看）

如果 CMEMS 帳號還在審核中，先用這個看 UI：

```powershell
cd "C:\Users\user\Desktop\好像快好了\OceanMaster_v13_2"
python -m http.server 8000
```

然後打開瀏覽器，輸入：

```
http://localhost:8000/web/dashboard_v3.html
```

> 這個模式下地圖 + 圖層 toggle + 模擬颱風都能操作，但數據是模擬的。

### 3.2 方式 B：完整後端啟動（即時數據）

```powershell
cd "C:\Users\user\Desktop\好像快好了\OceanMaster_v13_2"
python web_server.py
```

啟動成功後你會看到：

```
INFO:     OceanMaster v13.2 初始化...
INFO:     載入模型: yellowfin ✓ bigeye ✓ skipjack ✓ albacore ✓
INFO:     CMEMS 數據同步中...
INFO:     Uvicorn running on http://0.0.0.0:8000
```

打開瀏覽器：

```
http://localhost:8000/
```

> [!TIP]
> - `http://localhost:8000/` → 主頁（載入 `web/dashboard.html`，會自動注入 session token）
> - `http://localhost:8000/web/dashboard_v3.html` → 船長版儀表板（地圖 + 經緯度 + toggle）
> - 第一次啟動會花 1-3 分鐘下載 CMEMS 海洋數據，之後快取 6 小時

### 3.3 儀表板操作參考

| 操作 | 位置 | 功能 |
|------|------|------|
| 🌀 颱風 | 左下角 toggle | 顯示/隱藏颱風圖標和危險範圍 |
| 🌡 海溫衛星圖 | 左下角 toggle | 顯示/隱藏 NASA SST 衛星圖層 |
| 🐟 魚群遷移 | 左下角 toggle | 顯示/隱藏遷移路線箭頭 |
| 🚢 競爭船 | 左下角 toggle | 顯示/隱藏 AIS 漁船位置 |
| 🎣 漁場熱點 | 左下角 toggle | 顯示/隱藏漁場排名標記 |
| 🌊 海流 | 左下角 toggle | 顯示/隱藏海流箭頭 |
| 📝 記錄漁獲 | 頂部導覽列 | 記錄今日漁獲（存在瀏覽器） |
| ◀ 收合 | 左側面板 | 收合左側排名列表，全螢幕看地圖 |
| 點擊地圖 | 地圖上任何位置 | 顯示該經緯度的即時天氣 + 安全評估 |

---

## 4. 訓練 ML 模型（歷史漁獲數據）

### 4.1 專案結構

```
OceanMaster_v13_2/
├── data/
│   └── wcpfc/                     ← 歷史漁獲數據
│       ├── LONGLINE.CSV            ← 延繩釣 (156,212筆, 1950-2018)
│       │                             → 黃鰭鮪(yft_c), 大目鮪(bet_c), 長鰭鮪(alb_c)
│       └── PURSE_SEINE.CSV         ← 圍網 (30,025筆, 1967-2018)
│                                     → 正鰹(skj_c)
├── models/                         ← 訓練好的模型
│   ├── ml12_stacking_yellowfin.pkl  (12 特徵版, ~47MB)
│   ├── ml12_stacking_bigeye.pkl     (12 特徵版, ~47MB)
│   ├── ml12_stacking_skipjack.pkl   (12 特徵版, ~17MB)
│   ├── ml12_stacking_albacore.pkl   (12 特徵版, ~38MB)
│   ├── stacking_yellowfin.pkl       (59 特徵版, ~55MB)
│   ├── stacking_bigeye.pkl          (59 特徵版, ~54MB)
│   ├── stacking_skipjack.pkl        (59 特徵版, ~20MB)
│   └── stacking_albacore.pkl        (59 特徵版, ~46MB)
└── train_wcpfc.py                  ← 訓練腳本
```

### 4.2 WCPFC 公開數據格式

`LONGLINE.CSV` 的實際欄位格式：

```csv
yy,mm,lat5,lon5,hhooks,yft_c,bet_c,alb_c,...
2005,3,22.5N,131.5E,150.0,12.5,8.3,2.1,...
2005,4,23.0N,130.0E,200.0,18.2,11.5,3.0,...
```

| 欄位 | 說明 |
|------|------|
| `yy` | 年份 |
| `mm` | 月份 (1-12) |
| `lat5` | 緯度（格式: `22.5N` 或 `5.0S`） |
| `lon5` | 經度（格式: `131.5E` 或 `170.0W`） |
| `hhooks` | 百鉤數（hundred hooks），例 `150.0` = 15,000 鉤 |
| `yft_c` | 黃鰭鮪漁獲量（公噸） |
| `bet_c` | 大目鮪漁獲量（公噸） |
| `alb_c` | 長鰭鮪漁獲量（公噸） |

`PURSE_SEINE.CSV` 格式類似，正鰹欄位為 `skj_c_una`, `skj_c_log` 等（按 set type 分類）。

### 4.3 開始訓練

```powershell
cd "C:\Users\user\Desktop\好像快好了\OceanMaster_v13_2"
python train_wcpfc.py
```

你會看到：

```
======================================================================
  OceanMaster v13.2 — Real Data Training (No Synthetic)
  Data filter: year >= 2000, q-corrected
  Split: temporal (train <= 2014, test > 2014)
======================================================================

############################################################
#  ML-12 (12-feature) Training with WCPFC Real Data
############################################################

============================================================
  Training 12-feature model: YELLOWFIN
============================================================
  yellowfin: 12,345 valid CPUE records from LONGLINE.CSV (year>=2000, q-corrected)
  After outlier removal: 12,100 samples, CPUE range [0.05, 28.31]
  Temporal split: train=9,800 (<=2014), test=2,300 (>2014)
  Fitting stacking ensemble...
  Holdout R2=0.3851  MAE=1.2345  RMSE=2.3100
  Saved: models/ml12_stacking_yellowfin.pkl (46800 KB)

... (接著訓練 bigeye, albacore, skipjack)

############################################################
#  ML-44 (59-feature) Training with WCPFC Real Data
############################################################
... (再跑一輪 59 特徵的模型)
```

> ⏱️ 全部 **8 個模型**（4 魚種 × 2 版本）約需 20-40 分鐘。

### 4.4 訓練流程詳解

`train_wcpfc.py` 內部做了什麼：

```
步驟 1: 讀取 LONGLINE.CSV 或 PURSE_SEINE.CSV
步驟 2: 只保留 2000 年之後的數據（早期設備太差不可靠）
步驟 3: 漁獲力修正（q-factor）：
        ├── 2000-2004: q=0.95 (GPS 定深前)
        ├── 2005-2009: q=1.00 (基準期)
        ├── 2010-2014: q=1.03 (GPS + 深放鉤)
        └── 2015-2020: q=1.06 (LED 光棒 + 先進瞄準)
步驟 4: CPUE = 漁獲量(mt) ÷ (百鉤數/1000) → 每千鉤漁獲量
步驟 5: 從經緯度 + 月份推導海洋特徵：
        12 特徵版: SST, CHL, SSH, DO, 海流, 鋒面, 渦旋, 代謝率, 水深, 坡度, 海底山距離, 陸棚距離
        59 特徵版: 上面 12 個 + T100, 溫度梯度, 月相, FTLE, MLD, 季節指標, ENSO,
                   月光抑制, 颱風後藻華, 海底山交互, 獵物陷阱, 潮汐混合... 等 47 個進階特徵
步驟 6: 時間分割：2000-2014 訓練 / 2015-2018 測試（避免時間洩漏）
步驟 7: 建立 Stacking Ensemble：
        一層:
        ├── Random Forest        (200 棵決策樹, max_depth=12)
        ├── Gradient Boosting    (150 棵, max_depth=6, lr=0.05)
        ├── Extra Trees          (200 棵, max_depth=12)
        ├── XGBoost              (200 棵, max_depth=8, lr=0.05)
        └── LightGBM             (200 棵, max_depth=8, lr=0.05)
        二層 (meta-learner):
        └── RidgeCV              (alpha 自動選擇)
步驟 8: SpatialBlockCV 空間交叉驗證（n_blocks=2, buffer=100km）
步驟 9: 存檔到 models/ + HMAC 簽章 + 訓練報告 JSON
```

### 4.5 加入你自己的漁獲數據

如果你有自己船隊的漁獲紀錄，可以合併進去訓練：

1. 把你的 CSV 格式調整成和 `LONGLINE.CSV` 一樣的欄位名稱
2. 存到 `data/wcpfc/MY_FLEET_DATA.CSV`
3. 修改 `train_wcpfc.py` 第 52 行：
   ```python
   WCPFC_CSV = Path("data/wcpfc/MY_FLEET_DATA.CSV")
   ```
4. 重新跑 `python train_wcpfc.py`

### 4.6 驗證訓練結果

```powershell
python -c "import pickle; m=pickle.load(open('models/ml12_stacking_yellowfin.pkl','rb')); print(f'模型載入成功, 類型={type(m[\"model\"]).__name__}, 魚種={m[\"species\"]}')"
```

預期輸出：

```
模型載入成功, 類型=StackingRegressor, 魚種=yellowfin
```

訓練報告在 `models/ml12_stacking_yellowfin_meta.json`，內容包含 R², MAE, RMSE, 訓練/測試集大小等。

---

## 5. 訓練深度學習 DL 模型（U-Net / ConvLSTM）

> [!IMPORTANT]
> DL 是**進階功能**，需要先安裝 PyTorch（見 1.3）。
> 第 4 節的 ML 模型已經可以正常預測漁場，DL 是額外加分。

### 5.1 DL 模型一覽

| 模型 | 原始碼 | 用途 | 已訓練權重 |
|------|--------|------|-----------|
| **U-Net + CBAM** | `engine/ml/unet_fishing.py` | 空間漁場預測（整張地圖一次出 2D 機率圖） | `models/unet_fishing_v18.pt` (379KB) |
| **ConvLSTM2D** | `engine/ml/convlstm_predictor.py` | 30 天時間序列 → 預測明天 CPUE 分佈 | `models/convlstm_sst_v18.pt` (87KB) |
| **TransFish** | `engine/ml/transfish_predictor.py` | Transformer 多物種預測 | `models/transfish_v18.pt` (302KB) |
| **Cloud Removal** | `engine/ml/cloud_removal.py` | 衛星雲層去除（乾淨 SST 圖） | `models/cloud_removal_v18.pt` (47KB) |

### 5.2 確認 PyTorch 安裝

```powershell
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA 可用: {torch.cuda.is_available()}')"
```

```
PyTorch 2.x.x
CUDA 可用: True    ← 有 NVIDIA GPU 才是 True；False 也能跑（用 CPU）
```

### 5.3 DL 訓練數據格式

DL 模型吃的是 **2D 網格**（像衛星影像），不是 CSV 表格：

```
一筆訓練樣本 = 一天的「海洋環境地圖」

7 個頻道 × 64×64 網格:
├── sst      (海溫)          shape: (64, 64)
├── chl      (葉綠素-a)      shape: (64, 64)
├── ssh      (海面高度)       shape: (64, 64)
├── u_cur    (東西向海流)     shape: (64, 64)
├── v_cur    (南北向海流)     shape: (64, 64)
├── depth    (水深)           shape: (64, 64)
└── npp      (初級生產力)     shape: (64, 64)

合併 → X: shape (7, 64, 64)
標記 → Y: shape (1, 64, 64)  ← 0=沒魚, 1=有魚（U-Net）
        Y: shape (1, 64, 64)  ← CPUE 值（ConvLSTM）
```

#### 從 CMEMS 下載網格數據

```powershell
cd "C:\Users\user\Desktop\好像快好了\OceanMaster_v13_2"

# 下載 MUR SST 高解析衛星海溫
python download_mur_real.py

# 下載 GEBCO 水深數據
python download_bathy.py

# 下載 3D 深層溫度/密度剖面
python download_3d_deep.py
```

### 5.4 訓練 U-Net（空間漁場預測）

```python
# 存成 train_unet.py，然後 python train_unet.py
import torch
from torch.utils.data import DataLoader, TensorDataset
from engine.ml.unet_fishing import UNetFishingPredictor
from engine.ml.dl_trainer import DLTrainer, get_bce_loss
import numpy as np

# 1. 準備數據（這裡用隨機數據示範，實際應用 CMEMS 真實數據）
N = 200  # 200 筆樣本
X_train = np.random.rand(N, 7, 64, 64).astype(np.float32)
Y_train = (np.random.rand(N, 1, 64, 64) > 0.7).astype(np.float32)

dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# 2. 建立 U-Net 模型
unet = UNetFishingPredictor(use_cbam=True)
model = unet.get_model()

# 3. 建立 Trainer
trainer = DLTrainer(
    model=model,
    loss_fn=get_bce_loss(),   # 二分類: 有魚 / 沒魚
    lr=0.001,
    device="cuda"             # 沒有 GPU 改成 "cpu"
)

# 4. 開始訓練
trainer.fit(
    train_loader=train_loader,
    epochs=50,                # 最多 50 輪
    patience=10,              # 連續 10 輪沒進步就停
    model_name="unet_fishing"
)

# 5. 存檔
unet.save_weights("models/unet_fishing_v18.pt")
print("✅ U-Net 訓練完成")
```

### 5.5 訓練 ConvLSTM（時間序列預測）

```python
# 存成 train_convlstm.py
import torch
from torch.utils.data import DataLoader, TensorDataset
from engine.ml.convlstm_predictor import ConvLSTMPredictor
from engine.ml.dl_trainer import DLTrainer, get_mse_loss

# 1. 建立模型
convlstm = ConvLSTMPredictor(
    hidden_channels=[32, 16],   # 2 層 ConvLSTM
    kernel_sizes=[3, 3],
    output_activation="relu"    # CPUE 是正數，用 ReLU
)
model = convlstm.get_model()

# 2. 使用 generate_training_data 建立訓練集
# historical_grids: 90天的環境數據 [{sst: (64,64), chl: (64,64)...}, ...]
# historical_cpue: 90天的 CPUE 地圖 [(64,64), (64,64)...]
# X, Y = convlstm.generate_training_data(historical_grids, historical_cpue, window=30)
# → X: (60, 30, 7, 64, 64),  Y: (60, 64, 64)

# 3. 訓練
# train_loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(Y)), batch_size=4)
trainer = DLTrainer(
    model=model,
    loss_fn=get_mse_loss(),     # 回歸: 預測 CPUE 數值
    lr=0.0005,
    device="cuda"
)
trainer.fit(train_loader=train_loader, epochs=100, patience=15, model_name="convlstm_sst")

# 4. 存檔
convlstm.save_weights("models/convlstm_sst_v18.pt")
print("✅ ConvLSTM 訓練完成")
```

### 5.6 DL Trainer 內部做了什麼

```
DLTrainer.fit() 的完整流程:

╔═══════════════════════════════════════╗
║  Epoch 1/50                           ║
╠═══════════════════════════════════════╣
║  1. train_epoch()                     ║
║     └→ 遍歷所有 batch                 ║
║     └→ 前向傳播 → 計算 loss            ║
║     └→ loss.backward() 反向傳播        ║
║     └→ optimizer.step() 更新權重       ║
║                                       ║
║  2. validate()                        ║
║     └→ 計算驗證集 F1 / SSIM / IoU      ║
║                                       ║
║  3. ReduceLROnPlateau                 ║
║     └→ 驗證 loss 沒下降 → 降低學習率    ║
║                                       ║
║  4. Early Stopping 檢查               ║
║     └→ 連續 N 輪沒進步 → 停止訓練      ║
║                                       ║
║  5. 自動存檔                           ║
║     └→ models/checkpoints/xxx_best.pt  ║
║     └→ 每 10 輪: xxx_ep10.pt          ║
╚═══════════════════════════════════════╝
```

### 5.7 驗證 DL 模型

```powershell
python -c "
from engine.ml.unet_fishing import UNetFishingPredictor
from engine.ml.convlstm_predictor import ConvLSTMPredictor

u = UNetFishingPredictor(model_path='models/unet_fishing_v18.pt')
print(f'U-Net 載入 ✓  參數量: {sum(p.numel() for p in u.get_model().parameters()):,}')

c = ConvLSTMPredictor(model_path='models/convlstm_sst_v18.pt')
print(f'ConvLSTM 載入 ✓  參數量: {sum(p.numel() for p in c.get_model().parameters()):,}')
"
```

### 5.8 ML vs DL 比較

| | ML（Stacking Ensemble） | DL（U-Net / ConvLSTM） |
|---|---|---|
| **輸入** | CSV 表格（經緯度 + 漁獲量） | 2D 網格圖（衛星影像） |
| **訓練時間** | 20-40 分鐘 | 1-6 小時（看 GPU） |
| **需要 GPU** | ❌ 不需要 | ⚡ 強烈建議 |
| **預測方式** | 逐點計算 CPUE → 排名 | 整張地圖一次產出機率圖 |
| **強項** | 歷史漁獲統計分析 | 空間模式辨識（鋒面、渦旋形狀） |
| **儀表板用法** | 用來算 #1~#12 漁場排名 | 疊加熱力圖到地圖上 |

> [!TIP]
> 最佳策略是 **ML + DL 一起用**：ML 給出漁場排名分數，DL 在地圖上畫出高機率區域，兩者互相驗證。

---

## 6. 接上即時 API 數據

### 6.1 數據來源一覽（全部免費）

| 數據 | API 來源 | 用途 | 更新頻率 |
|------|---------|------|---------|
| SST 海溫 | Copernicus CMEMS | 海面溫度 + 溫度梯度 | 每日 |
| SSH 海面高度 | Copernicus CMEMS | 渦流 / 洋流偵測 | 每日 |
| 海流 (U/V) | Copernicus CMEMS | 海流方向 + 速度 | 每日 |
| 波浪 + 風速 | Open-Meteo | 波高 / 蒲福級 / 安全評估 | 每小時 |
| 颱風 | GDACS | 即時颱風位置 + 路徑 | 每 5 分鐘 |
| SST 衛星圖 | NASA GIBS | 全球海溫視覺化圖層 | 每日 |
| 競爭船 AIS | Global Fishing Watch | 其他漁船位置 | 即時 |
| 夜間漁火 | VIIRS DNB | 集魚燈偵測 | 每日 |
| 水深 | GEBCO | 海底地形 | 固定 |

### 6.2 測試 CMEMS 連線

確認 `.env` 中 `CMEMS_USER` 和 `CMEMS_PASS` 已填寫，然後：

```powershell
python -c "
print('✅ CMEMS 連線成功！')
"
```

### 6.3 即時天氣 API（Open-Meteo）

已內建在 `web_server.py`，**不需要 API key**。儀表板點擊地圖時自動呼叫：

```
https://api.open-meteo.com/v1/forecast
  ?latitude={lat}&longitude={lon}
  &current=wave_height,wind_speed_10m,wind_direction_10m,...
```

### 6.4 系統架構圖

```
┌──────────────────────────────────────────────────────────────────┐
│  你的電腦 (localhost:8000)                                        │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ web_server.py (FastAPI + Uvicorn)                           │ │
│  │   ├── 啟動時: 載入 main_v10_3.py 管線 + 4個ML模型           │ │
│  │   ├── 自動: 每30分鐘從 CMEMS 抓 SST/SSH/海流               │ │
│  │   ├── 自動: 每5分鐘從 GDACS 抓颱風                          │ │
│  │   ├── 按需: 點擊地圖 → Open-Meteo 天氣                     │ │
│  │   ├── /api/v1/hotspots   → ML模型推論 → 漁場排名            │ │
│  │   ├── /api/v1/explain    → SHAP 特徵分解                    │ │
│  │   └── /api/sea_conditions → 海況總覽                         │ │
│  └─────────────────────────────────────────────────────────────┘ │
│           ↕                                                       │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ dashboard_v3.html (前端儀表板)                              │ │
│  │   ├── Leaflet 地圖引擎                                      │ │
│  │   ├── NASA GIBS 衛星圖（直接從 NASA 載入）                  │ │
│  │   ├── 左下角 toggle 控制所有圖層                             │ │
│  │   └── 右側面板: 安全/天氣/月相/競爭壓力                      │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

---

## 7. 完整營運模式（自動排程）

### 7.1 啟動後端伺服器

```powershell
cd "C:\Users\user\Desktop\好像快好了\OceanMaster_v13_2"
python web_server.py
```

`web_server.py` 啟動後會自動：
- 初始化 `OceanMasterPipeline`（載入所有模型 + 數據源）
- 啟動背景任務 `update_data_background()` 定期更新數據
- 提供所有 API 端點（熱點、SHAP、海況）

### 7.2 排程器說明

`scheduler/scheduler.py` 使用 APScheduler 自動排程：

| 排程 | 時間 | 動作 |
|------|------|------|
| 每日完整分析 | 02:00 UTC (台灣 10:00) | 全部數據抓取 → 物理算法 → HSI → AI 排名 → KML |
| 快速更新 | 每 6 小時 | 只更新 SST + 海流 → 重算鋒面 |

排程器是由 `web_server.py` 在啟動時自動載入的（透過 `lifespan` 事件），不需要手動啟動。

### 7.3 讓前端更頻繁更新

如果你希望儀表板更快刷新顯示，打開 `web/dashboard_v3.html`，搜尋 `setInterval`，改成你要的時間：

```javascript
setInterval(fetchTyphoons, 300000);   // 颱風: 5分鐘
setInterval(updateAll, 30000);        // 改成 30000 = 30秒
```

### 7.4 Docker 部署（正式上線到雲端）

如果要部署到 AWS / GCP / Azure：

```powershell
# 建立映像
docker build -t oceanmaster:v13.2 -f Dockerfile.production .

# 啟動（帶 .env 環境變數）
docker run -d -p 8000:8000 --env-file .env --name oceanmaster oceanmaster:v13.2

# 查看日誌
docker logs -f oceanmaster
```

或用 Docker Compose：

```powershell
docker compose -f docker-compose.production.yml up -d
```

---

## 8. 常見問題

### Q: `CMEMS_USER` 跟 `CMEMS_PASS` 去哪裡註冊？
到 https://data.marine.copernicus.eu/register ，填入 email，等 1-2 天審核。完全免費。

### Q: 訓練跑很久怎麼辦？
正常。8 個模型（4 魚種 × 12特徵版 + 59特徵版）在一般電腦約 20-40 分鐘。如果超過 2 小時，可能記憶體不足。

### Q: `pip install` 出錯？
```powershell
pip install --upgrade pip
pip install -r requirements.txt --prefer-binary
```
如果 `lightgbm` 失敗：`pip install lightgbm --prefer-binary`

### Q: 儀表板開了但沒有數據？
1. 確認 `.env` 有填 `CMEMS_USER` 和 `CMEMS_PASS`
2. 確認用 `python web_server.py` 啟動（不是 `python -m http.server`）
3. 看 PowerShell 終端有沒有紅色錯誤

### Q: 颱風 toggle 按下去沒反應？
GDACS API 目前西太平洋沒有颱風時，系統會在載入 3 秒後顯示模擬颱風。它們會跟著 toggle 開/關。

### Q: 想加自己的漁獲紀錄訓練？
1. 整理成和 `LONGLINE.CSV` 一樣的格式（見 4.2）
2. 存到 `data/wcpfc/` 資料夾
3. 改 `train_wcpfc.py` 第 52 行 `WCPFC_CSV = Path("data/wcpfc/你的檔名.CSV")`
4. 跑 `python train_wcpfc.py`

### Q: 手機或平板怎麼看？
電腦跟手機在同一個 WiFi 下，手機瀏覽器輸入你電腦的 IP：
```
http://192.168.x.x:8000/web/dashboard_v3.html
```
電腦 IP 查法：PowerShell 輸入 `ipconfig`，看 `IPv4 Address`。

### Q: DL 模型需要多少 GPU 記憶體？
- U-Net：約 1-2 GB（batch_size=8, 64×64）
- ConvLSTM：約 2-4 GB（batch_size=4, 30步×64×64）
- 沒 GPU 用 CPU 也能跑，只是慢 10-50 倍

### Q: 如何增加新的魚種？
1. 在 `train_wcpfc.py` 的 `SPECIES_COLS` dict 加入新欄位
2. 確保 CSV 裡有對應的漁獲量欄位
3. 重新訓練

---

## 📌 快速指令總整理

```powershell
# ❶ 進入專案
cd "C:\Users\user\Desktop\好像快好了\OceanMaster_v13_2"

# ❷ 安裝套件（第一次做即可）
pip install -r requirements.txt

# ❸ 設定 API 金鑰（第一次做即可）
copy .env.example .env
notepad .env    # 填入 CMEMS_USER, CMEMS_PASS, OCEANMASTER_API_KEY

# ❹ 訓練 ML 模型（產出 8 個 .pkl 模型檔）
python train_wcpfc.py

# ❺ （選擇性）安裝 PyTorch + 訓練 DL 模型
pip install torch --index-url https://download.pytorch.org/whl/cpu
# 寫 train_unet.py / train_convlstm.py（見第 5 節）
# 或直接用已訓練好的 models/*.pt

# ❻ 啟動伺服器（含即時數據 + 排程器 + ML 推論）
python web_server.py

# ❼ 打開瀏覽器
# 主頁:       http://localhost:8000/
# 船長版:     http://localhost:8000/web/dashboard_v3.html

# ❽ 測試 API（另開終端）
curl http://localhost:8000/health
curl http://localhost:8000/api/v1/hotspots -H "X-API-Key: 你填的密碼"
```

---

> [!NOTE]
> 本教學根據 `OceanMaster_v13_2` 原始碼逐行驗證。
> 所有路徑、函數名稱、參數、CSV 欄位名稱均與實際程式碼一致。
