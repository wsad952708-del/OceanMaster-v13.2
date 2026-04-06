# OceanMaster DL 訓練指南

## 系統架構

```
┌─────────────────────────────────────────────────────────┐
│                Training Pipeline                         │
│                                                          │
│  ┌──────────────┐   ┌────────────────┐   ┌───────────┐  │
│  │ 數據來源      │   │ 衛星回查       │   │ DL 模型    │  │
│  │              │   │                │   │           │  │
│  │ ① 真實漁獲   │──→│ CMEMS (L4)     │──→│ XGBoost   │  │
│  │ ② WCPFC公開  │   │ ERDDAP (備援)  │   │ Bi-LSTM   │  │
│  │ ③ 合成數據   │   │ 氣候學 (最後)  │   │ +Attention │  │
│  └──────────────┘   └────────────────┘   └───────────┘  │
│         ↑                                     │          │
│         └──── 增量學習 ←── 精度比較 ────────────┘          │
└─────────────────────────────────────────────────────────┘
```

## 檔案說明

| 檔案 | 功能 |
|------|------|
| `engine/ml/training_data_builder.py` | 三層數據來源 + 食物鏈合成數據 |
| `engine/satellite/historical_fetcher.py` | 衛星歷史回查 (84維特徵) |
| `engine/ml/dl_models_v2.py` | XGBoost + Bi-LSTM+Attention 模型 |
| `engine/ml/training_pipeline.py` | 完整訓練閉環 |

---

## 使用方式

### 模式 1：合成數據訓練（現在可用）

```python
from engine.ml.training_pipeline import TrainingPipeline

pipeline = TrainingPipeline()
results = pipeline.run(min_samples=2000, train_xgb=True, train_lstm=False)
```

不需要任何真實數據，直接跑。合成數據內建食物鏈延遲邏輯。

### 模式 2：加入 WCPFC 公開數據

把 WCPFC CSV 放到 `data/` 目錄：
```
data/
  WCPFC_L_PUBLIC_BY_YY_MM_FLAG.csv    ← 延繩釣
  WCPFC_S_PUBLIC_BY_YY_MM.csv         ← 圍網
```

然後：
```python
pipeline = TrainingPipeline(data_dir="data")
results = pipeline.run()  # 自動偵測 WCPFC
```

### 模式 3：插入真實漁獲日誌（等數據到位）

**步驟 1**：準備 CSV，格式如下：

```csv
date,lat,lon,species,catch_kg,depth
2024-03-15,25.3,131.5,yellowfin,850,150
2024-03-16,24.8,132.1,bigeye,420,280
2024-03-16,25.1,131.8,skipjack,1200,50
```

**步驟 2**：放到指定目錄：
```
data/catch_logs/
  2024_vessel_A.csv
  2024_vessel_B.csv
  ...
```

**步驟 3**：啟動重新訓練：
```python
pipeline = TrainingPipeline()
result = pipeline.retrain_with_new_data()
# → 自動載入 → 衛星回查 → 訓練 → 比較精度
```

> **不需要修改任何程式碼**，放 CSV 然後呼叫即可。

---

## 84 維特徵說明

### 環境基礎 (15維)
SST, CHL, SSH, SSS, MLD, Z20, DO, 風速, 波高, 水深, NPP, PAR, Kd490, SST梯度, SSH梯度

### 鋒面/渦旋 (10維)
鋒面強度, 鋒面距離, 渦旋強度, 渦旋類型, 渦旋年齡, 黑潮距離, U/V海流, 流速切變, 匯聚度

### 食物鏈時序 (12維) ← **核心新增**
CHL_lag3/7/14/21, NPP_lag7/14, SST變化率(3d/7d), 湧升指數, 藻華狀態/天數/強度

### 食物鏈推估 (11維)
藻華類型, PFT矽藻比, PFT微型比, 浮游動物密度, 餌料魚潛力, 食物鏈階段(0-4), 食物鏈ETA, CHL鋒面, 獵物溫度匹配, VIIRS燈船, NPP變化率

### 時空/天文 (11維)
年日, 月份, UTC時, 月相, 太陽高度角, 港口距離, 陸棚距離, EEZ, 海底山距離, GFW漁時, AIS船密度

---

## 模型說明

### Model A — XGBoost（立即可用）
- **最低數據量**：500 筆
- **輸入**：84 維特徵
- **輸出**：漁獲量(kg) + 魚種分類
- **特點**：快速訓練、特徵重要性可解釋
- **精度**：取決於數據質量，合成數據約 RMSE 150kg

### Model B — Bi-LSTM+Attention（主力模型）
- **最低數據量**：500 筆真實時序
- **輸入**：84維 × 30天時間窗口
- **輸出**：魚種 + 漁獲量 + 食物鏈ETA
- **特點**：自動學習最佳食物鏈延遲
- **Physics-informed loss**：違反生態規則自動加大懲罰
- **需要**：PyTorch (`pip install torch`)

---

## 預測輸出解讀

```json
{
  "predictions": [
    {
      "species": "yellowfin",
      "catch_kg_est": 580,
      "eta_days": 5,
      "hsi_score": 0.82
    }
  ]
}
```

| 欄位 | 意義 |
|------|------|
| `species` | 預測最可能的魚種 |
| `catch_kg_est` | 預估漁獲量 (公斤/次作業) |
| `eta_days` | 食物鏈 ETA：預估幾天後魚群到達 |
| `hsi_score` | 棲地適合度指數 (0-1, >0.7 = 好漁場) |

### 建議出航判斷
- HSI > 0.7 + ETA < 3 天 → **建議立即出航**
- HSI > 0.7 + ETA 3-7 天 → **3天後出航**
- HSI < 0.5 → **不建議**

---

## 增量學習機制

每次新數據到位：
1. 系統自動回查衛星歷史特徵
2. 合併新舊數據，重新訓練
3. **比較新舊模型精度**
4. 只有**精度提升才更新**線上模型
5. 合成數據權重自動降低 (0.3 → 真實 1.0)

---

## 食物鏈延遲（DL 核心）

```
Day 0   → 湧升/鋒面出現       ← 衛星看到 SST 下降
Day 3-7 → 浮游植物爆發         ← 衛星看到 CHL 飆高
Day 10  → 浮游動物高峰         ← 模型推估
Day 15  → 餌料魚聚集           ← 模型推估
Day 20  → 鮪魚到達             ← 你要預測的！
```

**CHL_lag14 是最重要的特徵** — 因為 14 天前的 CHL 對應「餌料魚剛聚集」的食物鏈階段。

---

## 常見問題

**Q: 沒有真實數據能訓練嗎？**
A: 可以。合成數據內建了食物鏈延遲和生態參數，XGBoost 直接可跑。但精度有限，真實數據進來後會大幅提升。

**Q: 需要多少真實數據？**
A: XGBoost 500筆可用，Bi-LSTM 需要 500+ 筆有時間連續性的數據。

**Q: 如何判斷模型好不好？**
A: 看 `output/training_report.json` 和 `output/feature_importance_report.md`。特徵重要性排序能告訴你模型學到了什麼。
