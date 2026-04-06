# OceanMaster v13.2 — 系統狀態總報告
> 最後更新：2026-03-03 03:45 UTC+8  
> 查閱本文檔即可掌握系統中所有「真實 vs 模擬」資料的完整狀態

---

## 一、即時真實數據（✅ 已連接、正在使用）

| 資料項目 | API 來源 | 更新頻率 | 驗證狀態 |
|---------|----------|---------|---------|
| SST 海表溫度 | CMEMS Copernicus + NOAA OISST ERDDAP | 每 6 小時（Pipeline 背景更新） | ✅ 帳號 `wsad952708@gmail.com` 認證成功 |
| SSH 海面高度 | CMEMS GLORYS forecast | 每 6 小時 | ✅ 3D 溫度場 (35×361×661) 下載成功 |
| 3D 海洋溫度場 | CMEMS `cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m` | 每 6 小時 | ✅ 用於溫躍層/Z20/MLD 計算 |
| 風速 / 氣壓 / 降雨 | Open-Meteo Forecast API | 即時（點地圖時抓取） | ✅ 每次回傳 HTTP 200 |
| 波高 / 波週期 / 波向 | Open-Meteo Marine API | 即時 | ✅ 實測 wave=[0.3, 5.1]m |
| SST 衛星底圖 | NASA GIBS WMTS (MUR SST) | 每天更新（T-1 天） | ✅ 直接在 Leaflet 地圖上顯示 |
| 颱風即時警報 | GDACS GeoJSON API | 每 5 分鐘 | ✅ 正確回傳 0 個颱風（3 月非颱風季） |
| 水深地形 | GEBCO ETOPO via ERDDAP | 靜態 | ✅ 深度範圍 [-10,460m, 2,513m] |
| ENSO 狀態 | ONI 指數 | Pipeline 啟動時 | ⚠️ 使用硬編碼 fallback（ONI=0.20, Neutral） |
| WCPFC 校準 | WCPFC LONGLINE/PURSE_SEINE CSV | 快取（5 天更新） | ✅ 30,954 筆 grid-month 紀錄 |
| Argo 浮標 | IFREMER ERDDAP | Pipeline 啟動時 | ⚠️ 即時資料 400 錯誤，使用 9 組氣候學替代 |
| 漁業市場價格 | 台灣農委會 Open Data | Pipeline 啟動時 | ⚠️ SSL 憑證過期，使用 fallback 基線 |

---

## 二、ML 模型（✅ 已訓練、正在使用）

| 模型 | 類型 | 訓練資料 | 特徵數 | 驗證狀態 |
|------|------|---------|-------|---------|
| `stacking_yellowfin.pkl` | XGBoost Stacking | WCPFC LONGLINE.CSV (156,212 筆) | 44 維 + 12 維雙版本 | ✅ 已載入、SHAP 啟用 |
| `stacking_bigeye.pkl` | XGBoost Stacking | 同上 | 44 + 12 維 | ✅ 已載入 |
| `stacking_skipjack.pkl` | XGBoost Stacking | WCPFC PURSE_SEINE.CSV (30,025 筆) | 44 + 12 維 | ✅ 已載入 |
| `stacking_albacore.pkl` | XGBoost Stacking | WCPFC LONGLINE.CSV | 44 + 12 維 | ✅ 已載入 |
| SHAP TreeExplainer | 模型解釋 | 4 魚種各自的 RandomForestRegressor | 12 特徵 | ✅ 真實 SHAP 值輸出 |

> **重要**：這些模型使用 **WCPFC 公開統計資料（5°×5° 月均粗網格）** 訓練。
> 精度夠用但不如用自有漁獲紀錄訓練的模型。模型每次 Pipeline 啟動時自動載入。

---

## 三、DL 深度學習模型（❌ 架構已建、權重未訓練）

| 模型 | 檔案 | 架構 | 訓練需求 | 訓練狀態 |
|------|------|------|---------|---------|
| ConvLSTM 時序預測 | `engine/ml/dl_data_pipeline.py` + `dl_trainer.py` | 多天 SST/CHL 格網 → 3 天後漁獲預測 | 3-6 個月每日 Pipeline 數據累積（目前 50 筆） | ❌ 未訓練 |
| U-Net 衛星影像分割 | 同上 | 256×256 衛星影像 → 漁場分割圖 | 標記衛星影像（「有魚/無魚」）或 VIIRS 燈船 proxy | ❌ 未訓練 |
| OceanSRGAN 超解析度 | `engine/ocean_srgan.py` | 0.25° SST → 0.028° 高解析 | GPU 訓練 + 配對的低解析/高解析衛星影像 | ❌ 使用 Bicubic+USM fallback |
| MLD 預報模型 | `engine/mld_forecast.py` | CMEMS MLD 預報 | CMEMS 預報資料累積 | ❌ 使用 temp_3d 計算 fallback |
| sklearn MLP 時序 | `engine/temporal_forecast.py` | 模擬 LSTM 的非線性映射 | 時序環境數據累積 | ⚠️ 輕量版已可用 |

### DL 啟動條件
- **LSTM**：Pipeline `IncrementalLearner` 每次跑會自動存資料（目前累計 50 筆）。**需累計 ~500 筆**（約 3 個月，每天 6 次），然後執行 `python dl_trainer.py`
- **CNN/U-Net**：需要人工標記的衛星影像，或用 VIIRS 夜間燈船位置當 proxy label
- **SRGAN**：需配對的低解析/高解析衛星影像對進行 GAN 訓練
- **GPU**：RTX 4060 (6GB VRAM) 足夠跑小型 LSTM（batch=32），CNN 勉強（batch 需降到 8-16）

---

## 四、儀表板模擬資料（⚠️ 使用 Math.random / 寫死值）

### 4.1 競爭船 AIS 追蹤（完全模擬）
- **位置**：`dashboard_v3.html` 第 2040-2083 行
- **標註**：`// Simulated AIS vessel fleet (replace with GFW API when available)`
- **內容**：12-18 艘隨機產生的漁船，座標/航向/速度全部 `Math.random()`
- **接法**：已有 GFW API key（`.env` 的 JWT token），`engine/data_fetcher_v2.py` 第 1399 行已有 GFW fetch 邏輯
- **缺什麼**：前端 `dashboard_v3.html` 的 `updateVessels()` 函數需改為從 `/api/vessels` 或 GFW 直接抓

### 4.2 CPUE 歷史漁場（完全模擬）
- **位置**：`dashboard_v3.html` 第 2862-2930 行
- **標註**：`// Simulated CPUE data from WCPFC LONGLINE.CSV patterns (same month historical)`
- **內容**：隨機座標 + 隨機 CPUE 值 + 隨機魚種比例 + 隨機因子貢獻度
- **接法**：後端 `engine/env_cpue_estimator.py` 已有真實 CPUE 估算，結果在 hotspots JSON 的 `cpue_index` 欄位
- **缺什麼**：前端需改為從 Pipeline 輸出的 `hotspots.json` 讀取歷史 CPUE，或接 WCPFC 公開資料庫

### 4.3 模擬颱風展示（有真 API 時不啟用）
- **位置**：`dashboard_v3.html` 第 1635-1710 行
- **標註**：`// ═══ SIMULATED DEMO TYPHOON WITH FORECAST TRACK ═══`
- **內容**：2 個寫死的假颱風（TD01W、TS02W），標示「模擬資料 — 將由即時 API 替換」
- **狀態**：**此段程式碼只在 GDACS API 失敗時才啟用**，正常情況下不會顯示。GDACS 即時 API 已接通（第 1576-1631 行的 `fetchTyphoons()`）

### 4.4 颱風預報路徑（半模擬）
- **位置**：`dashboard_v3.html` 第 1498-1520 行 `generateForecastTrack()`
- **內容**：基於基本西北太平洋颱風統計（先西北後轉東北）+ 隨機擾動
- **接法**：後端 `engine/typhoon_tracker.py` 有完整的 72h Beta-Advection 預報模型
- **缺什麼**：前端 `fetchTyphoons()` 取得 GDACS 座標後，用前端預報路徑而非後端模型；改為從後端 `/api/typhoons` 讀取

### 4.5 多船協作通訊（完全模擬）
- **位置**：`dashboard_v3.html` 第 2980 行
- **標註**：`// ═══ FEATURE 7: MULTI-VESSEL COLLABORATION (Simulated) ═══`
- **內容**：固定的 3 艘「友船」漁獲回報（寫死在 HTML 裡）
- **缺什麼**：需要實際的船隊通訊系統（衛星通訊 or WebSocket）

### 4.6 潮汐預報（近似計算）
- **位置**：`dashboard_v3.html` 第 3096-3120 行 `updateTideDisplay()`
- **內容**：基於月相週期的簡化潮汐計算，非真實天文潮汐模型
- **接法**：CWA 氣象署 API 有精確潮汐預報（需 API key，`.env` 中已留佔位符）

### 4.7 海流箭頭（參數化近似）
- **位置**：`dashboard_v3.html` 第 3420-3445 行
- **內容**：基於黑潮 SST 梯度的參數化箭頭方向/速度，非真實海流向量場
- **接法**：Pipeline 已從 CMEMS 抓到真實海流場（`u_current`, `v_current`），但前端未讀取

### 4.8 時序播放器的風向（模擬）
- **位置**：`dashboard_v3.html` 第 2670 行
- **內容**：`const windDir = (hourIdx * 5 + 180) % 360; // simulated direction shift`
- **缺什麼**：需從 Open-Meteo 歷史 API 讀取逐時風向

### 4.9 雨滴動畫（視覺模擬）
- **位置**：`dashboard_v3.html` 第 2684-2687 行
- **狀態**：純 CSS 動畫裝飾效果，隨機粒子。**非資料問題，僅為 UI 動效。**

---

## 五、API Key 狀態

| 服務 | Key 位置 | 狀態 |
|------|---------|------|
| CMEMS (Copernicus) | `.env` → `CMEMS_USER` / `CMEMS_PASS` | ✅ 有帳號密碼，認證成功 |
| NASA FIRMS (VIIRS) | `.env` → `FIRMS_API_KEY` | ✅ 有 Key，後端可用 |
| GFW (全球漁船追蹤) | `.env` → `GFW_API_KEY` | ✅ 有 JWT token，後端 `data_fetcher_v2.py` 可用，**前端未接** |
| CWA (中央氣象署) | `.env` → `CWA_API_KEY` | ❌ 未填寫，需去 https://opendata.cwa.gov.tw/ 免費註冊 |
| OceanMaster API | `.env` → `OCEANMASTER_API_KEY` | ✅ 已設定 |
| NASA GIBS WMTS | 不需要 Key | ✅ 直接使用 |
| Open-Meteo | 不需要 Key | ✅ 直接使用 |
| GDACS | 不需要 Key | ✅ 直接使用 |
| NOAA ERDDAP | 不需要 Key | ✅ 直接使用 |

---

## 六、後端 Fallback 機制（降級但不中斷）

Pipeline 設計為 graceful degradation — 任何資料源失敗都有替代方案：

| 資料 | 主要來源 | Fallback | 目前狀態 |
|------|---------|----------|---------|
| SST | CMEMS → NOAA OISST | 氣候學平均 | ✅ 主要來源正常 |
| SSH | CMEMS GLORYS | HYCOM NRT | ✅ 主要來源正常 |
| 波高 | NOAA WW3 ERDDAP | Open-Meteo Marine | ⚠️ WW3 返回 404，使用 Open-Meteo |
| 鹽度 SSS | SMAP v5 Monthly | HYCOM → WOA 氣候學 | ⚠️ SMAP 404，使用 HYCOM fallback |
| SHAP | TreeExplainer (shap) | 靜態權重法 | ✅ TreeExplainer 正常運行 |
| Argo 浮標 | IFREMER ERDDAP 即時 | 9 組氣候學剖面 | ⚠️ API 400，使用氣候學 |
| 漁價 | 台灣 data.moa.gov.tw | 固定基線價格 | ⚠️ SSL 過期，使用 fallback |
| GFW 漁船 | GFW 4Wings API | WCPFC 先驗 + 合成漁場 | ⚠️ Key 有但未傳入 pipeline |
| ONI (ENSO) | NOAA CPC | 硬編碼近月值 | ⚠️ 使用 hardcoded fallback |

---

## 七、自動更新機制

```
┌─────────────────────────────────────────────────────────┐
│                     後端 (web_server.py)                  │
│                                                          │
│  啟動時: 載入 hotspots.json → 記憶體                       │
│  背景: update_data_background()                          │
│         └→ 每 6 小時跑完整 Pipeline                       │
│         └→ 更新 hotspots.json + latest.json              │
│         └→ 記憶體 latest_data 原子更新                    │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│                  前端 (dashboard_v3.html)                 │
│                                                          │
│  pollData()        每 30 秒  → 漁場熱點更新               │
│  fetchTyphoons()   每  5 分  → 颱風警報更新               │
│  updateVessels()   每 30 秒  → 船隻位置更新 (目前模擬)     │
│  checkAlerts()     每 30 秒  → 安全警報                    │
│  tick()            每  1 分  → 時鐘更新                    │
│  Open-Meteo        點擊觸發  → 天氣即時查詢               │
└─────────────────────────────────────────────────────────┘
```

---

## 八、優化路線圖（按優先級排序）

### P0 — 立即可做（有 key、有程式碼）
1. [ ] **GFW 漁船接到前端** — `.env` 已有 JWT，`data_fetcher_v2.py` 有 fetch 邏輯，需在 `updateVessels()` 改為 `/api/vessels`
2. [ ] **CPUE 歷史改為真資料** — Pipeline 的 `cpue_index` 已有真值，前端改讀 GeoJSON 的 properties

### P1 — 需註冊（免費）
3. [ ] **CWA API 接入** — 去 https://opendata.cwa.gov.tw/ 註冊，填入 `.env`，可取得精密潮汐 + 台灣本地天氣預報

### P2 — 需累積資料（時間解決）
4. [ ] **LSTM 時序預測** — 等 Pipeline 累積 500+ 筆（~3 個月），執行 `dl_trainer.py`
5. [ ] **自有漁獲紀錄** — 提供你的漁獲 CSV → 重訓 XGBoost → 精度大幅提升

### P3 — 需額外資源
6. [ ] **CNN 衛星影像辨識** — 需標記衛星影像或取得 VIIRS 燈船位置
7. [ ] **SRGAN 超解析度** — 需配對影像 + GPU 訓練時間
8. [ ] **雲端部署** — 目前 localhost，需 GCP/AWS 部署才能船上使用
