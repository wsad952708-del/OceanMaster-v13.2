# OceanMaster v13.2 — 漁場預測系統簡報

---

## Slide 1 — 系統定位

**OceanMaster 是一套整合即時衛星數據、海洋物理模型與機器學習的延繩釣漁場預測系統，
目前覆蓋西太平洋 10 種商業魚種，具備 WCPFC 歷史數據驗證。**

- 輸入：7 個免費公開衛星/海洋數據源 + WCPFC 歷史漁獲
- 輸出：每日 20 個最佳漁場座標（含建議放鉤水深、CPUE 估值、安全評級）
- 覆蓋範圍：5°N–35°N, 120°E–175°E（西北太平洋延繩釣主漁場）

---

## Slide 2 — 技術架構

### 2.1 數據來源

| 數據源 | API / 供應商 | 解析度 | 延遲 | 來源檔案 |
|--------|-------------|--------|------|---------|
| SST（海表溫度） | CMEMS `cmems_mod_glo_phy-thetao` | 0.083° (~9km) | 1 天 | [data_fetcher_v2.py:377](file:///c:/Users/user/Desktop/好像快好了/OceanMaster_v13_2/engine/data_fetcher_v2.py#L377) |
| CHL-a（葉綠素） | NOAA VIIRS (ERDDAP) | 4km | 2-8 天 | [data_fetcher_v2.py:L513](file:///c:/Users/user/Desktop/好像快好了/OceanMaster_v13_2/engine/data_fetcher_v2.py#L513) |
| 3D 溫度剖面 | HYCOM GLBy0.08 (OPeNDAP) | 0.08° × 6 層 | 1-2 天 | [data_fetcher_v2.py:L846](file:///c:/Users/user/Desktop/好像快好了/OceanMaster_v13_2/engine/data_fetcher_v2.py#L846) |
| SSH / 海流 | CMEMS `cmems_mod_glo_phy` | 0.083° | 1 天 | [data_fetcher_v2.py:L1430](file:///c:/Users/user/Desktop/好像快好了/OceanMaster_v13_2/engine/data_fetcher_v2.py#L1430) |
| 溶氧（DO） | CMEMS BGC `cmems_mod_glo_bgc` | 0.25° | 1 天 | [dissolved_oxygen.py:L135](file:///c:/Users/user/Desktop/好像快好了/OceanMaster_v13_2/engine/dissolved_oxygen.py#L135) |
| 水深 | GEBCO / ETOPO1 (ERDDAP) | 15弧秒 (~500m) | 靜態 | [gebco_features.py](file:///c:/Users/user/Desktop/好像快好了/OceanMaster_v13_2/engine/gebco_features.py) |
| 天氣/浪高 | Open-Meteo Marine + Forecast | 0.25° | 即時 | [weather_fetcher.py](file:///c:/Users/user/Desktop/好像快好了/OceanMaster_v13_2/engine/weather_fetcher.py) |
| 漁船活動 | GFW (Global Fishing Watch) | 0.1° | 3 天 | [gfw_data_loader.py](file:///c:/Users/user/Desktop/好像快好了/OceanMaster_v13_2/engine/gfw_data_loader.py) |
| 校準基準 | WCPFC LONGLINE.CSV | 5°×5° | 年度更新 | [calibration.py:L100](file:///c:/Users/user/Desktop/好像快好了/OceanMaster_v13_2/engine/calibration.py#L100) |

> ⚠️ **全部使用免費公開數據源。** 無需付費衛星訂閱。CMEMS 需免費帳號。

### 2.2 核心演算法

| 演算法 | 論文來源 | 實作位置 |
|--------|---------|---------|
| VGPM 初級產力 | Behrenfeld & Falkowski 1997, L&O 42:1-20 | [forage_engine.py:L41-47](file:///c:/Users/user/Desktop/好像快好了/OceanMaster_v13_2/engine/forage_engine.py#L41) |
| Z_eu 真光層深度 | Morel & Berthon 1989 | [forage_engine.py:L84](file:///c:/Users/user/Desktop/好像快好了/OceanMaster_v13_2/engine/forage_engine.py#L84) |
| Iverson 營養轉換 | Iverson 1990, ε=0.04 | [forage_engine.py:L280](file:///c:/Users/user/Desktop/好像快好了/OceanMaster_v13_2/engine/forage_engine.py#L280) |
| SEAPODYM 棲地指數 | Lehodey et al. 2008, MEPS | [greenfish_hsi.py:L27-45](file:///c:/Users/user/Desktop/好像快好了/OceanMaster_v13_2/engine/greenfish_hsi.py#L27) |
| Deutsch Phi 代謝指數 | Deutsch et al. 2015, Science | [species_params.py:L16-23](file:///c:/Users/user/Desktop/好像快好了/OceanMaster_v13_2/engine/species_params.py#L16) |
| Cayula-Cornillon 鋒面偵測 | Cayula & Cornillon 1992, JGR | [algorithms.py:L52](file:///c:/Users/user/Desktop/好像快好了/OceanMaster_v13_2/engine/algorithms.py#L52) |
| FTLE 拉格朗日傳輸 | Shadden et al. 2005, Physica D | [algorithms.py:L178](file:///c:/Users/user/Desktop/好像快好了/OceanMaster_v13_2/engine/algorithms.py#L178) |
| EKE 渦動能 | Standard oceanography | [algorithms.py:L405](file:///c:/Users/user/Desktop/好像快好了/OceanMaster_v13_2/engine/algorithms.py#L405) |
| SEAPODYM-LMTL 微中層 | Lehodey et al. 2010 | [primary_production.py:L213-287](file:///c:/Users/user/Desktop/好像快好了/OceanMaster_v13_2/engine/primary_production.py#L213) |
| DVM 晝夜垂直遷徙 | Schaefer & Fuller 2010 | [fish_behavior_model.py:L12](file:///c:/Users/user/Desktop/好像快好了/OceanMaster_v13_2/engine/fish_behavior_model.py#L12) |

### 2.3 ML 架構

| 層級 | 模型 | 參數 | 來源檔案 |
|------|------|------|---------|
| Base 1 | RandomForest | 200 trees, depth=12 | [stacking_ensemble.py:L658](file:///c:/Users/user/Desktop/好像快好了/OceanMaster_v13_2/engine/ml/stacking_ensemble.py#L658) |
| Base 2 | GradientBoosting | 150 trees, depth=6, lr=0.05 | [stacking_ensemble.py:L662](file:///c:/Users/user/Desktop/好像快好了/OceanMaster_v13_2/engine/ml/stacking_ensemble.py#L662) |
| Base 3 | XGBoost | 200 trees, depth=8, lr=0.05 | [stacking_ensemble.py:L669](file:///c:/Users/user/Desktop/好像快好了/OceanMaster_v13_2/engine/ml/stacking_ensemble.py#L669) |
| Base 4 | LightGBM | 200 trees, depth=8, lr=0.05 | [stacking_ensemble.py:L677](file:///c:/Users/user/Desktop/好像快好了/OceanMaster_v13_2/engine/ml/stacking_ensemble.py#L677) |
| Base 5 | CatBoost | 200 iters, depth=8, lr=0.05 | [stacking_ensemble.py:L688](file:///c:/Users/user/Desktop/好像快好了/OceanMaster_v13_2/engine/ml/stacking_ensemble.py#L688) |
| Base 6 | Ridge | α=1.0 | [stacking_ensemble.py:L694](file:///c:/Users/user/Desktop/好像快好了/OceanMaster_v13_2/engine/ml/stacking_ensemble.py#L694) |
| **Meta** | **RidgeCV** | **α=[0.01, 0.1, 1.0, 10.0]** | [stacking_ensemble.py:L698](file:///c:/Users/user/Desktop/好像快好了/OceanMaster_v13_2/engine/ml/stacking_ensemble.py#L698) |
| CV | KFold(5) + SpatialBlockCV | buffer=50km | [stacking_ensemble.py:L699, L751](file:///c:/Users/user/Desktop/好像快好了/OceanMaster_v13_2/engine/ml/stacking_ensemble.py#L699) |
| 安全 | HMAC-SHA256 模型簽章 | 防 pickle RCE | [stacking_ensemble.py:L1022](file:///c:/Users/user/Desktop/好像快好了/OceanMaster_v13_2/engine/ml/stacking_ensemble.py#L1022) |
| 特徵 | 44 個物理+生物特徵 | SST/CHL/DO/EKE/MLD/Z20/DVM/Phi/... | [stacking_ensemble.py:L350-400](file:///c:/Users/user/Desktop/好像快好了/OceanMaster_v13_2/engine/ml/stacking_ensemble.py#L350) |

---

## Slide 3 — 已驗證的能力

### 驗證方法

- **數據集**：WCPFC LONGLINE.CSV — 156,212 筆紀錄, 1950-2018
- **驗證期**：2010-2018（9 年，現代漁業技術穩定期）
- **方式**：用 SST 氣候態估算 HSI proxy → 與實際 CPUE (catch/1000hooks) 做 Spearman 秩相關
- **腳本**：[validate_hindcast.py](file:///c:/Users/user/Desktop/好像快好了/OceanMaster_v13_2/validate_hindcast.py)
- **報告**：[output/validation_report.md](file:///c:/Users/user/Desktop/好像快好了/OceanMaster_v13_2/output/validation_report.md)

### 結果

| 物種 | 樣本數 | Spearman ρ | p-value | Hit Rate (Top-20% HSI → 高於中位 CPUE) | 評級 |
|------|--------|-----------|---------|---------------------------------------|------|
| 長鰭鮪 (albacore) | 2,195 | **0.5786** | <1e-196 | **86.9%** | ✅ 優 |
| 黃鰭鮪 (yellowfin) | 2,351 | 0.2499 | <1e-34 | 56.7% | 🟡 可 |
| 大目鮪 (bigeye) | 2,365 | -0.0532 | 9.64e-03 | 36.2% | ⚠️ 需改進 |

> [已驗證:validate_hindcast.py, output/validation_report.md:L29-33]

### Quintile 遞增驗證（以長鰭鮪為例）

| HSI 五等分 | 平均 CPUE (mt/1000hooks) |
|-----------|------------------------|
| Q1 (最低 20%) | 1.17 |
| Q2 | 6.22 |
| Q3 | 7.60 |
| Q4 | 12.91 |
| **Q5 (最高 20%)** | **22.38** |

→ HSI 最高區域的 CPUE 是最低的 **19 倍**。[已驗證:validation_report.md:L80-84]

### 工程品質

| 指標 | 數值 | 來源 |
|------|------|------|
| 代碼審查 | 60+ 檔案全審完, 0 🔴 未解決 | 本次審查 |
| 單元測試 | 40/40 passed (2.49s) | [tests/test_science.py](file:///c:/Users/user/Desktop/好像快好了/OceanMaster_v13_2/tests/test_science.py), [tests/test_api.py](file:///c:/Users/user/Desktop/好像快好了/OceanMaster_v13_2/tests/test_api.py) |
| REST API | 6 端點, Swagger 文件 | [api/app.py](file:///c:/Users/user/Desktop/好像快好了/OceanMaster_v13_2/api/app.py) |
| Pipeline 穩定性 | 19 hotspots, exit code 0 | 最近 5 次執行均正常 |

---

## Slide 4 — 已知限制

### 4.1 數據延遲

| 數據源 | 延遲 | 影響 |
|--------|------|------|
| CMEMS SST | ~1 天 | SST 鋒面位置可能偏移 5-15 km |
| VIIRS CHL | 2-8 天 | 短暫藻華事件可能遺漏 |
| HYCOM 3D 溫度 | 1-2 天 | 溫躍層深度變化未即時反映 |
| GFW 漁船活動 | 3 天 | 本週漁船動態看不到 |
| WCPFC 校準 | 年度 | 無法反映年內漁獲結構變化 |

> [已驗證:data_fetcher_v2.py 各 fetch 函數的 time offset 設定]

### 4.2 大目鮪預測弱 (ρ = -0.05)

**原因**：bigeye CPUE 主要由**深層環境**驅動（溫躍層深度 200-400m, 深層 DO），
不是表層 SST。本次 hindcast 使用 SST 氣候態 proxy，無法捕捉深層變異。

**證據**：`commercial_core_v2.py` L322 的 SEAPODYM 權重中，bigeye 的 `w_phi=0.25`
（代謝指數，需要 3D DO 數據）佔權重最重，但 proxy 沒有 DO 資訊。
[已驗證:commercial_core_v2.py:L322, validation_report.md:L32]

**解法**：接入真實 CMEMS 3D DO + GLORYS 溫度後，bigeye 預期改善至 ρ > 0.2。
即時 pipeline 已使用 CMEMS DO，只是 hindcast proxy 未使用。

### 4.3 尚未有漁船實測數據

- 目前驗證全部基於 **WCPFC 公開彙總數據**（5°×5° 解析度）
- **未做過前瞻驗證 (forward test)**：沒有「先預測 → 再看實際漁獲」的紀錄
- ML 模型的 cross-validation R² 是內部指標，不等於實際預測準確度

### 4.4 其他已知問題

| 項目 | 態 | 來源 |
|------|------|------|
| Pipeline 耗時 ~14 分鐘 | 有 cache 後第二次 ~2 秒，但首次仍需等 | [data_fetcher_v2.py:L1508](file:///c:/Users/user/Desktop/好像快好了/OceanMaster_v13_2/engine/data_fetcher_v2.py#L1508) |
| 無 JAXA Himawari API 帳號 | 每小時 SST 功能未啟用 | [data_fetcher_v2.py:L1554](file:///c:/Users/user/Desktop/好像快好了/OceanMaster_v13_2/engine/data_fetcher_v2.py#L1554) |
| GFW API 回 422 | GFW v3 格式參數變更，漁船活動暫時無數據 | pipeline log |
| ETOPO1 SSL timeout | 偶發，fallback 到 GEBCO 正常 | pipeline log |
| 所有物種數據均來自延繩釣 | 圍網/一支釣物種的 CPUE 驗證尚無 | WCPFC CSV 只有 longline |

---

## Slide 5 — 目前缺少什麼

### 5.1 數據層面

| 缺口 | 為什麼重要 | 補充方案 | 估計成本 |
|------|----------|---------|---------|
| **真實漁船日誌** | 唯一能做 forward test 的數據。沒有就不知道預測準不準 | 合作漁船提供每日漁位+漁獲量 | ¥0（由對方提供） |
| **高解析度歷史 CPUE** | WCPFC 只有 5°×5°，太粗。真實漁位是 0.01° | 漁船日誌或 VMS 軌跡 | ¥0（由對方提供） |
| **Himawari 每小時 SST** | 可追蹤日內鋒面移動 | 註冊 JAXA PTREE 免費帳號 | ¥0（免費） |
| **Argo 即時剖面** | 改善深層溫度/DO 估計 | 已有代碼但用氣候態，改接 Argo API | 1 天工作量 |

### 5.2 驗證層面

| 缺口 | 為什麼重要 | 補充方案 | 估計時間 |
|------|----------|---------|---------|
| **前瞻驗證 (Forward Test)** | 商業系統最關鍵指標：「你上次說這裡有魚，真的有嗎？」 | 找 2 艘友軍漁船跑 2-3 個月 | 2-3 個月 |
| **各物種分別驗證** | 目前 bigeye 幾乎無效 | 需要深層環境 + 真實漁獲對比 | 依賴漁船日誌 |
| **圍網/一支釣驗證** | 目前只驗證延繩釣 | 需要對應漁法的漁獲數據 | 依賴數據 |
| **其他海域驗證** | 目前只覆蓋 120°E-175°E | 需要調整 SST/CHL 數據源的經緯度範圍 | 半天（代碼已支援） |

### 5.3 工程層面

| 缺口 | 補充方案 | 估計費用 |
|------|---------|---------|
| Docker 容器化 | 寫 Dockerfile + docker-compose | 半天 |
| LINE Bot / 手機介面 | LINE Messaging API webhook | 2-3 天 |
| 雲端部署 | AWS/GCP VM + HTTPS | 月費 ~USD 50-100 |
| CI/CD 自動測試 | CI/CD Actions + pytest | 半天 |

---

## Slide 6 — 與世界競爭者的客觀比較

### 我們有的 vs 他們有的

| 面向 | OceanMaster | GreenFish (TW 商業) | CATSAT v6 (US) | INCOIS PFZ (India) |
|------|-------------|-------|---------|----------|
| **即時衛星整合** | ✅ 7 數據源 | ✅ 多源 | ✅ SST+CHL | ✅ SST+CHL |
| **3D 溫度剖面** | ✅ HYCOM 6 層 | ✅ [未驗證] | ❌ 2D only | ❌ 2D only |
| **ML 預測** | ✅ Stacking 6 模型 | ✅ ML+漁船日誌 | ❌ 規則 | ❌ 規則 |
| **SEAPODYM 棲地公式** | ✅ 完整實作 | ❌ [未驗證] | 部分 | ❌ |
| **Deutsch Phi 代謝指數** | ✅ 10 種物種 | ❌ [未驗證] | ❌ | ❌ |
| **食物鏈模型** | ✅ VGPM → 餌料 → DVM | ❌ [未驗證] | ✅ AMM | ❌ |
| **REST API** | ✅ 6 端點 + Swagger | ✅ 封閉式 | ❌ | ❌ |
| **代碼公開審查** | ✅ 60+ 檔案審完 | ❌ 封閉 | ❌ 封閉 | 部分公開 |
| **單元測試** | ✅ 40 tests | ❌ [未驗證] | ❌ [未驗證] | ❌ [未驗證] |

### 他們有但我們沒有的

| 面向 | GreenFish | CATSAT | INCOIS PFZ | 我們 |
|------|-----------|--------|------------|------|
| **漁船日誌訓練** | ✅ 數十年 | ✅ | ✅ ICAR 數據 | ❌ 無 |
| **Forward test 實測** | ✅ 長期累積 | ✅ 發表 | ✅ 發表 | ❌ 未做 |
| **使用者基數** | ✅ 數百艘船 | ✅ | ✅ 數千艘船 | ❌ 0 |
| **LINE/APP 前端** | ✅ | N/A | ✅ 手機 | ❌ 未做 |
| **24/7 客服支援** | ✅ | ✅ | ✅ | ❌ |
| **多年運營經驗** | ✅ | ✅ >10 年 | ✅ >15 年 | ❌ 新系統 |

> ⚠️ **GreenFish、CATSAT、INCOIS 的演算法細節均未公開。** 標「未驗證」代表我無法確認他們是否有此功能，並非指他們一定沒有。

---

## Slide 7 — 漁船合作方案

### 船長需要提供什麼

| 項目 | 格式 | 頻率 | 說明 |
|------|------|------|------|
| 每日漁位 | 經緯度 (度.分) | 每天 | 投繩/收繩位置 |
| 每日漁獲 | 魚種 + 尾數 + 重量(kg) | 每天 | 至少分：yellowfin / bigeye / albacore |
| 海況紀錄 | SST (°C)、浪高 (m) | 盡量 | 船上溫度計讀數即可 |

→ **一行 LINE 訊息就能回報**，例如：
`N25°12 E131°45 / yellowfin 8尾 320kg / bigeye 3尾 180kg / SST 26°C`

### 系統能給什麼（只列做得到的）

| 項目 | 說明 | 目前狀態 |
|------|------|---------|
| 每日 top-20 漁場座標 | 含 HSI 分數、建議放鉤水深 | ✅ 已運行 |
| 每週更新預測地圖 | HTML 互動地圖，可離線開啟 | ✅ 已運行 |
| CPUE 信賴區間 | 基於 WCPFC bootstrap | ✅ 已運行 |
| 安全評級 | 浪高/風速/颱風/低壓 | ✅ 已運行 |
| **ML 模型隨漁獲校準** | 越用越準（增量學習） | ✅ 每 30 次自動 fine-tune |

### 預期改善時間表

| 時點 | 預期效果 | 條件 |
|------|---------|------|
| 合作 1 個月 | 系統開始累積此船隊的漁獲模式 | 每日回報 |
| 合作 3 個月 | forward test 數字出爐（hit rate、CPUE 提升比例） | 持續回報 |
| 合作 6 個月 | ML 模型已針對該船隊 fine-tune，預測準確度預期提升 15-30% | [推測，尚未驗證] |

> ⚠️ 6 個月預期改善幅度 (15-30%) 為推測值，基於 incremental_learner.py 的 XGBoost warm-start 機制和 WCPFC CV 結果外推，**尚未經實際漁獲驗證**。

---

## Slide 8 — 升級路徑與成本

| 升級項目 | 具體做法 | 估計費用 | 優先序 |
|---------|---------|---------|--------|
| **Docker 化** | 寫 Dockerfile + docker-compose，一鍵啟動 | 工程師半天 | P2 |
| **LINE Bot** | LINE Messaging API webhook，船長用 LINE 收報告 | 2-3 天開發 | P1（對方要求時） |
| **雲端部署** | AWS EC2 t3.medium + SSL + cron 每日跑 | ~USD 50-100/月 | P2 |
| **手機 PWA** | 餘 HTML 地圖已可在手機開，加 PWA manifest 即可離線用 | 1 天 | P3 |
| **增加海域** | 調整 lat/lon 範圍 + 確認數據源覆蓋 | 半天 | 看需求 |
| **增加魚種** | `species_params.py` 加參數 + 訓練模型 | 1-2 天/種 | 看需求 |
| **Argo 即時** | 改接 Argo API，替代氣候態 | 1 天 | P2 |
| **JAXA Himawari** | 註冊免費帳號，啟用每小時 SST | 半天 | P3 |
| **CI/CD** | CI/CD Actions 自動跑 pytest + lint | 半天 | P2 |
| **合規認證** | 如需出口或投標：ISO 27001 / ISMS 文件 | 視規模而定 | P4 |

### 總估計

| 階段 | 內容 | 費用 |
|------|------|------|
| **現在** | 系統已可運行，帶報告去談合作 | ¥0 |
| **合作初期** | Docker + LINE Bot + 雲端 | ~USD 200 + 3 天工程 |
| **合作 3 個月** | Forward test 報告出爐 | ¥0（數據由合作方提供） |
| **合作 6 個月** | 專屬 ML 模型 + 第二版驗證報告 | ~USD 600 雲端費 + 2 天工程 |

---

*文件生成時間: 2026-02-26 18:53*
*系統版本: OceanMaster v13.2 (commit af7113c)*
*所有數字來源標注於表格 `來源` 欄或以 `[已驗證:檔案:行號]` 標記*
*未經驗證的推測值以 `[推測]` 或 `[未驗證]` 明確標記*
