# 🦅 OceanMaster v13.2 — 系統總指揮記憶檔
# CLAUDE.md | 每次啟動自動載入 | 禁止刪除或修改此檔案

---

## ▌MISSION STATEMENT

```
目標：打造台灣最強遠洋 AI 漁場預測系統，超越中國「海鷹 AI」
獵物：黃鰭鮪 / 大目鮪 / 柔魚（西太平洋 WCPFC 漁區）
現況：架構建設期，無真實漁獲日誌（等漁業公司談妥後接入）
策略：架構穩固優先 → 確認問題再修復 → 擴充功能 → 最後接入訓練數據
```

---

## ▌PHASE 0：地毯式全檔案掃描（當前優先任務）

> 本專案由 AI 大量生成，實際功能可能遠比任何舊掃描報告更豐富或更複雜。
> **所有判斷必須基於親自讀取原始碼，禁止引用或相信任何舊掃描報告結論。**

### 掃描鐵律

```
法則 1：親眼讀過原始碼才能做判斷，不得依賴記憶或舊報告
法則 2：發現疑似問題 → 繼續讀關聯檔案 → 確認後才記錄
法則 3：確認有問題 → 記錄進【問題清單】→ 不立即修改
法則 4：確認沒問題 → 標記「舊報告誤判」
法則 5：不確定 → 標記「需交叉比對」→ 讀更多關聯檔案
法則 6：本階段禁止修改任何程式碼，除非明顯 typo/語法錯誤
法則 7：AI 生成代碼常有隱藏功能，請給予充分深度閱讀
```

### 每個檔案的標準輸出格式

```
【檔案】engine/xxx.py
【大小】XX KB / 約 XXX 行
【核心功能】（一句話白話說明）
【關鍵 Class/函數】（逐一列出，說明各自做什麼）
【輸入/輸出】（數據格式、維度）
【完成度】✅完整 / ⚠️部分 / 🔲骨架 / ❓不確定
【依賴關係】（import 了誰、被誰呼叫）
【與舊報告差異】（若有差異特別標注）
【隱藏功能】（舊報告未提及的功能）
【問題記錄】（確認有問題才填，否則填「無」）
```

---

### 掃描優先順序（由核心往外）

#### 🔴 第一層：系統主幹（逐行精讀，不可略過）

```
1. main_v10_3.py          (136KB) ← 9步主管線，必須完全理解每一步
2. config.py              (18KB)  ← 全系統設定中樞
3. web_server.py          (77KB)  ← 27+個 FastAPI 端點完整列出
4. pipeline/data_orchestrator.py  ← 數據調度邏輯
5. pipeline/feature_builder.py    ← 特徵矩陣建構
6. pipeline/prediction_engine.py  ← 推論管線協調
7. pipeline/safety_filter.py      ← 安全規則完整清單
```

#### 🔴 第二層：核心科學引擎（精讀數學公式）

```
8.  engine/commercial_core_v2.py  (32KB) ← 代謝指數Φ + SEAPODYM
9.  engine/algorithms.py          (34KB) ← 13個海洋演算法逐一確認
10. engine/ai_fusion.py           (22KB) ← 40%HSI + 60%ML 融合機制
11. engine/food_chain_predictor.py(20KB) ← 食物鏈級聯（Cushing MMH）
12. engine/lagrangian_advection.py(57KB) ← RK4粒子追蹤
13. engine/data_fetcher_v2.py     (94KB) ← 16個API抓取（最大檔案）
14. engine/ocean_physics.py       (30KB) ← 物理海洋計算
15. engine/kuroshio_engine.py     (18KB) ← 黑潮引擎
```

#### 🔴 第三層：AI 核心（重點深挖，必須徹底）

```
ML 系統（engine/ml/ 全部18個檔案）：
16. stacking_ensemble.py    (78KB) ← 66維特徵工程 + Stacking 主模型
17. dl_models_v2.py         (27KB) ← XGBoost + BiLSTM-Attention
18. training_pipeline.py    (25KB) ← 自動化訓練管線 + RFECV
19. convlstm_predictor.py          ← ConvLSTM SST 時空預測
20. unet_fishing.py                ← U-Net 空間漁場預測
21. transfish.py                   ← Transformer 魚群遷移
22. dl_trainer.py                  ← DL 統一訓練器
23. training_data_builder.py       ← 特徵集建構
24. pinn_loss.py                   ← Physics-Informed Loss（物理公式確認）
25. imitation_learner.py           ← 模仿學習（確認實際完成度）
26. federated_trainer.py           ← 聯邦學習（確認實際完成度）
27. ssl_pretrainer.py              ← 自監督預訓練（確認實際完成度）
28. ocean_srgan.py          (42KB) ← SRGAN 超解析度
29. accuracy_booster.py     (30KB) ← 精度提升器
30. shap_explainer.py       (15KB) ← SHAP 可解釋性
31. training_orchestrator.py(15KB) ← 5階段訓練（RL 部分實際完成多少？）
32. genetic_optimizer.py    (15KB) ← 基因演算法（實際完成多少？）

FM / AI Agent（engine/fm/ 全部5個檔案）：
33. fishing_agent.py  (21KB) ← ReAct Agent（工具調用完成了哪些？）
34. rag_engine.py     (11KB) ← RAG 向量檢索（實際完成了什麼？）
35. llm_finetuner.py  (9KB)  ← LoRA 微調（實際完成了什麼？）
36. agent_bootstrap.py(18KB) ← Agent 啟動器（完整還是骨架？）

多智能體（engine/multi_agent/）：
37. fleet_coordinator.py (7KB) ← 多船隊協調（完整還是骨架？）
```

#### 🟡 第四層：生態與漁業專業模組

```
38. engine/fish_behavior_model.py   ← 魚種行為模型
39. engine/species_params.py        ← 物種生態參數庫（11物種）
40. engine/dvm_model.py             ← 晝夜垂直遷移
41. engine/forage_engine.py         ← 餌料生物引擎
42. engine/micronekton_model.py     ← 微游生物模型
43. engine/npz_model.py             ← NPZ 生態模型
44. engine/lunar_model.py           ← 月相引擎
45. engine/zooplankton_proxy.py     ← 浮游動物代理
46. engine/dissolved_oxygen.py      ← 溶解氧（WOA2023）
47. engine/omz_model.py             ← 氧最小層（OMZ）
48. engine/eddy_detector.py         ← 渦旋偵測
49. engine/ocean_color_fronts.py    ← 海色鋒面
50. engine/typhoon_tracker.py       ← 颱風追蹤
51. engine/typhoon_golden_zone.py   ← 颱風後黃金漁區
52. engine/migration_corridor.py    ← 遷移走廊預測
```

#### 🟡 第五層：船隊操作與輸出

```
53. engine/safety_checker.py       ← 安全規則完整清單
54. engine/route_planner_v2.py     ← A* 航線規劃
55. engine/ais_shadow_fishing.py   (49KB) ← AIS 暗漁偵測
56. engine/kml_generator.py        ← KML 輸出
57. engine/html_map_generator.py   ← HTML 互動地圖
58. engine/cpue_estimator.py       ← CPUE 估計
59. engine/hook_depth.py           ← 鉤深建議
60. engine/fuel_predictor.py       ← 燃油預測
61. engine/roi_calculator.py       ← ROI 計算
```

#### 🟢 第六層：ML 訓練系統（ml_system/）

```
62. oceanmaster_ml_trainer_v2.py  (25KB)
63. backtest_engine.py            (19KB) ← 4策略回測
64. historical_data_collector.py  (19KB) ← Synthetic CPUE 生成
65. integrated_predictor.py       (18KB) ← 4策略融合預測
66. feature_engineering_pipeline.py(11KB) ← 36維特徵（確認與66維差異）
67. deployment_package.py         (13KB) ← ML API 部署
68. run_pipeline.py               (8KB)  ← 訓練入口
```

#### 🟢 第七層：驗證與測試

```
69. scripts/run_L2_validation.py  (20KB) ← 9步端到端真實數據驗證
70. scripts/batch_remaining.py    (9KB)
71. scripts/batch_verify.py       (7KB)
72. scripts/batch_verify_3.py     (6KB)
73. scripts/compare_fronts.py     (2KB)
74. scripts/scan_all_features.py  (2KB)
75. tests/ 全部22個測試檔         ← 逐一確認哪些能跑通
```

#### 🟢 第八層：其餘所有 .py（零遺漏）

```
76. engine/ 下剩餘全部 .py（共 106+ 個）逐一掃描
    重點：是否有任何舊報告從未提及的功能？
77. 根目錄全部 .py（共 22 個）
78. api/ 全部4個檔案
79. scheduler/ 全部2個檔案
```

---

## ▌掃描完成後的標準輸出報告

```
【報告 A】系統真實功能完整清單
  完全基於本次親自掃描，不引用舊報告
  格式：#編號 | 功能名稱 | 所在檔案 | 真實完成度 | 說明

【報告 B】ML/DL/RL/FM/AI Agent 各模組深度評估
  每個模組的真實完成度（0~100%）
  已實作的具體功能（函數級別）
  骨架部分需要什麼才能完成
  與舊掃描報告的差異

【報告 C】確認有問題的清單（僅列親自確認過的）
  格式：問題名稱 | P0/P1/P2 | 哪個檔案哪行 | 影響範圍 | 建議

【報告 D】舊報告誤判清單
  格式：誤判項目 | 實際正確狀況 | 依據

【報告 E】隱藏功能清單（舊報告從未提及的）
  這是最重要的發現，AI 生成代碼常有意外的豐富功能

【報告 F】數據流完整追蹤
  從 API 抓取 → 特徵工程 → 模型推論 → 輸出的完整路徑
  確認是否有斷點或不通的地方

【報告 G】建議行動優先序
  P0：立即要做的（影響系統運行）
  P1：重要但不緊急
  P2：優化項目
  P3：未來擴充
```

---

## ▌系統架構速查（供掃描時對照）

### 常用指令

```bash
python main_v10_3.py                              # 核心推論管線
uvicorn web_server:app --port 8000                # Web 服務
python ml_system/run_pipeline.py --all-species    # ML 訓練
python scripts/run_L2_validation.py               # 9步端到端驗證
python -m pytest tests/ -v                        # 全部測試
```

### 核心公式（掃描時交叉確認）

```python
# 代謝指數 Φ（Deutsch 2015）
Φ = pO₂ / [Pcrit × exp(Eo/kB × (1/Tref - 1/T))]

# AI 融合
final_score = 0.40 × HSI + 0.60 × ML_prediction

# FTLE（RK4 粒子追蹤）
FTLE = ln(√λ_max) / T

# EKE（地轉流）
u_g = -(g/f) × ∂SSH/∂y
EKE = 0.5 × (u_g² + v_g²)

# Stacking Ensemble
base_models = [RandomForest, XGBoost, LightGBM, Ridge]
meta_learner = Ridge(alpha=1.0) with 5-fold CV
```

### 66 維特徵順序（掃描時確認一致性）

```
A(5):  sst, chl, ssh, do, current_speed
B(4):  front_strength, front_direction, chl_front, productivity_front
C(4):  eke, ow_param, eddy_edge, ssh_anomaly
D(3):  phi, phi_viability, metabolic_demand
E(4):  bathy_depth, bathy_slope, dist_seamount, dist_shelf_break
F(5):  month_sin, month_cos, doy_sin, doy_cos, hour_sin
G(6):  sst_lag7, sst_lag15, chl_lag7, chl_lag15, sst_rate_3d, chl_rate_7d
H(4):  lat_norm, lon_norm, dist_coast, dist_hotspot_prev
I(5):  wind_speed, wave_height, pressure_msl, precipitation, typhoon_dist
J(7):  salinity, temp_50m, temp_100m, temp_200m, do_50m, mld, d20_depth
bonus(19): ENSO_oni, lunar_factor, viirs_light, forage_index + 物種特異
```

---

## ▌掃描注意事項（全程有效）

### 禁止誤報為問題的事項

```
✅ synthetic data 訓練      → 刻意的，等真實漁獲日誌
✅ RL/FM/Agent 骨架         → 刻意的，分階段實作
✅ rag_database/ 為空       → 刻意的，待向量化
✅ fm_checkpoints/ 為空     → 刻意的，FM訓練後填入
✅ TODO 標記                → 開發路線圖，不是問題
✅ 多個版本號（v10/v12/v13/v18） → 各模組獨立迭代，正常
✅ 337+ except Exception    → 低優先度，不影響核心功能
```

### 真正需要深入確認的事項

```
❓ dl_models_v2.py pickle.load() → HMAC 驗證是否存在？
❓ .env 是否在 .gitignore        → 明文憑證是否保護？
❓ 特徵維度 36 vs 66 vs 84/94   → 衝突還是各有用途？
❓ DEPRECATED 標記              → 舊代碼是否還被呼叫？
❓ 16個外部API                  → 哪些目前真的可以連線？
❓ 已訓練模型（models/*.pt）     → 能否正常載入推論？
```

---

## ▌Phase 1 之後的路線圖

```
Phase 0（當前）：地毯式掃描 → 產出完整真實報告
     ↓
Phase 1：確認問題清單 → 逐一修復（從P0開始）
     ↓
Phase 2：架構升級
  → 溶氧垂直剖面擴充（5層深度）
  → ENSO/PDO/IOD 氣候指數加入
  → 柔魚15天滑動窗口預測升級
  → RAG engine 啟動（docs/ 21個文件向量化）
  → fishing_agent.py 工具調用完善
     ↓
Phase 3（等漁業公司談妥後）：
  → 接入真實漁獲日誌
  → 重訓所有 ML/DL 模型
  → 啟動 RL 航線規劃真實訓練
     ↓
Phase 4（商用版）：
  → 聯邦學習（多船隊隱私訓練）
  → 邊緣部署（船上離線版）
  → LLM LoRA 微調（漁業專用語言模型）
```

---

## ▌重要學術依據

```
Deutsch 2015 (Nature)         → 代謝指數 Φ
Lehodey 2008 (PLOS ONE)       → SEAPODYM 棲息地
Cayula & Cornillon 1992       → SST 鋒面雙峰法
Belkin & O'Reilly 2009        → BOA 葉綠素鋒面
Okubo 1970 + Weiss 1991       → 渦旋分類 W=Sn²+Ss²-ω²
Brill & Bushnell 2001         → 黃鰭鮪 P50 溶氧耐受
Seibel 2007                   → 魷魚代謝指數參數
Rosa & Seibel 2008            → 強壯魷 OMZ forager
Cushing 1990 (MMH)            → 食物鏈 42天延遲
Hobday 2016                   → 海洋熱浪偵測標準
```

---

## ▌角色呼叫快速指令

```
【全檔案掃描】
"請依照 CLAUDE.md 的掃描順序，對 [目標檔案] 進行深度掃描，
 輸出標準格式報告，本階段只分析不修改"

【科學公式驗證】
"請扮演物理海洋學博士，驗證 [目標檔案] 的 [公式]
 是否符合 [引用論文] 的學術標準"

【ML/AI 模型深挖】
"請扮演頂級ML工程師，對 engine/ml/ 下所有模型進行深度掃描，
 特別注意實際完成度，不可依賴舊報告"

【安全問題確認】
"請扮演資安工程師，親自讀取 [目標檔案] 第 [X] 行，
 確認問題是否真實存在，給出原始碼證據"

【代碼修復】（Phase 1 才使用）
"請扮演資深Python工程師，已確認 [問題] 位於 [檔案:行號]，
 請修復並說明修改原因"
```

---

*CLAUDE.md 版本：2026-03-21 v1.0*
*專案：OceanMaster v13.2（魚鷹核心）*
*當前階段：Phase 0 — 地毯式全檔案掃描*
*下次更新：Phase 0 掃描完成後更新報告結果*
