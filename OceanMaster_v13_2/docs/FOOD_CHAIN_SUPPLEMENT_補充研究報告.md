# OceanMaster 補充研究報告 — 衛星衍生特徵與魚群習性盲點
> 更新：2026-03-03（已驗證版）
> 分類：有學術論文佐證 ✅ / 推理衍生 🔬 / 尚無研究 ❓ / 已否定 ❌

---

## 一、月相效應（Lunar Phase）✅ 有學術論文

### 學術發現

| 魚種 | 最佳月相 | 機制 | 文獻 |
|------|---------|------|------|
| **黃鰭鮪** | **弱月光期** (新月~上弦月) | 夜間黑暗→DSL上升→餌料集中→獵食效率↑ | Poisson et al. 2010 (Reunion); Noguez 2007 (GoM) — 但區域差異大，效果不如其他魚種顯著 |
| **大目鮪** | **弱月光期** ⚠️ 區域差異大 | 深層 DSL 在黑暗中上升→獵物集中 | Poisson et al. 2010 (Reunion→弱月光↑); Jatmiko 2016 (東印度洋→**滿月↑**) — 結果矛盾 |
| **長鰭鮪** | **滿月** | 月光增強夜間延繩釣CPUE | Poisson et al. 2010 (Reunion CPUE 滿月最高) — Ifremer 確認 |
| **劍魚** | **上弦月/下弦月** (弱月光) | 弱光時劍魚淺層活動增加→延繩釣CPUE↑；滿月時劍魚下潛避光或看到釣具 | Poisson et al. 2010 |
| **魷魚** | **新月** (非滿月) ⚠️ 修正 | 趨光性強→人工燈船光源在新月時效果最佳；滿月時自然光與燈船競爭 | 漁業操作經驗；VIIRS 分析支持 |

> **⚠️ 注意**：月相效應高度區域依賴。Reunion Island 與東印度洋的大目鮪結論相反。
> 建議 DL 模型讓算法自己學，不要硬編碼月相規則。

### 機制解釋（已驗證）

```
新月（最黑暗）
  → DSL（深層散射層）夜間上升更淺（已驗證 ✅）
  → 餌料魚隨之聚集在表層
  → 鮪魚/劍魚在表層更容易捕獵
  → 延繩釣 CPUE 可能最高（劍魚、部分鮪魚）

滿月（最明亮）
  → 劍魚可能下潛或偵測到漁具→ CPUE↓
  → 長鰭鮪反而 CPUE↑（Reunion 資料）
  → 魷魚：人工燈船效果被月光稀釋→ CPUE↓
  → 效果因魚種和區域高度不同
```

### 對 DL 的意義

```python
# 現有特徵 lunar_phase 已有，建議新增：
lunar_phase_sin = sin(2π × lunar_day / 29.5)   # 週期編碼
lunar_phase_cos = cos(2π × lunar_day / 29.5)
lunar_illumination = 0~1                         # 月光強度（0=新月）
days_to_new_moon = 0~14                          # 距下次新月天數
new_moon_window = 1 if lunar_day ∈ [0,7] else 0 # 弱月光視窗
```

**重要性：★★★☆☆（中等，但容易加入且幾乎零成本）**

### 主要文獻
- **Poisson, Gaertner, Taquet, Durbec & Bigelow (2010)** — "Effects of lunar cycle and fishing operations on longline-caught pelagic fish" (Aquat. Living Resour.) — **本段核心引用**
- DeBruyn & Meeuwig (2001) — 月相分析的統計方法論（periodic regression）— 方法論參考，非生態發現
- Jatmiko et al. (2016) — 東印度洋大目鮪月相效應（結論與 Poisson 2010 相反）
- Noguez et al. (2007) — 墨西哥灣鮪魚月相分析

---

## 二、溶氧最低層壓縮效應（OMZ Habitat Compression）✅ 重要學術發現

### 這是目前報告中**最大的盲點**

**核心論文**：Stramma, Prince, Schmidtko, Luo, Hoolihan, Visbeck, Wallace, Brandt & Körtzinger (2012) — "Expansion of oxygen minimum zones may reduce available habitat for tropical pelagic fishes" — *Nature Climate Change*, Vol. 2, pp. 33-37

### 論文核心數據（已驗證 ✅）

```
研究區域：熱帶東北大西洋 (0-25°N, 12-30°W)

OMZ 上升速率：≤1 公尺/年（已驗證 ✅）

棲地損失：
  1960-2010 年間損失 5.95 × 10¹³ m³
  = 15% 的可用棲地（已驗證 ✅）

DO 閾值：3.5 mL/L — 熱帶遠洋魚的生存下限（已驗證 ✅）

驗證方式：47 隻藍槍魚的電子標記數據（已驗證 ✅）

額外發現：OMZ 擴張與遠洋掠食者多樣性下降 10-50% 相關
```

### 對漁場預測的意義

| 情況 | 結果 | 對你系統的意義 |
|------|------|-------------|
| OMZ 較淺（<100m）| 鮪魚壓縮在表層 | CPUE ↑↑（魚更容易釣） |
| OMZ 較深（>200m）| 鮪魚可自由垂直移動 | CPUE 正常 |
| OMZ 急速上升 | 魚群聚集在特定區域 | 漁場集中化 |

### 衛星可以衍生 OMZ 邊界 ✅

```
CMEMS PISCES 生態地球化學模型
→ 提供全球 3D 溶氧場（包含 0-5000m 各層）
→ 計算 DO < 3.5 mL/L 的最淺深度 = OMZ 上界
→ 這個深度就是「魚的可用棲地下界」

新特徵：
omz_depth_m = CMEMS DO < 3.5 mL/L 的最淺深度
habitat_thickness = omz_depth_m - MLD（有效棲地厚度）
habitat_compression_idx = 1 / habitat_thickness（壓縮指數）
```

**重要性：★★★★★（非常重要，尤其大目鮪/藍槍魚）**

---

## 三、ENSO 聖嬰/反聖嬰現象 ✅ 有大量學術研究

### ENSO 對鮪魚的影響（SEAPODYM 模型驗證）

**核心文獻**：Lehodey et al. (1997, 2003, 2008) — SEAPODYM 鮪魚棲地模型

| ENSO 狀態 | 西太平洋 | 中太平洋 | 機制 |
|----------|---------|---------|------|
| **聖嬰（El Niño）** | CPUE ↓ 鮪魚東移 | CPUE ↑ 鮪魚增加 | 暖池東擴→鮪魚跟隨 |
| **反聖嬰（La Niña）** | CPUE ↑ 鮪魚回西太 | CPUE ↓ | 冷水回到赤道→湧升增強→食物鏈旺盛 |
| **中性（Neutral）** | 正常分佈 | 正常 | — |

> **額外發現**：聖嬰年大型黃鰭鮪比例增加（WCPFC 報告）；
> 黃鰭鮪 CPUE 在強 La Niña 後數月達到高峰（IATTC 分析）。

### 台灣相關效應 🔬 推理合理但缺直接論文

```
聖嬰時：
  → 黑潮可能增強（推理合理，但具體強度缺乏專門研究）
  → 台灣東部漁場可能北移
  → 需要台灣漁業署 CPUE 資料驗證

反聖嬰時：
  → 西太平洋整體漁場回到傳統位置
  → 台灣周邊漁場活躍度可能增加
```

### 衛星衍生 ENSO 指標

```python
# ENSO 指標（CMEMS / NOAA 即時提供）
nino34_anomaly = Niño3.4 區域 SST 異常值
                 (5°N-5°S, 170°W-120°W 的平均 SST - 氣候值)

enso_phase = "el_nino" if nino34 > 0.5
           = "la_nina" if nino34 < -0.5
           = "neutral" otherwise

# 太平洋十年振盪（PDO）— 更長週期
pdo_index = 北太平洋 SST 主成分（NOAA 提供）
```

**重要性：★★★★☆（大尺度背景場，影響整個漁季）**

---

## 四、盲點補充（已驗證升級）

### 4.1 深層散射層（DSL）夜間垂直遷移 ✅ 有大量學術研究

> ⬆️ 從報告原標記 ✅ 確認維持 — 完全正確

```
DSL = 由燈籠魚(Myctophidae)、磷蝦、頭足類組成的深層生物層
白天：深度 400-800m（逃避掠食者）
夜晚：上升到 50-200m（覓食）

被稱為「地球上最大的生物量遷移」(已驗證 ✅)

對掠食者的影響：
  大目鮪（已驗證 ✅）：
    → 白天頻繁深潛至 ~400m 獵食 DSL
    → 夜間在較淺層捕食上升的燈籠魚+磷蝦
    → 50-60% 飲食來自中深層 twilight zone
    → 燈籠魚是大目鮪幼魚的主要食物
```

### 衍生特徵（已驗證可行）
```python
diel_migration_index = f(太陽角度, 水深, DO剖面)
night_fishing_advantage = 日落後3-5小時最佳
diel_phase = 0(白天) / 1(夜間) / 0.5(晨昏)
```

**重要性：★★★★☆（大目鮪預測必需）**

---

### 4.2 黑潮蛇行與離岸暖水環 ✅ 有學術論文（升級）

> ⬆️ 從原報告 🔬 升級為 ✅ — 有多篇台灣+日本論文支持

```
學術支持：
  - Frontiers in Marine Science：冷渦旋讓黑潮東蛇行（已驗證 ✅）
  - NTOU (台灣海洋大學)：台灣東側黑潮蛇行專題研究
  - Frontiers 2024：劍魚聚集在黑潮延伸反氣旋暖水環周邊
  - FRA Japan（日本水產研究所）：漁民根據黑潮鋒面+暖水環邊緣選漁場

暖水環機制（已驗證 ✅）：
  → 黑潮偶爾脫落獨立的反氣旋渦旋（warm core ring）
  → 暖水環內 SST 較高、混合層穩定
  → 環邊緣有營養鹽湧升→食物鏈成熟
  → 鮪魚和劍魚聚集在環的邊緣

渦旋類型對鮪魚的影響（已驗證 ✅）：
  反氣旋(暖水環) → 鮪魚聚集 ← 深層暖水+高含氧量，減少熱/缺氧壓力
  氣旋(冷水環)   → 鮪魚迴避
```

### 衍生特徵
```python
kuroshio_meander_index = SSH曲率在黑潮路徑上的異常
warm_core_ring_detected = 黑潮外側獨立正SSH渦旋 (SLA > 0.1m, 半徑 > 50km)
warm_core_ring_edge_dist = 距暖水環邊緣距離 (最佳漁場位置)
```

**重要性：★★★★☆（台灣漁場極相關）**

---

### 4.3 SAR 魚群偵測 ❌ 已否定（降級）

> ⬇️ 從原報告 ❓ 降級為 ❌ — 學術搜索未找到實證

```
原報告聲明：
  "大量魚群游過後，魚鱗脂質會讓海面變得更光滑，SAR 可以偵測"

驗證結果：
  ❌ SAR 偵測「油汙」(oil slick) 是成熟技術（VV偏振+dark spot）
  ❌ 但「魚群脂肪痕跡」的偵測在學術文獻中幾乎無實證
  ❌ SAR 主要海洋應用：油汙偵測、漁船偵測、養殖設施監測
  ❌ 直接偵測野生魚群的生物膜痕跡 → 理論上可能但實際中未被證實

SAR 對你的系統的真正用處：
  ✅ 漁船偵測（間接推估漁場位置）→ 類似 GFW
  ✅ 大型油汙偵測（生態災害預警）
  ❌ 直接偵測魚群 → 不可行
```

**重要性：★ → 降級。SAR 不是魚群偵測工具。**

---

### 4.4 水下噪音指標 — 維持原標記（需船上設備）

正確：非衛星能力範圍，不再贅述。

---

### 4.5 颱風過後效應 ✅ 有學術研究（引用修正）

> 機制完全正確，但原引用 "Kimura 2000" 找不到。替換為正確引用。

```
颱風 → 漁場機制（已驗證 ✅）：
  1. 颱風強力攪拌上層海洋
  2. 深層冷水+營養鹽被帶到光照層
  3. SST 急降 + 營養鹽暴增
  4. 1-2 週後 CHL 平均增加 41%（MDPI 2024 統計確認 ✅）
  5. 3-4 週後食物鏈完整成熟
  6. 鮪魚漁場出現

台灣夏秋颱風季節：
  每次颱風 = 潛在的漁場觸發器

影響因子（已驗證）：
  - 颱風強度越強 → 混合越深 → 營養鹽越多
  - 但暖水渦旋可隔離深層冷水 → 部分颱風效果有限
  - 颱風移動速度越慢 → 局部效果越強
```

### 衍生特徵（可行 ✅）
```python
typhoon_days_ago = 最近颱風經過300km範圍內的天數（0=無颱風）
                   # 資料來源：IBTrACS 全球颱風資料庫（免費）
typhoon_intensity = 颱風最大風速（kt）
post_typhoon_bloom = CHL(today) / CHL(typhoon前7天均值)
                     # >2.0 = 颱風誘發藻華
typhoon_mixing_depth = f(typhoon_intensity, translation_speed)
                       # 估算混合深度
```

**重要性：★★★★☆（台灣漁場非常相關）**

### 修正後文獻
- ~~Kimura 2000~~ → **Lin, Liu & Tang (2003)** — "Upper ocean thermal structure and the Western North Pacific category 5 typhoons" *GRL*
- **Zhao, Tang & Wang (2008)** — "Phytoplankton bloom and associated upper ocean conditions in the northwest Pacific after Typhoon" *JGR Oceans*
- **Pan et al. (2024)** — 颱風風場半徑內 CHL 平均增加 41% — *Remote Sensing (MDPI)*

---

## 五、衛星衍生的新特徵建議（已驗證彙整）

| 新特徵 | 計算方式 | 來源 | 學術支持 | 重要性 |
|--------|---------|------|---------|--------|
| `omz_depth_m` | CMEMS DO < 3.5 mL/L 的最淺深度 | CMEMS PISCES | Stramma 2012 ✅ | ★★★★★ |
| `habitat_thickness` | omz_depth - MLD | 計算衍生 | Stramma 2012 ✅ | ★★★★★ |
| `lunar_illumination` | 天文計算 (0=新月, 1=滿月) | 天文公式 | Poisson 2010 ✅ | ★★★☆☆ |
| `new_moon_window` | 距新月 0-7 天 = 1 | 天文計算 | Poisson 2010 ✅ | ★★★☆☆ |
| `nino34_anomaly` | Niño3.4 SST 異常 | CMEMS/NOAA | Lehodey 2008 ✅ | ★★★★☆ |
| `enso_phase` | -1/0/1 | 計算 | SEAPODYM ✅ | ★★★★☆ |
| `kuroshio_meander` | 黑潮路徑 SSH 曲率 | CMEMS SSH | Frontiers 2024 ✅ | ★★★★☆ |
| `warm_core_ring` | SLA > 0.1m 獨立渦旋 | CMEMS SSH | FRA Japan ✅ | ★★★★☆ |
| `typhoon_days_ago` | IBTrACS 颱風後天數 | IBTrACS（免費）| Lin 2003 ✅ | ★★★★☆ |
| `post_typhoon_bloom` | CHL比值 | MODIS + IBTrACS | Pan 2024 ✅ | ★★★☆☆ |
| `diel_phase` | 白天/夜間/晨昏 | 太陽角度 | DSL文獻 ✅ | ★★★☆☆ |
| `sst_anomaly_30d` | SST - 30天移均 | CMEMS | 基礎 | ★★★☆☆ |
| `pdo_index` | 太平洋十年振盪 | NOAA | 週期太長 | ★★☆☆☆ |
| ~~`sar_fish_slick`~~ | ~~SAR 魚群脂肪~~ | ~~Sentinel-1~~ | ❌ 無學術實證 | ⛔ 移除 |

---

## 六、最重要的三個補充建議（已驗證排序）

### 🥇 第一名：OMZ 棲地壓縮指數
**立即加入，尤其對大目鮪預測影響最大**
```python
habitat_thickness = omz_depth_m - mld
# 當 habitat_thickness < 100m → 魚被壓縮 → CPUE↑
# 學術佐證：Stramma et al. 2012 Nature Climate Change ✅
```

### 🥈 第二名：颱風混合效應
**台灣漁業最相關，CHL 平均增加 41%**
```python
typhoon_days_ago = days since last typhoon within 300km
# 颱風後 14-28 天 = 食物鏈成熟 = 潛在漁場
# 學術佐證：Lin 2003; Zhao 2008; Pan 2024 ✅
```

### 🥉 第三名：ENSO 背景場 + 黑潮暖水環
**長週期預測必備 + 台灣漁場直接相關**
```python
nino34_anomaly = current ENSO SST anomaly
warm_core_ring = Kuroshio warm core ring detection
# 學術佐證：Lehodey 2008 SEAPODYM; Frontiers 2024 ✅
```

---

## 七、總結：衛星能衍生的極限（已驗證修正版）

```
✅ 可以衍生（直接或間接）：
  海表面物理量（SST/CHL/SSH/SSS/風/鹽）
  初級生產力（NPP/VGPM）
  食物鏈前兩層（湧升→藻華）
  渦旋/鋒面動力學
  OMZ 深度（CMEMS PISCES 3D 模型）       ← ★★★★★
  ENSO 指數（CMEMS/NOAA）
  月相（天文計算）
  颱風混合效應（IBTrACS + MODIS）         ← ★★★★☆
  黑潮蛇行/暖水環（CMEMS SSH）            ← ★★★★☆
  漁船位置（VIIRS 夜光 + AIS + GFW）
  DSL 間接推估（DO 剖面+太陽角度）

❌ 衛星永遠看不到（需要其他數據）：
  浮游動物密度（只能模型推估）
  餌料魚聚集（只能模型推估）
  魚群實際位置（需要聲納或標記）
  水下噪音/聲學（需要船上設備）
  魚的洄游路徑（需要電子標記）
  魚群脂肪痕跡（SAR 理論上可能，實際無法 ❌）

🔬 未來 5 年可能突破：
  PACE 高光譜浮游動物間接偵測
  AI 從衛星多源融合圖像推估中層食物網狀態
```

---

## 參考文獻（已驗證修正）

### 月相效應
- **Poisson, Gaertner, Taquet, Durbec & Bigelow (2010)** — *Aquat. Living Resour.* — 月相與延繩釣漁獲（Reunion Island）— **核心引用**
- DeBruyn & Meeuwig (2001) — *Marine Ecology Progress Series* — 月相分析統計方法論（periodic regression vs ANOVA）
- Jatmiko et al. (2016) — 東印度洋大目鮪月相效應

### OMZ 棲地壓縮
- **Stramma, Prince et al. (2012)** — *Nature Climate Change* 2:33-37 — OMZ 擴張與鮪魚棲地壓縮 — **核心引用**
- Mislan et al. (2017) — 加州洋流 OMZ 棲地壓縮預測

### ENSO
- **Lehodey et al. (1997, 2003, 2008)** — SEAPODYM 鮪魚棲地模型與 ENSO 效應

### DSL 垂直遷移
- Dagorn et al. — 大目鮪深潛行為與 DSL 獵食
- Oceanographic Magazine — 大型掠食者 50-60% 飲食來自 twilight zone

### 黑潮蛇行
- Frontiers in Marine Science (2024) — 冷渦旋與黑潮蛇行; 劍魚與暖水環
- NTOU 台灣海洋大學 — 台灣東側黑潮動力學

### 颱風效應
- **Lin, Liu & Tang (2003)** — *GRL* — 颱風後上層海洋熱力結構
- **Zhao, Tang & Wang (2008)** — *JGR Oceans* — 颱風後浮游植物爆發
- **Pan et al. (2024)** — *Remote Sensing (MDPI)* — 颱風風場內 CHL 平均增 41%
- ~~Kimura 2000~~ — ❌ 查無此文獻，已移除
