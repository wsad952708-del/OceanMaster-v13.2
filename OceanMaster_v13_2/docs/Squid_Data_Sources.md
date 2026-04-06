# 魷魚模組資料來源 (Squid Data Sources)

> OceanMaster v13.2 Phase 12 — 北太赤魷 & 日本魷魚

---

## 1. 北太平洋漁業委員會 (NPFC)

| 項目 | 內容 |
|------|------|
| 全名 | North Pacific Fisheries Commission |
| 網址 | https://www.npfc.int |
| 資料類型 | 魷魚 CPUE、漁獲量統計、漁業管理措施 |
| 覆蓋物種 | *Ommastrephes bartramii* (北太赤魷)、*Todarodes pacificus* (日本魷魚) |
| 解析度 | 1°×1° 月統計 |
| 備註 | NPFC 是北太平洋公海魷魚漁業的主要 RFMO，台灣為會員國 |

### 關鍵資料集
- **NPFC Annual Statistics**: 各會員國魷魚年度漁獲量
- **Scientific Observer Data**: 漁場觀察員紀錄 (需申請)
- **CMM 2022-09**: 魷魚資源養護管理措施

---

## 2. 台灣漁業署 (Fisheries Agency, Taiwan)

| 項目 | 內容 |
|------|------|
| 全名 | 行政院農業部漁業署 |
| 網址 | https://www.fa.gov.tw |
| 資料類型 | 魷釣漁船 VDR 航跡、漁獲回報、遠洋漁業統計 |
| 覆蓋物種 | 北太赤魷、阿根廷魷魚、日本魷魚 |
| 備註 | 台灣為全球前三大魷釣國。漁業署 E-Logbook 為高品質 CPUE 來源 |

### 關鍵資料集
- **遠洋漁業統計年報**: 按漁區、魚種的年度漁獲統計
- **VMS / E-Logbook**: 漁船即時監控與電子漁撈日誌 (限授權研究)
- **魷釣漁業作業規範**: 公海魷釣漁船管理辦法

---

## 3. 聯合國糧農組織 (FAO)

| 項目 | 內容 |
|------|------|
| 全名 | Food and Agriculture Organization of the United Nations |
| 網址 | https://www.fao.org/fishery |
| 資料類型 | 全球漁獲量統計、物種分布 |
| 覆蓋物種 | 全球頭足類 (Cephalopoda) |
| 解析度 | FAO 統計區 × 年度 |

### 關鍵資料集
- **FishStatJ / Global Capture Production**: 全球各國漁獲量 (含魷魚細分類)
- **FIRMS**: 漁業資源監測系統
- **Species Fact Sheet**: *O. bartramii* & *T. pacificus* 生態檔案

---

## 4. 輔助資料來源

| 來源 | 用途 | 網址 |
|------|------|------|
| VIIRS Nightfire | 魷釣漁船燈光偵測 (集魚燈) | https://eogdata.mines.edu/products/vnf/ |
| NOAA Lunar Phase | 月相計算 (新月=最佳漁期) | USNO / Jean Meeus algorithm |
| CMEMS Chl-a | 浮游植物濃度 (魷魚餌料代理) | https://marine.copernicus.eu |
| HYCOM Currents | 黑潮延伸體位置追蹤 | https://www.hycom.org |

---

## 引用格式

```
NPFC. (2024). Annual Report of the North Pacific Fisheries Commission. Vancouver, Canada.
漁業署. (2024). 中華民國遠洋漁業統計年報. 台北: 農業部漁業署.
FAO. (2024). Global Capture Production (FishStatJ). Rome: FAO.
```
