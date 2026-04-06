# FAO / WCPFC CPUE 數據取得指南

## 數據來源

### 1. WCPFC（西太平洋漁業委員會）— 推薦

**網址**: https://www.wcpfc.int/statistical-bulletins

- 最相關的數據源（覆蓋台灣遠洋鮪漁作業海域）
- 提供延繩釣（longline）CPUE 數據
- 格式：5°×5° 網格 CSV

**下載步驟**：
1. 進入 https://www.wcpfc.int/statistical-bulletins
2. 選擇 "Public Domain Data"
3. 下載 "Longline catch and effort"
4. 選擇物種：YFT, BET, ALB, SKJ
5. 下載 CSV 格式

### 2. IOTC（印度洋漁業委員會）

**網址**: https://iotc.org/data/datasets

- 印度洋作業數據
- 與 WCPFC 格式類似

### 3. FAO Global Capture Production

**網址**: https://www.fao.org/fishery/statistics-query/en/capture

- 全球漁獲統計
- 較低空間解析度

### 4. IATTC（美洲熱帶鮪魚委員會）

**網址**: https://www.iattc.org/en-US/Data/Public-domain

- 東太平洋數據

---

## CSV 格式轉換

下載的原始數據通常需要轉換為 OceanMaster 格式：

### 目標格式

```csv
year,month,lat,lon,species,catch_mt,effort,cpue
2020,1,-5.0,130.0,YFT,78.2,150000,0.521
2020,1,-5.0,130.0,BET,45.3,150000,0.302
```

### 欄位說明

| 欄位 | 類型 | 說明 |
|------|------|------|
| year | int | 年份 |
| month | int | 月份 (1-12) |
| lat | float | 緯度中心點 (°N, 南半球為負) |
| lon | float | 經度中心點 (°E) |
| species | str | 物種代碼: YFT, BET, ALB, SKJ |
| catch_mt | float | 漁獲量 (公噸) |
| effort | float | 投入努力 (鈎數) |
| cpue | float | 單位努力漁獲量 (MT/1000hooks) |

### 使用系統內建轉換器

```bash
# WCPFC 原始 CSV → OceanMaster 格式
python prepare_real_data.py --input raw_wcpfc.csv --output data/real/fao_cpue.csv
```

---

## 訓練流程

```bash
# 1. 放入數據
cp your_data.csv data/real/fao_cpue.csv

# 2. 重新訓練所有物種
python train_and_validate.py --all --data-path data/real/fao_cpue.csv

# 3. 檢查訓練報告
cat models/training_report_v12.md

# 4. 重啟服務使新模型生效
# Docker: docker-compose restart
# systemd: sudo systemctl restart oceanmaster
```

---

## 注意事項

> ⚠️ WCPFC 公開數據為 5°×5° 網格聚合數據，空間解析度有限。
> 更高解析度的 1°×1° 數據需要向各 RFMO 申請專案存取權限。

> ⚠️ 不同 RFMO 的 CPUE 標準化方式不同，合併多來源數據時需特別注意。
