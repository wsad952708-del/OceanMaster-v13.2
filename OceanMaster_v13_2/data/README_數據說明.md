# OceanMaster 數據目錄說明

## 資料夾結構

```
data/
├── synthetic/              ← 合成訓練數據（AI 生成，用於測試）
│   ├── sample_yellowfin.csv
│   ├── sample_bigeye.csv
│   ├── sample_albacore.csv
│   └── sample_skipjack.csv
└── real/                   ← 真實數據（放你下載的漁獲數據）
    └── fao_cpue.csv        ← 把真實 CSV 放在這裡
```

## 合成數據 vs 真實數據

### 合成數據（synthetic/）
- 由 AI 根據科學關係生成（44 特徵 + 1 目標變數）
- 用於驗證系統架構正確性
- R² ≈ 0.88–0.91（因為數據從已知公式生成）
- **不代表真正的預測能力**

### 真實數據（real/）
- 從 WCPFC / IOTC / FAO 下載的歷史漁獲記錄
- 預期 R² ≈ 0.35–0.55（真實世界噪音更大）
- **這才是真正有商業價值的訓練數據**

## 如何替換為真實數據

1. 下載 WCPFC CPUE CSV（見 `docs/FAO_WCPFC_Data_Guide.md`）
2. 放到 `data/real/fao_cpue.csv`
3. 執行：
   ```bash
   python train_and_validate.py --all --data-path data/real/fao_cpue.csv
   ```
4. 新模型會自動保存到 `models/`

## CSV 格式要求

真實數據 CSV 需包含以下欄位：

```csv
year,month,lat,lon,species,catch_mt,effort,cpue
2020,1,-5,130,YFT,78.2,150000,0.521
2020,1,-5,130,BET,45.3,150000,0.302
```

物種代碼：
| 代碼 | 物種 |
|------|------|
| YFT | 黃鰭鮪 (Yellowfin) |
| BET | 大目鮪 (Bigeye) |
| ALB | 長鰭鮪 (Albacore) |
| SKJ | 正鰹 (Skipjack) |
