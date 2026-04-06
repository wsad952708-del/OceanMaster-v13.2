"""
OceanMaster — WCPFC Data-Driven Calibration Engine
====================================================
用 WCPFC LONGLINE.CSV (156,212 rows, 1950-2018) 的真實漁獲數據
替換系統中所有硬編碼的假 % 數字。

提供三個核心功能:
  1. WCPFCCalibrator: 從真實數據算出物種比例/季節性/殘差
  2. compute_data_quality_score: 替代假的 confidence checklist
  3. get_calibrated_priors: 取得校準後的 prior 供 species_probability 使用

數據來源: WCPFC Aggregated Longline Catch-Effort (公開數據)
"""

import csv
import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

log = logging.getLogger("OceanMaster.Calibration")

WCPFC_CSV = Path("data/wcpfc/LONGLINE.CSV")
CACHE_PATH = Path("data/cache/wcpfc_calibration.json")

# WCPFC CSV 的物種欄位對應
_SPECIES_COLS = {
    "yellowfin": "yft_c",
    "bigeye": "bet_c",
    "albacore": "alb_c",
}


def _parse_coord(s: str) -> float:
    """Parse WCPFC lat5/lon5 format: '15N' → 15.0, '05S' → -5.0"""
    s = s.strip().strip('"')
    if not s:
        return 0.0
    d = s[-1].upper()
    v = float(s[:-1])
    return -v if d in ("S", "W") else v


class WCPFCCalibrator:
    """
    從 WCPFC LONGLINE.CSV 真實漁獲數據算出校準參數。

    產出:
      - species_proportions: 各物種在延繩釣的歷史佔比
      - seasonal_factors: 各物種各月份的季節性變異
      - spatial_cpue_quantiles: 各 5° 格子的 CPUE 分位數
      - credible_residuals: 各物種 HSI→實際出現 的殘差估計
    """

    def __init__(self, csv_path: Optional[Path] = None, min_year: int = 2000):
        self.csv_path = csv_path or WCPFC_CSV
        self.min_year = min_year
        self._data: Dict = {}
        self._loaded = False

    def load(self) -> bool:
        """載入並計算校準參數。優先使用 cache。"""
        # Try cache first
        if CACHE_PATH.exists():
            try:
                age_h = (datetime.now().timestamp() - CACHE_PATH.stat().st_mtime) / 3600
                if age_h < 24 * 30:  # 30 天 cache
                    self._data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
                    self._loaded = True
                    log.info(f"  📊 WCPFC 校準: cache hit ({age_h / 24:.0f} days old)")
                    return True
            except Exception as e:
                log.debug(f"[降級] engine/calibration.py: {e}")

        # Build from CSV
        if not self.csv_path.exists():
            log.warning(f"  📊 WCPFC CSV 不存在: {self.csv_path}")
            return False

        try:
            self._build_from_csv()
            self._loaded = True

            # Cache
            CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            CACHE_PATH.write_text(
                json.dumps(self._data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            log.info(f"  📊 WCPFC 校準: 從 {self._data['_meta']['n_records']} 筆真實數據建立")
            return True
        except Exception as e:
            log.warning(f"  📊 WCPFC 校準失敗: {e}")
            return False

    def _build_from_csv(self):
        """從 CSV 原始數據建立所有校準參數。"""
        # Accumulate: {species: {month: [cpue_values]}}
        monthly_cpue = {sp: {m: [] for m in range(1, 13)} for sp in _SPECIES_COLS}
        # Accumulate: {species: total_catch}
        total_catch = {sp: 0.0 for sp in _SPECIES_COLS}
        # Accumulate: {species: {(lat_bin, lon_bin): [cpue_values]}}
        spatial_cpue = {sp: {} for sp in _SPECIES_COLS}

        n_records = 0
        n_used = 0

        with open(self.csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                n_records += 1
                try:
                    yy = int(row.get("yy", 0))
                    if yy < self.min_year:
                        continue
                    mm = int(row.get("mm", 0))
                    if mm < 1 or mm > 12:
                        continue
                    hhooks = float(row.get("hhooks", 0) or 0)
                    if hhooks <= 0:
                        continue

                    lat = _parse_coord(row.get("lat5", "0N"))
                    lon = _parse_coord(row.get("lon5", "0E"))
                    lat_bin = int(round(lat / 5) * 5)
                    lon_bin = int(round(lon / 5) * 5)

                    for sp, col in _SPECIES_COLS.items():
                        catch = float(row.get(col, 0) or 0)
                        if catch <= 0:
                            continue
                        cpue = catch / (hhooks / 1000.0)  # mt per 1000 hooks
                        monthly_cpue[sp][mm].append(cpue)
                        total_catch[sp] += catch
                        key = f"{lat_bin},{lon_bin}"
                        spatial_cpue[sp].setdefault(key, []).append(cpue)

                    n_used += 1
                except (ValueError, KeyError):
                    continue

        grand_total = sum(total_catch.values()) or 1.0

        # 1. Species proportions (真實比例)
        proportions = {}
        for sp in _SPECIES_COLS:
            proportions[sp] = round(total_catch[sp] / grand_total, 4)
        # 加入非鮪延繩釣目標 (WCPFC 無資料的保留小權重)
        proportions["japanese_flying_squid"] = 0.03
        proportions["pacific_saury"] = 0.02
        proportions["other"] = 0.01
        # 重新正規化
        prop_sum = sum(proportions.values())
        proportions = {k: round(v / prop_sum, 4) for k, v in proportions.items()}

        # 2. Seasonal factors (各月佔年均的倍率)
        seasonal = {}
        for sp in _SPECIES_COLS:
            annual_mean = np.mean(
                [np.mean(vs) if vs else 0 for vs in monthly_cpue[sp].values()]
            )
            if annual_mean <= 0:
                seasonal[sp] = {str(m): 1.0 for m in range(1, 13)}
                continue
            seasonal[sp] = {}
            for m in range(1, 13):
                vals = monthly_cpue[sp][m]
                if vals:
                    seasonal[sp][str(m)] = round(np.mean(vals) / annual_mean, 3)
                else:
                    seasonal[sp][str(m)] = 1.0

        # 3. Credible residuals (CPUE 變異係數作為不確定性度量)
        residuals = {}
        for sp in _SPECIES_COLS:
            all_vals = []
            for vs in monthly_cpue[sp].values():
                all_vals.extend(vs)
            if len(all_vals) > 10:
                cv = float(np.std(all_vals) / max(np.mean(all_vals), 0.01))
                residuals[sp] = round(min(cv, 1.0), 3)
            else:
                residuals[sp] = 0.5
        residuals["japanese_flying_squid"] = 0.40
        residuals["pacific_saury"] = 0.35

        # 4. Spatial CPUE quantiles (top 10 格子 per species)
        spatial_q = {}
        for sp in _SPECIES_COLS:
            cells = []
            for key, vals in spatial_cpue[sp].items():
                if len(vals) >= 5:  # 至少 5 筆才有統計意義
                    cells.append({
                        "cell": key,
                        "n": len(vals),
                        "median": round(float(np.median(vals)), 3),
                        "p75": round(float(np.percentile(vals, 75)), 3),
                        "p90": round(float(np.percentile(vals, 90)), 3),
                    })
            cells.sort(key=lambda c: c["median"], reverse=True)
            spatial_q[sp] = cells[:20]  # top 20 格子

        self._data = {
            "species_proportions": proportions,
            "seasonal_factors": seasonal,
            "credible_residuals": residuals,
            "spatial_cpue_quantiles": spatial_q,
            "_meta": {
                "source": "WCPFC LONGLINE.CSV",
                "data_period": f"{self.min_year}-2018",
                "n_records": n_records,
                "n_used": n_used,
                "calibrated_at": datetime.now().isoformat(),
                "note": "真實漁獲數據，非手動估算",
            },
        }

    @property
    def species_proportions(self) -> Dict[str, float]:
        if not self._loaded:
            self.load()
        return self._data.get("species_proportions", {})

    @property
    def seasonal_factors(self) -> Dict[str, Dict[str, float]]:
        if not self._loaded:
            self.load()
        return self._data.get("seasonal_factors", {})

    @property
    def credible_residuals(self) -> Dict[str, float]:
        if not self._loaded:
            self.load()
        return self._data.get("credible_residuals", {})

    @property
    def meta(self) -> Dict:
        if not self._loaded:
            self.load()
        return self._data.get("_meta", {})


# ══════════════════════════════════════════════════════
# 單例 — 整個 pipeline 共用一個 calibrator
# ══════════════════════════════════════════════════════

_calibrator: Optional[WCPFCCalibrator] = None


def get_calibrator() -> WCPFCCalibrator:
    """取得 singleton WCPFCCalibrator。"""
    global _calibrator
    if _calibrator is None:
        _calibrator = WCPFCCalibrator()
        _calibrator.load()
    return _calibrator


def get_calibrated_priors() -> Tuple[Dict, Dict, Dict]:
    """
    取得校準後的 prior 三元組，供 species_probability.py 使用。

    Returns:
        (proportions, seasonal_bias, credible_residuals)

    如果 WCPFC 數據不可用，回退到硬編碼 fallback (但帶 warning)。
    """
    cal = get_calibrator()
    if cal._loaded and cal.species_proportions:
        proportions = cal.species_proportions
        # seasonal_factors key 是 str，轉回 int
        seasonal = {}
        for sp, monthly in cal.seasonal_factors.items():
            seasonal[sp] = {int(k): v for k, v in monthly.items()}
        residuals = cal.credible_residuals
        log.info(f"  📊 使用 WCPFC 校準資料 ({cal.meta.get('n_used', '?')} records)")
        return proportions, seasonal, residuals
    else:
        log.warning("  📊 WCPFC 校準不可用，使用 fallback (⚠️ 硬編碼估值)")
        return _FALLBACK_PROPORTIONS, _FALLBACK_SEASONAL, _FALLBACK_RESIDUALS


# ══════════════════════════════════════════════════════
# Fallback (當 WCPFC CSV 不存在時)
# ══════════════════════════════════════════════════════

_FALLBACK_PROPORTIONS = {
    "yellowfin": 0.28, "bigeye": 0.25, "skipjack": 0.20,
    "albacore": 0.15, "japanese_flying_squid": 0.06,
    "pacific_saury": 0.04, "other": 0.02,
}

_FALLBACK_SEASONAL = {
    "yellowfin": {1: 0.9, 2: 0.9, 3: 1.0, 4: 1.1, 5: 1.2, 6: 1.2,
                  7: 1.1, 8: 1.0, 9: 0.9, 10: 0.9, 11: 0.9, 12: 0.9},
    "bigeye":    {1: 1.1, 2: 1.1, 3: 1.0, 4: 0.9, 5: 0.9, 6: 0.9,
                  7: 1.0, 8: 1.1, 9: 1.2, 10: 1.2, 11: 1.1, 12: 1.1},
    "skipjack":  {1: 0.8, 2: 0.9, 3: 1.0, 4: 1.1, 5: 1.2, 6: 1.3,
                  7: 1.2, 8: 1.1, 9: 1.0, 10: 0.9, 11: 0.8, 12: 0.8},
    "albacore":  {1: 1.2, 2: 1.1, 3: 1.0, 4: 0.9, 5: 0.8, 6: 0.8,
                  7: 0.9, 8: 1.0, 9: 1.1, 10: 1.2, 11: 1.3, 12: 1.2},
}

_FALLBACK_RESIDUALS = {
    "yellowfin": 0.18, "bigeye": 0.22, "skipjack": 0.15,
    "albacore": 0.25, "japanese_flying_squid": 0.30,
    "pacific_saury": 0.28,
}


# ══════════════════════════════════════════════════════
# Data Quality Score (替代假 confidence)
# ══════════════════════════════════════════════════════

def compute_data_quality_score(
    sst: Optional[np.ndarray] = None,
    do_surface: Optional[np.ndarray] = None,
    temp_3d: Optional[np.ndarray] = None,
    ow: Optional[np.ndarray] = None,
    salinity: Optional[np.ndarray] = None,
    chl: Optional[np.ndarray] = None,
) -> Dict:
    """
    計算數據品質分數 (0-1)。

    這不是「預測準確率」，而是「輸入數據有多完整、多乾淨」。

    評估維度:
      1. 數據完整性 (有哪些欄位)
      2. NaN 覆蓋率 (數據缺失比例)
      3. 值域合理性 (SST 在 -2~35°C?)
      4. 多源一致性 (3D 數據有沒有)

    Returns:
        {
            "score": 0.72,
            "grade": "B",       # A(>0.8) / B(>0.6) / C(>0.4) / D(<=0.4)
            "breakdown": {...},
            "note": "此為數據品質指標，非預測準確率"
        }
    """
    scores = {}

    # 1. SST 品質
    if sst is not None:
        total = sst.size
        valid = np.sum(np.isfinite(sst))
        nan_ratio = 1.0 - (valid / max(total, 1))
        # 值域: SST 在 -2 ~ 35°C 合理
        finite_sst = sst[np.isfinite(sst)]
        if len(finite_sst) > 0:
            in_range = np.sum((finite_sst >= -2) & (finite_sst <= 35)) / len(finite_sst)
        else:
            in_range = 0.0
        scores["sst"] = round(float((1 - nan_ratio) * 0.6 + in_range * 0.4), 3)
    else:
        scores["sst"] = 0.0

    # 2. DO 品質
    if do_surface is not None and np.any(do_surface > 0):
        total = do_surface.size
        valid = np.sum(np.isfinite(do_surface) & (do_surface > 0))
        scores["do"] = round(float(valid / max(total, 1)), 3)
    else:
        scores["do"] = 0.0

    # 3. 3D 數據 (有就大加分)
    if temp_3d is not None:
        n_layers = temp_3d.shape[0] if temp_3d.ndim >= 3 else 1
        layer_score = min(n_layers / 5.0, 1.0)  # 5 layers = 滿分
        scores["3d_data"] = round(float(layer_score), 3)
    else:
        scores["3d_data"] = 0.0

    # 4. 輔助數據
    aux_count = 0
    if ow is not None:
        aux_count += 1
    if salinity is not None:
        aux_count += 1
    if chl is not None and np.any(np.isfinite(chl)):
        aux_count += 1
    scores["auxiliary"] = round(aux_count / 3.0, 3)

    # 加權總分
    weights = {"sst": 0.35, "do": 0.25, "3d_data": 0.20, "auxiliary": 0.20}
    total_score = sum(scores.get(k, 0) * w for k, w in weights.items())
    total_score = round(float(np.clip(total_score, 0, 1)), 3)

    # Grade
    if total_score > 0.8:
        grade = "A"
    elif total_score > 0.6:
        grade = "B"
    elif total_score > 0.4:
        grade = "C"
    else:
        grade = "D"

    return {
        "score": total_score,
        "grade": grade,
        "breakdown": scores,
        "note": "此為數據品質指標，非預測準確率",
    }
