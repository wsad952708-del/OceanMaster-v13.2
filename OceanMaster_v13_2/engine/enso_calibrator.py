"""
OceanMaster v13.2 — ENSO 台灣漁場校準引擎 [v13-enso]
====================================================
根據 ENSO 狀態 (ONI 指數) 調整漁場位移、黑潮強度、SST異常。

科學依據:
  El Niño (ONI > +0.5):
    - 黑潮減弱、路徑東移
    - 台灣東部 SST 偏低 0.5-1.5°C
    - 漁場南移 1-3°
    - 秋刀魚漁場南界北退

  La Niña (ONI < -0.5):
    - 黑潮增強、路徑西移
    - 台灣東部 SST 偏高 0.5-1.0°C
    - 漁場北移 1-2°
    - 鬼頭刀/旗魚季提前

  Neutral (-0.5 ≤ ONI ≤ +0.5):
    - 正常年份，無顯著修正

資料源:
  NOAA ONI: https://origin.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php
"""

import numpy as np
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone

log = logging.getLogger("OceanMaster.ENSO")


# ═══════════════════════════════════════════════════
# ENSO 物種修正係數
# ═══════════════════════════════════════════════════

# ONI > 0 = El Niño, ONI < 0 = La Niña
# modifier 值: 正=增益, 負=衰減
#
# [v15.3-audit] 參數來源標注:
#   ONI ±0.5 門檻 — [📖 NOAA 標準定義]
#   各物種 HSI shift 幅度 — [🔬 內部設定] 經驗性參數，無文獻直接支持此精確數值
#   lat_shift_per_oni — [🔬 內部設定] 基於一般性 ENSO 影響描述，非特定論文數字
ENSO_SPECIES_MODIFIER = {
    # El Niño 時的 HSI 修正 (乘以 ONI 的 sign 和 magnitude)
    "yellowfin": {
        "el_nino_hsi_shift": -0.05,   # 微降 (暖池東移)
        "la_nina_hsi_shift": +0.03,   # 微升
        "lat_shift_per_oni": -0.8,    # El Niño 南移 0.8°/ONI
    },
    "bigeye": {
        "el_nino_hsi_shift": -0.08,   # 溫躍層深化不利
        "la_nina_hsi_shift": +0.05,
        "lat_shift_per_oni": -1.0,
    },
    "skipjack": {
        "el_nino_hsi_shift": -0.03,   # 適應力強
        "la_nina_hsi_shift": +0.02,
        "lat_shift_per_oni": -0.5,
    },
    "albacore": {
        "el_nino_hsi_shift": -0.06,
        "la_nina_hsi_shift": +0.04,
        "lat_shift_per_oni": -1.2,    # 溫帶種受 ENSO 影響大
    },
    "mahi_mahi": {
        "el_nino_hsi_shift": +0.03,   # 暖水種受益
        "la_nina_hsi_shift": -0.02,
        "lat_shift_per_oni": -0.3,
    },
    "blue_marlin": {
        "el_nino_hsi_shift": +0.02,   # 暖水種微受益
        "la_nina_hsi_shift": +0.02,   # La Niña 黑潮強也有利
        "lat_shift_per_oni": -0.4,
    },
    "mackerel_scad": {
        "el_nino_hsi_shift": -0.05,
        "la_nina_hsi_shift": +0.05,   # 上升流增強有利
        "lat_shift_per_oni": -0.6,
    },
    "pacific_saury": {
        "el_nino_hsi_shift": -0.12,   # 冷水種受害最大
        "la_nina_hsi_shift": +0.08,
        "lat_shift_per_oni": -1.5,    # 南界大幅北退
    },
    "neon_flying_squid": {
        "el_nino_hsi_shift": -0.04,
        "la_nina_hsi_shift": +0.03,
        "lat_shift_per_oni": -0.7,
    },
    "japanese_flying_squid": {
        "el_nino_hsi_shift": -0.06,
        "la_nina_hsi_shift": +0.05,
        "lat_shift_per_oni": -0.9,
    },
}

# ONI 歷史月值 (用於無法即時取得時的 fallback)
# [v15.3-audit] ⚠️ 這些是開發時填入的估計值，可能與真實 ONI 不符。
# 系統應優先從 NOAA API 即時取得 ONI，此表僅為最後手段。
# 使用 fallback 值時會記錄警告。
ONI_RECENT = {
    (2025, 1): -0.7, (2025, 2): -0.5, (2025, 3): -0.3,
    (2025, 4): -0.2, (2025, 5):  0.0, (2025, 6):  0.2,
    (2025, 7):  0.3, (2025, 8):  0.4, (2025, 9):  0.3,
    (2025, 10): 0.2, (2025, 11): 0.1, (2025, 12): 0.0,
    (2026, 1):  0.1, (2026, 2):  0.2,
}


class ENSOCalibrator:
    """ENSO 校準器"""

    def __init__(self, oni: Optional[float] = None):
        """
        Parameters
        ----------
        oni : float or None
            Oceanic Niño Index (-3 ~ +3)
            None = 自動查詢最近值
        """
        if oni is not None:
            self.oni = float(oni)
        else:
            self.oni = self._get_recent_oni()

        self.phase = self._classify_phase()
        log.info(f"  🌊 ENSO: ONI={self.oni:.2f} ({self.phase})")

    def _get_recent_oni(self) -> float:
        """取得最近的 ONI 值"""
        now = datetime.now(timezone.utc)
        key = (now.year, now.month)
        if key in ONI_RECENT:
            log.warning(f"  ⚠️ ENSO: using hardcoded ONI fallback for {key} "
                        f"(value={ONI_RECENT[key]}). "
                        f"This may be stale/inaccurate. "
                        f"Prefer real-time NOAA API data.")
            return ONI_RECENT[key]
        # 嘗試上個月
        prev = (now.year, now.month - 1) if now.month > 1 else (now.year - 1, 12)
        if prev in ONI_RECENT:
            log.warning(f"  ⚠️ ENSO: using hardcoded ONI fallback for {prev} "
                        f"(value={ONI_RECENT[prev]}). "
                        f"This may be stale/inaccurate.")
            return ONI_RECENT[prev]
        # 預設中性 — 比用錯誤的舊值更安全
        log.warning("  ⚠️ ENSO: no recent ONI data available, "
                    "defaulting to neutral (0.0). "
                    "ENSO corrections will be disabled.")
        return 0.0

    def _classify_phase(self) -> str:
        """分類 ENSO 相位"""
        if self.oni >= 1.5:
            return "Strong El Niño"
        elif self.oni >= 0.5:
            return "El Niño"
        elif self.oni <= -1.5:
            return "Strong La Niña"
        elif self.oni <= -0.5:
            return "La Niña"
        return "Neutral"

    def get_species_modifier(self, species: str) -> Dict[str, float]:
        """
        取得物種的 ENSO 修正。

        Returns
        -------
        dict with keys:
            hsi_modifier : float — 加到 HSI 上的修正值
            lat_shift : float — 漁場緯度位移 (度, 負=南移)
            kuroshio_factor : float — 黑潮強度修正因子 (1.0=正常)
            sst_anomaly : float — SST 異常 (°C)
        """
        params = ENSO_SPECIES_MODIFIER.get(species, {
            "el_nino_hsi_shift": -0.03,
            "la_nina_hsi_shift": +0.02,
            "lat_shift_per_oni": -0.5,
        })

        if self.oni > 0.5:
            # El Niño
            magnitude = min(self.oni / 2.0, 1.0)  # 正規化到 0-1
            hsi_mod = params["el_nino_hsi_shift"] * magnitude
            sst_anom = -0.5 * magnitude  # 台灣東部偏冷
            kuroshio_f = 1.0 - 0.15 * magnitude  # 黑潮減弱
        elif self.oni < -0.5:
            # La Niña
            magnitude = min(abs(self.oni) / 2.0, 1.0)
            hsi_mod = params["la_nina_hsi_shift"] * magnitude
            sst_anom = +0.3 * magnitude  # 台灣東部偏暖
            kuroshio_f = 1.0 + 0.10 * magnitude  # 黑潮增強
        else:
            # Neutral
            hsi_mod = 0.0
            sst_anom = 0.0
            kuroshio_f = 1.0

        lat_shift = params["lat_shift_per_oni"] * self.oni

        return {
            "hsi_modifier": round(hsi_mod, 4),
            "lat_shift": round(lat_shift, 2),
            "kuroshio_factor": round(kuroshio_f, 3),
            "sst_anomaly": round(sst_anom, 2),
            "oni": self.oni,
            "phase": self.phase,
        }

    def apply_to_hsi(
        self,
        hsi: np.ndarray,
        lats: np.ndarray,
        species: str,
    ) -> np.ndarray:
        """
        將 ENSO 修正應用到 HSI 網格。

        Lat-dependent: 高緯度受 ENSO 影響更大。
        """
        mod = self.get_species_modifier(species)
        if abs(mod["hsi_modifier"]) < 0.001:
            return hsi

        # 緯度依賴: 30°N 以上受影響更大
        if lats.ndim == 1:
            lat_factor = np.clip((lats - 15) / 20, 0.3, 1.5)
            lat_factor = lat_factor[:, np.newaxis]  # broadcast
        else:
            lat_factor = np.clip((lats - 15) / 20, 0.3, 1.5)

        adjustment = mod["hsi_modifier"] * lat_factor
        adjusted = np.clip(hsi + adjustment, 0.0, 1.0).astype(np.float32)

        n_changed = np.sum(np.abs(adjusted - hsi) > 0.01)
        if n_changed > 0:
            log.info(f"  ENSO [{species}]: {mod['phase']}, "
                     f"HSI delta={mod['hsi_modifier']:+.3f}, "
                     f"lat_shift={mod['lat_shift']:+.1f}°, "
                     f"kuroshio×{mod['kuroshio_factor']:.2f}")

        return adjusted

    def summary(self) -> Dict[str, Any]:
        """完整摘要"""
        return {
            "oni": self.oni,
            "phase": self.phase,
            "species_modifiers": {
                sp: self.get_species_modifier(sp)
                for sp in ENSO_SPECIES_MODIFIER
            },
        }

    # ═══════════════════════════════════════════════════
    # [v16.0 B1] PDO + IOD 氣候指標
    # ═══════════════════════════════════════════════════

    @staticmethod
    def compute_pdo_modifier(pdo_index: float) -> Dict[str, float]:
        """
        [v16.0 B1] Pacific Decadal Oscillation modifier.

        PDO > 0 (warm phase): 北太平洋 SST 偏暖 → 鮪魚北移
        PDO < 0 (cool phase): 北太平洋 SST 偏冷 → 漁場南縮

        Ref: Mantua et al. (1997) Bull. AMS 78:1069-1079.
        Recent PDO values: ~0.0 to +0.5 (2025-2026, weak warm phase)
        """
        pdo = np.clip(pdo_index, -3, 3)
        phase = ("warm" if pdo > 0.5 else "cool" if pdo < -0.5 else "neutral")

        # HSI shift: positive PDO → northward shift
        hsi_shift = 0.02 * pdo  # ±0.06 max
        lat_shift = 0.3 * pdo   # ±0.9° max
        sst_shift = 0.2 * pdo   # ±0.6°C

        return {
            "pdo_index": round(pdo, 2),
            "pdo_phase": phase,
            "hsi_shift": round(hsi_shift, 4),
            "lat_shift_deg": round(lat_shift, 2),
            "sst_shift_c": round(sst_shift, 2),
        }

    @staticmethod
    def compute_iod_modifier(iod_index: float) -> Dict[str, float]:
        """
        [v16.0 B1] Indian Ocean Dipole modifier.

        IOD > 0 (positive): 東印度洋冷 → 影響澳洲北部漁場
        IOD < 0 (negative): 東印度洋暖 → 西北太平洋間接影響

        Ref: Saji et al. (1999) Nature 401:360-363.
        Note: IOD primarily affects Indian Ocean, secondary effect on WPO.
        """
        iod = np.clip(iod_index, -2, 2)
        phase = ("positive" if iod > 0.4 else "negative" if iod < -0.4 else "neutral")

        # Weak secondary effect on WPO
        hsi_shift = -0.01 * iod  # positive IOD slightly reduces WPO HSI
        sst_shift = -0.1 * iod

        return {
            "iod_index": round(iod, 2),
            "iod_phase": phase,
            "hsi_shift": round(hsi_shift, 4),
            "sst_shift_c": round(sst_shift, 2),
        }

    def get_multi_index_modifier(
        self,
        species: str,
        pdo_index: float = 0.0,
        iod_index: float = 0.0,
    ) -> Dict[str, Any]:
        """
        [v16.0 B1] Combined ENSO + PDO + IOD modifier.

        Returns a merged dict with all three climate indices' effects.
        """
        enso = self.get_species_modifier(species)
        pdo = self.compute_pdo_modifier(pdo_index)
        iod = self.compute_iod_modifier(iod_index)

        combined_hsi = enso["hsi_modifier"] + pdo["hsi_shift"] + iod["hsi_shift"]
        combined_lat = enso["lat_shift"] + pdo["lat_shift_deg"]

        return {
            "enso": enso,
            "pdo": pdo,
            "iod": iod,
            "combined_hsi_modifier": round(combined_hsi, 4),
            "combined_lat_shift": round(combined_lat, 2),
        }
