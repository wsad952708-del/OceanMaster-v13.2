"""
OceanMaster v13.2 — 擴充數據源 (Phase 4)
========================================
[v11 Data-1] CMEMS BGC Fetcher
[v11 Data-2] Argo Float Data Fetcher
[v11 Data-3] ONI Auto-Download
[v11 Data-4] GEBCO Seamount Proximity + Shelf Distance

所有 API 均使用免費端點 — 無需付費訂閱。
"""

import numpy as np
import logging
import json
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple

log = logging.getLogger("OceanMaster.DataSources")


# ═══════════════════════════════════════════════════
#  [v11 Data-1] CMEMS BGC (Copernicus Marine)
# ═══════════════════════════════════════════════════

class CMEMSBGCFetcher:
    """
    CMEMS Biogeochemical 數據擷取器

    數據源: Copernicus Marine Service (免費註冊)
      - 全球海洋生態模型 (GLORYS BGC)
      - 變數: NO3, PO4, Si, Fe, Chl, O2, pH, talk, DOC

    用於增強 NPP 計算和營養鹽限制評估

    免費替代方案: 使用 E.U. Copernicus Marine open data
    """

    BASE_URL = "https://nrt.cmems-du.eu/motu-web/Motu"

    # 全球 BGC 產品
    PRODUCTS = {
        "daily_bgc": {
            "service_id": "GLOBAL_ANALYSISFORECAST_BGC_001_028-TDS",
            "product_id": "cmems_mod_glo_bgc-bio_anfc_0.25deg_P1D-m",
            "variables": ["no3", "po4", "si", "fe", "o2", "chl", "phyc"],
        },
        "monthly_bgc": {
            "service_id": "GLOBAL_ANALYSISFORECAST_BGC_001_028-TDS",
            "product_id": "cmems_mod_glo_bgc-bio_anfc_0.25deg_P1M-m",
            "variables": ["no3", "po4", "o2", "chl"],
        },
    }

    # 營養鹽限制評估 (Liebig 最小定律)
    NUTRIENT_LIMITS = {
        "no3": {"unit": "mmol/m3", "limiting_below": 0.5, "optimal_range": (2.0, 15.0)},
        "po4": {"unit": "mmol/m3", "limiting_below": 0.1, "optimal_range": (0.3, 1.5)},
        "si":  {"unit": "mmol/m3", "limiting_below": 1.0, "optimal_range": (3.0, 20.0)},
        "fe":  {"unit": "nmol/L",  "limiting_below": 0.1, "optimal_range": (0.2, 2.0)},
    }

    def __init__(self, cache_dir: str = "cache/cmems"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def compute_nutrient_limitation_index(
        self,
        no3: Optional[np.ndarray] = None,
        po4: Optional[np.ndarray] = None,
        si: Optional[np.ndarray] = None,
        fe: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        計算營養鹽限制指數 (0-1)

        使用 Liebig 最小定律: 生長受限於最少的必需營養鹽
        NLI = min(f(NO3), f(PO4), f(Si), f(Fe))
        f(x) = x / (x + Km)  (Michaelis-Menten 動力學)

        Returns:
            nli: shape 與輸入相同, 0=嚴重限制, 1=充足
        """
        indices = []

        if no3 is not None:
            km = 0.7  # 半飽和常數 mmol/m3
            indices.append(no3 / (no3 + km))

        if po4 is not None:
            km = 0.15
            indices.append(po4 / (po4 + km))

        if si is not None:
            km = 2.0
            indices.append(si / (si + km))

        if fe is not None:
            km = 0.12  # nmol/L
            indices.append(fe / (fe + km))

        if not indices:
            log.warning("  無營養鹽數據 → NLI 預設 0.7")
            return np.array([0.7])

        # Liebig 最小定律
        nli = np.minimum.reduce(indices)
        return np.clip(nli, 0.0, 1.0).astype(np.float32)

    def estimate_nutrient_regime(
        self,
        lat: float, lon: float, month: int,
    ) -> Dict[str, float]:
        """
        基於氣候態估計營養鹽制度 (無需實時下載)

        使用 WOA2023 氣候態的簡化公式:
          - 赤道上升流區: NO3 高, Fe 低 (HNLC)
          - 亞熱帶環流: NO3 低 (寡營養 oligotrophic)
          - 溫帶/亞極地: NO3 中等, 季節性高

        Returns:
            dict with no3, po4, si, fe, nli, regime
        """
        # 基於緯度的營養鹽氣候態
        abs_lat = abs(lat)

        # NO3 分佈 (mmol/m3, 表層)
        if abs_lat < 10:
            # 赤道上升流
            no3 = 3.0 + 2.0 * (1 - abs(lon - 180) / 180)
        elif abs_lat < 30:
            # 亞熱帶環流 (寡營養)
            no3 = 0.2 + 0.3 * max(0, (abs_lat - 15) / 15)
        else:
            # 溫帶/亞極地
            no3 = 2.0 + 6.0 * max(0, (abs_lat - 30) / 30)
            # 春季 bloom 後下降
            if 3 <= month <= 9 and lat > 0:  # NH spring/summer
                no3 *= 0.3

        # PO4 (Redfield ratio: N:P ≈ 16:1)
        po4 = no3 / 16.0 + 0.05

        # Si (高緯度豐富, 低緯度限制)
        si = max(0.5, no3 * 0.8 + abs_lat * 0.1)

        # Fe (HNLC 區域低, 近陸高)
        fe = 0.3
        # 太平洋赤道 HNLC
        if abs_lat < 10 and 150 < lon < 280:
            fe = 0.08
        # 南大洋 HNLC
        if lat < -45:
            fe = 0.05

        nli = min(
            no3 / (no3 + 0.7),
            po4 / (po4 + 0.15),
            fe / (fe + 0.12),
        )

        # 分類制度
        if nli < 0.3:
            regime = "oligotrophic"
        elif no3 > 3.0 and fe < 0.15:
            regime = "HNLC"
        elif nli > 0.7:
            regime = "eutrophic"
        else:
            regime = "mesotrophic"

        return {
            "no3": round(float(no3), 2),
            "po4": round(float(po4), 3),
            "si": round(float(si), 2),
            "fe": round(float(fe), 3),
            "nli": round(float(nli), 3),
            "regime": regime,
        }


# ═══════════════════════════════════════════════════
#  [v11 Data-2] Argo Float Data Fetcher
# ═══════════════════════════════════════════════════

class ArgoDataFetcher:
    """
    Argo 浮球剖面數據擷取器

    數據源: Argo GDAC (free, public domain)
      - 全球 ~4000 個浮球
      - 溫度/鹽度/壓力剖面 (0-2000m)
      - BGC-Argo: DO, pH, Chl-a, particles

    API: Argovis (免費, 無需 API key)
      https://argovis-api.colorado.edu/argo/
    """

    ARGOVIS_BASE = "https://argovis-api.colorado.edu"

    def __init__(self, cache_dir: str = "cache/argo"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_profiles(
        self,
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
        date_start: str,
        date_end: Optional[str] = None,
        max_profiles: int = 100,
    ) -> list:
        """
        從 Argovis API 擷取 Argo 剖面

        Parameters:
            lat_range: (lat_min, lat_max)
            lon_range: (lon_min, lon_max)
            date_start: ISO date string (YYYY-MM-DD)
            date_end: Optional end date
            max_profiles: 最大剖面數

        Returns:
            list of dicts with temp/sal/pres profiles
        """
        lat_min, lat_max = lat_range
        lon_min, lon_max = lon_range

        if date_end is None:
            date_end = date_start

        url = (
            f"{self.ARGOVIS_BASE}/argo?"
            f"startDate={date_start}T00:00:00Z&"
            f"endDate={date_end}T23:59:59Z&"
            f"polygon=[["
            f"[{lon_min},{lat_min}],[{lon_max},{lat_min}],"
            f"[{lon_max},{lat_max}],[{lon_min},{lat_max}],"
            f"[{lon_min},{lat_min}]"
            f"]]"
        )

        try:
            req = urllib.request.Request(url)
            req.add_header("Accept", "application/json")
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())

            profiles = []
            for item in data[:max_profiles]:
                profile = {
                    "id": item.get("_id", ""),
                    "lat": item.get("geolocation", {}).get("coordinates", [0, 0])[1],
                    "lon": item.get("geolocation", {}).get("coordinates", [0, 0])[0],
                    "date": item.get("timestamp", ""),
                    "temp": item.get("data", [{}])[0].get("temp", []) if item.get("data") else [],
                    "psal": item.get("data", [{}])[0].get("psal", []) if item.get("data") else [],
                    "pres": item.get("data", [{}])[0].get("pres", []) if item.get("data") else [],
                }
                profiles.append(profile)

            log.info(f"  Argo: {len(profiles)} profiles from {date_start}")
            return profiles

        except Exception as e:
            log.warning(f"  Argo fetch failed: {e}")
            return []

    def compute_mld_from_profiles(
        self,
        profiles: list,
        delta_t: float = 0.5,
    ) -> Dict[str, Any]:
        """
        從 Argo 剖面計算混合層深度 (MLD)

        使用 de Boyer Montégut (2004) 定義:
          MLD = 最淺深度 where ΔT > 0.5°C from surface (10m)

        Parameters:
            profiles: from fetch_profiles
            delta_t: 溫度閾值 (°C)

        Returns:
            dict with lats, lons, mld_values
        """
        lats, lons, mlds = [], [], []

        for p in profiles:
            temp = np.array(p.get("temp", []), dtype=float)
            pres = np.array(p.get("pres", []), dtype=float)

            if len(temp) < 5 or len(pres) < 5:
                continue

            # 找 10m 參考溫度
            ref_idx = np.argmin(np.abs(pres - 10.0))
            t_ref = temp[ref_idx] if np.isfinite(temp[ref_idx]) else temp[0]

            # 找 MLD: |T - T_ref| > delta_t
            mld = 200.0  # default
            for i in range(ref_idx + 1, len(temp)):
                if np.isfinite(temp[i]) and abs(temp[i] - t_ref) > delta_t:
                    mld = float(pres[i])
                    break

            lats.append(p["lat"])
            lons.append(p["lon"])
            mlds.append(mld)

        return {
            "lats": np.array(lats),
            "lons": np.array(lons),
            "mld": np.array(mlds),
            "n_profiles": len(mlds),
        }


# ═══════════════════════════════════════════════════
#  [v11 Data-3] ONI Auto-Download
# ═══════════════════════════════════════════════════

class ONIFetcher:
    """
    ENSO ONI 指數自動下載器

    數據源: NOAA CPC (free, public)
      https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt

    ONI = 3-month running mean of SST anomaly in Niño 3.4 region
      El Niño: ONI ≥ +0.5 for 5 consecutive months
      La Niña: ONI ≤ -0.5 for 5 consecutive months
    """

    ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"

    # 各月份到季節代碼的映射
    SEASON_MAP = {
        1: "DJF", 2: "JFM", 3: "FMA", 4: "MAM",
        5: "AMJ", 6: "MJJ", 7: "JJA", 8: "JAS",
        9: "ASO", 10: "SON", 11: "OND", 12: "NDJ",
    }

    def __init__(self, cache_dir: str = "cache/oni"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache = {}

    def fetch_oni(self) -> Dict[str, float]:
        """
        下載並解析 ONI 數據

        Returns:
            dict mapping "YYYY-SEASON" to ONI value
            e.g., {"2024-DJF": 1.2, "2024-JFM": 0.8, ...}
        """
        cache_file = self.cache_dir / "oni_data.json"

        # 檢查快取 (24 小時有效)
        if cache_file.exists():
            import os
            age_hours = (datetime.now().timestamp() - os.path.getmtime(str(cache_file))) / 3600
            if age_hours < 24:
                with open(cache_file, "r") as f:
                    self._cache = json.load(f)
                    log.info(f"  ONI: loaded {len(self._cache)} entries from cache")
                    return self._cache

        try:
            with urllib.request.urlopen(self.ONI_URL, timeout=15) as resp:
                text = resp.read().decode("ascii")

            oni_data = {}
            for line in text.strip().split("\n")[1:]:  # skip header
                parts = line.split()
                if len(parts) >= 4:
                    season = parts[0]
                    year = parts[1]
                    oni_val = float(parts[3])
                    key = f"{year}-{season}"
                    oni_data[key] = oni_val

            # 儲存快取
            with open(cache_file, "w") as f:
                json.dump(oni_data, f)

            self._cache = oni_data
            log.info(f"  ONI: downloaded {len(oni_data)} entries")
            return oni_data

        except Exception as e:
            log.warning(f"  ONI fetch failed: {e}")
            return self._cache or {}

    def get_current_oni(self, date: Optional[datetime] = None) -> float:
        """
        取得指定日期的 ONI 值

        Parameters:
            date: 目標日期 (預設: 今天)

        Returns:
            ONI value (float), 0.0 if not available
        """
        if date is None:
            date = datetime.now()

        if not self._cache:
            self.fetch_oni()

        season = self.SEASON_MAP.get(date.month, "DJF")
        key = f"{date.year}-{season}"
        oni = self._cache.get(key)

        if oni is not None:
            return float(oni)

        # 嘗試上個月的季節
        prev_date = date - timedelta(days=30)
        prev_season = self.SEASON_MAP.get(prev_date.month, "DJF")
        prev_key = f"{prev_date.year}-{prev_season}"
        oni = self._cache.get(prev_key, 0.0)

        return float(oni)

    def classify_enso_state(self, oni: float) -> str:
        """ENSO 狀態分類"""
        if oni >= 1.5:
            return "strong_el_nino"
        elif oni >= 0.5:
            return "el_nino"
        elif oni <= -1.5:
            return "strong_la_nina"
        elif oni <= -0.5:
            return "la_nina"
        else:
            return "neutral"


# ═══════════════════════════════════════════════════
#  [v11 Data-4] GEBCO Seamount Proximity + Shelf Distance
# ═══════════════════════════════════════════════════

class GEBCOProximityCalculator:
    """
    GEBCO 地形特徵計算器

    計算:
      1. 距最近海底山的距離 (seamount proximity)
      2. 距大陸棚邊緣的距離 (shelf break distance)

    科學依據:
      - 海底山: 上升流、聚集效應 → 高生產力 (Morato et al. 2010)
      - 大陸棚: 營養鹽充沛、底魚目標 (Pauly & Zeller 2016)

    數據源: GEBCO 2023 (免費, netCDF 格式)
    替代: 使用公開的海底山資料庫 (Kim & Wessel 2011, Yesson 2011)
    """

    # 已知大型海底山座標 (全球主要海底山 — Yesson et al. 2011 + 手動補充)
    # 僅列出太平洋/印度洋與漁業相關的重要海底山
    KNOWN_SEAMOUNTS = [
        # 西太平洋 (台灣漁場相關)
        (19.5, 140.0, "Mariana Arc"),
        (14.0, 145.0, "Northwest Pacific Seamounts"),
        (28.0, 143.0, "Ogasawara Islands"),
        (9.0, 168.0, "Marshall Islands"),
        (21.5, 132.5, "Kyushu-Palau Ridge"),
        (20.0, 127.0, "Okinawa Trough"),
        # 中太平洋
        (19.5, -155.0, "Hawaii"),
        (12.0, -170.0, "Line Islands"),
        # 印度洋
        (-5.0, 65.0, "Chagos-Maldives Ridge"),
        (-33.0, 58.0, "Southwest Indian Ridge"),
        (-38.0, 77.0, "Southeast Indian Ridge"),
        # 東太平洋
        (5.0, -110.0, "East Pacific Rise"),
        (-10.0, -108.0, "Sala y Gomez"),
        # 南太平洋
        (-20.0, 170.0, "Vanuatu-Fiji"),
        (-15.0, -145.0, "Society Islands"),
    ]

    # 大陸棚等深線近似 (200m 等深線)
    # Format: (lat, lon) 的控制點
    SHELF_BREAK_POINTS = [
        # 東亞大陸棚
        (25.0, 121.5), (27.0, 122.5), (30.0, 128.5),
        (35.0, 132.0), (38.0, 135.0),
        # 南海
        (18.0, 109.5), (15.0, 112.0), (10.0, 113.0),
        # 澳洲
        (-12.0, 131.0), (-15.0, 147.0), (-25.0, 153.5),
        # 紐西蘭
        (-35.0, 174.0), (-40.0, 177.0),
    ]

    def __init__(self):
        self._seamount_latlons = np.array(
            [(s[0], s[1]) for s in self.KNOWN_SEAMOUNTS]
        )
        self._shelf_latlons = np.array(self.SHELF_BREAK_POINTS)

    def compute_seamount_distance(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
    ) -> np.ndarray:
        """
        計算每個網格點到最近海底山的距離 (km)

        Parameters:
            lat: 2D lat grid
            lon: 2D lon grid

        Returns:
            dist_km: 距離 (km), shape 同 lat
        """
        result = np.full_like(lat, 1000.0, dtype=np.float32)

        for i in range(lat.shape[0]):
            for j in range(lat.shape[1]):
                min_dist = 1000.0
                for slat, slon in self._seamount_latlons:
                    dlat = (lat[i, j] - slat) * 111.0
                    dlon = (lon[i, j] - slon) * 111.0 * np.cos(np.radians(lat[i, j]))
                    d = np.sqrt(dlat**2 + dlon**2)
                    if d < min_dist:
                        min_dist = d
                result[i, j] = min_dist

        return result

    def compute_shelf_distance(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
    ) -> np.ndarray:
        """
        計算每個網格點到大陸棚邊緣的距離 (km)

        Parameters:
            lat: 2D lat grid
            lon: 2D lon grid

        Returns:
            dist_km: 距離 (km), shape 同 lat, 正=近海, 負=離岸
        """
        result = np.full_like(lat, 500.0, dtype=np.float32)

        for i in range(lat.shape[0]):
            for j in range(lat.shape[1]):
                min_dist = 500.0
                for slat, slon in self._shelf_latlons:
                    dlat = (lat[i, j] - slat) * 111.0
                    dlon = (lon[i, j] - slon) * 111.0 * np.cos(np.radians(lat[i, j]))
                    d = np.sqrt(dlat**2 + dlon**2)
                    if d < min_dist:
                        min_dist = d
                result[i, j] = min_dist

        return result

    def compute_all_geomorphic_features(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        計算所有地貌特徵

        Returns:
            dict with dist_to_seamount, dist_to_shelf_break,
                  seamount_influence (0-1), shelf_influence (0-1)
        """
        dist_sm = self.compute_seamount_distance(lat, lon)
        dist_shelf = self.compute_shelf_distance(lat, lon)

        # 海底山影響: 距離 < 100km → 高影響
        sm_influence = np.exp(-dist_sm / 80.0)
        # 大陸棚影響: 距離 < 200km → 高影響
        shelf_influence = np.exp(-dist_shelf / 150.0)

        return {
            "dist_to_seamount": dist_sm,
            "dist_to_shelf_break": dist_shelf,
            "seamount_influence": sm_influence.astype(np.float32),
            "shelf_influence": shelf_influence.astype(np.float32),
        }
