"""
OceanMaster — OBIS 物種出現驗證器
==================================
用 OBIS (Ocean Biodiversity Information System) 物種出現記錄
驗證 HSI 模型的 AUC-ROC。

學術依據:
  - Phillips et al. (2006) Ecol. Model. 190:231-259 — SDM 驗證方法
  - Elith & Leathwick (2009) Species Distribution Models

數據源:
  - OBIS API: https://api.obis.org/v3/
  - GBIF API: https://api.gbif.org/v1/ (補充)
  - 免費、無需登入 (基本查詢)

使用方式:
  validator = OBISValidator()
  records = await validator.fetch_occurrences("Thunnus albacares",
      lat_range=(10, 30), lon_range=(120, 150))
  auc = validator.compute_auc(hsi_grid, lats, lons, records)
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict
from pathlib import Path

log = logging.getLogger("OceanMaster.OBIS")

# 鮪魚物種名對應
SPECIES_MAP = {
    "skipjack":  "Katsuwonus pelamis",
    "yellowfin": "Thunnus albacares",
    "bigeye":    "Thunnus obesus",
    "albacore":  "Thunnus alalunga",
}


class OBISValidator:
    """
    OBIS 物種出現記錄驗證器

    用途: 驗證 HSI 模型預測是否與物種實際出現位置相符。
    方法: AUC-ROC — 出現點 HSI vs 隨機海洋點 HSI 的分離度。
    """

    OBIS_URL = "https://api.obis.org/v3/occurrence"
    GBIF_URL = "https://api.gbif.org/v1/occurrence/search"

    # GBIF taxon keys for tuna
    GBIF_TAXON_KEYS = {
        "Katsuwonus pelamis": 2285270,
        "Thunnus albacares": 2285274,
        "Thunnus obesus": 2285277,
        "Thunnus alalunga": 2285272,
    }

    def __init__(self, cache_dir: str = "data/obis_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    async def fetch_occurrences(
        self,
        species: str,
        lat_range: Tuple[float, float] = (-60, 60),
        lon_range: Tuple[float, float] = (-180, 180),
        year_range: Tuple[int, int] = (2015, 2025),
        max_records: int = 5000,
    ) -> pd.DataFrame:
        """
        從 OBIS + GBIF 下載物種出現記錄。

        Parameters:
            species: 內部物種名 ("yellowfin") 或學名
            lat_range, lon_range: 區域篩選
            year_range: 時間範圍
            max_records: 最大記錄數

        Returns:
            DataFrame: [lat, lon, date, source, species]
        """
        # 標準化物種名
        sci_name = SPECIES_MAP.get(species, species)

        cache_file = self.cache_dir / f"obis_{species}_{year_range[0]}_{year_range[1]}.csv"
        if cache_file.exists():
            df = pd.read_csv(cache_file)
            log.info(f"  OBIS: loaded {len(df)} records from cache ({species})")
            return df

        all_records = []

        # 1. OBIS API
        obis_records = await self._fetch_obis(
            sci_name, lat_range, lon_range, year_range, max_records // 2
        )
        all_records.extend(obis_records)

        # 2. GBIF API (補充)
        gbif_records = await self._fetch_gbif(
            sci_name, lat_range, lon_range, year_range, max_records // 2
        )
        all_records.extend(gbif_records)

        if not all_records:
            log.warning(f"  OBIS/GBIF: No records found for {species}")
            return pd.DataFrame()

        df = pd.DataFrame(all_records)

        # 去重 (相同位置+日期)
        df = df.drop_duplicates(subset=["lat", "lon", "date"])

        # 快取
        df.to_csv(cache_file, index=False)
        log.info(f"  OBIS/GBIF: {len(df)} unique records for {species}")

        return df

    async def _fetch_obis(
        self, sci_name: str,
        lat_range: Tuple, lon_range: Tuple,
        year_range: Tuple, max_records: int,
    ) -> list:
        """OBIS API 查詢"""
        records = []
        try:
            import httpx

            params = {
                "scientificname": sci_name,
                "geometry": f"POLYGON(({lon_range[0]} {lat_range[0]},"
                            f"{lon_range[1]} {lat_range[0]},"
                            f"{lon_range[1]} {lat_range[1]},"
                            f"{lon_range[0]} {lat_range[1]},"
                            f"{lon_range[0]} {lat_range[0]}))",
                "startdate": f"{year_range[0]}-01-01",
                "enddate": f"{year_range[1]}-12-31",
                "size": min(max_records, 5000),
                "fields": "decimalLatitude,decimalLongitude,eventDate",
            }

            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(self.OBIS_URL, params=params)

            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results", [])
                for r in results:
                    lat = r.get("decimalLatitude")
                    lon = r.get("decimalLongitude")
                    date = r.get("eventDate", "")
                    if lat is not None and lon is not None:
                        records.append({
                            "lat": float(lat),
                            "lon": float(lon),
                            "date": str(date)[:10] if date else "",
                            "source": "OBIS",
                            "species": sci_name,
                        })
                log.info(f"    OBIS: {len(records)} records for {sci_name}")
            else:
                log.debug(f"    OBIS: HTTP {resp.status_code}")
        except Exception as e:
            log.debug(f"    OBIS: {e}")

        return records

    async def _fetch_gbif(
        self, sci_name: str,
        lat_range: Tuple, lon_range: Tuple,
        year_range: Tuple, max_records: int,
    ) -> list:
        """GBIF API 查詢"""
        records = []
        try:
            import httpx

            taxon_key = self.GBIF_TAXON_KEYS.get(sci_name)
            params = {
                "decimalLatitude": f"{lat_range[0]},{lat_range[1]}",
                "decimalLongitude": f"{lon_range[0]},{lon_range[1]}",
                "year": f"{year_range[0]},{year_range[1]}",
                "hasCoordinate": "true",
                "limit": min(max_records, 300),
            }
            if taxon_key:
                params["taxonKey"] = taxon_key
            else:
                params["scientificName"] = sci_name

            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(self.GBIF_URL, params=params)

            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results", [])
                for r in results:
                    lat = r.get("decimalLatitude")
                    lon = r.get("decimalLongitude")
                    date = r.get("eventDate", "")
                    if lat is not None and lon is not None:
                        records.append({
                            "lat": float(lat),
                            "lon": float(lon),
                            "date": str(date)[:10] if date else "",
                            "source": "GBIF",
                            "species": sci_name,
                        })
                log.info(f"    GBIF: {len(records)} records for {sci_name}")
            else:
                log.debug(f"    GBIF: HTTP {resp.status_code}")
        except Exception as e:
            log.debug(f"    GBIF: {e}")

        return records

    # ─── AUC-ROC 驗證 ───

    def compute_auc(
        self,
        hsi_grid: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
        occurrences: pd.DataFrame,
        n_background: int = 5000,
    ) -> Dict:
        """
        計算 HSI 模型的 AUC-ROC。

        原理 (Phillips et al. 2006):
          - 在物種出現點取 HSI 值 (正樣本)
          - 在隨機海洋點取 HSI 值 (負樣本)
          - AUC = 正樣本 HSI 高於負樣本 HSI 的機率

        AUC 判讀:
          - 0.5: 隨機 (模型無預測力)
          - 0.6-0.7: 差
          - 0.7-0.8: 可接受
          - 0.8-0.9: 好
          - >0.9: 極好

        Returns:
            dict with: auc, n_presence, n_background,
                       mean_hsi_presence, mean_hsi_background
        """
        if occurrences.empty or hsi_grid.size == 0:
            return {"auc": 0.5, "n_presence": 0, "n_background": 0,
                    "mean_hsi_presence": 0, "mean_hsi_background": 0}

        ny, nx = hsi_grid.shape

        # 提取出現點的 HSI
        presence_hsi = []
        for _, row in occurrences.iterrows():
            lat_idx = np.argmin(np.abs(lats - row["lat"]))
            lon_idx = np.argmin(np.abs(lons - row["lon"]))
            if 0 <= lat_idx < ny and 0 <= lon_idx < nx:
                val = hsi_grid[lat_idx, lon_idx]
                if np.isfinite(val):
                    presence_hsi.append(val)

        if len(presence_hsi) < 5:
            log.warning(f"  AUC: too few presence points ({len(presence_hsi)})")
            return {"auc": 0.5, "n_presence": len(presence_hsi),
                    "n_background": 0,
                    "mean_hsi_presence": 0, "mean_hsi_background": 0}

        # 隨機背景點的 HSI
        rng = np.random.default_rng(42)
        ocean_mask = np.isfinite(hsi_grid)
        ocean_indices = np.argwhere(ocean_mask)

        if len(ocean_indices) < n_background:
            n_background = len(ocean_indices)

        bg_indices = ocean_indices[
            rng.choice(len(ocean_indices), size=n_background, replace=False)
        ]
        background_hsi = [hsi_grid[j, i] for j, i in bg_indices]

        # 計算 AUC (Wilcoxon-Mann-Whitney 公式)
        presence = np.array(presence_hsi)
        background = np.array(background_hsi)

        # AUC = P(presence_hsi > background_hsi)
        n_p = len(presence)
        n_b = len(background)
        correct = 0
        ties = 0
        for p_val in presence:
            correct += np.sum(background < p_val)
            ties += np.sum(background == p_val)
        auc = (correct + 0.5 * ties) / (n_p * n_b)

        result = {
            "auc": float(auc),
            "n_presence": n_p,
            "n_background": n_b,
            "mean_hsi_presence": float(np.mean(presence)),
            "mean_hsi_background": float(np.mean(background)),
        }

        log.info(f"  AUC validation: {auc:.3f} "
                 f"(presence={n_p}, background={n_b}, "
                 f"HSI_pres={np.mean(presence):.3f}, "
                 f"HSI_bg={np.mean(background):.3f})")

        return result

    # ─── 完整驗證流程 ───

    async def validate_species(
        self,
        species: str,
        hsi_grid: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
    ) -> Dict:
        """
        對單一物種執行完整 OBIS/GBIF 驗證。

        Returns:
            dict with: auc, n_records, species, validation_grade
        """
        df = await self.fetch_occurrences(
            species,
            lat_range=(float(lats.min()), float(lats.max())),
            lon_range=(float(lons.min()), float(lons.max())),
        )

        result = self.compute_auc(hsi_grid, lats, lons, df)
        result["species"] = species
        result["n_records"] = len(df)

        # 分級
        auc = result["auc"]
        if auc >= 0.8:
            result["validation_grade"] = "GOOD"
        elif auc >= 0.7:
            result["validation_grade"] = "ACCEPTABLE"
        elif auc >= 0.6:
            result["validation_grade"] = "POOR"
        else:
            result["validation_grade"] = "FAIL"

        return result
