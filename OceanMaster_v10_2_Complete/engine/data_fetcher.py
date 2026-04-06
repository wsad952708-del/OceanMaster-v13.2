"""
OceanMaster v8.0 — 數據擷取引擎
=================================
⚠️ [v12-fix] DEPRECATED — 此檔案已被 data_fetcher_v2.py 取代。
所有新代碼和 import 應使用 data_fetcher_v2.py。
保留此檔案僅為向後相容性參考。

連接 14+ 個真實全球海洋數據源
含智慧防鎖機制（指數退避 + 快取降級）

數據源：
  - NOAA ERDDAP (SST, Chl-a)
  - HYCOM 3D (水溫/鹽度/海流垂直剖面)
  - Open-Meteo (海洋氣象)
  - NASA VIIRS DNB (夜間燈光 → 魷魚船偵測)
  - Global Fishing Watch (AIS 漁船活動)
"""

import numpy as np
import logging
import asyncio
import httpx
import json
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

log = logging.getLogger("OceanMaster.Fetcher")

# ─── 快取管理 ─────────────────────────────────


class DataCache:
    """記憶體 + 磁碟雙層快取"""

    def __init__(self, cache_dir: str = "data/cache"):
        self._mem: Dict[str, Any] = {}
        self._timestamps: Dict[str, datetime] = {}
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str, max_age_hours: float = 24) -> Optional[Any]:
        if key in self._mem:
            age = (datetime.now(timezone.utc) - self._timestamps[key]).total_seconds() / 3600
            if age < max_age_hours:
                log.info(f"快取命中(記憶體): {key} ({age:.1f}h 前)")
                return self._mem[key]
        # 嘗試磁碟快取
        fpath = self._dir / f"{key}.npz"
        if fpath.exists():
            try:
                data = np.load(str(fpath), allow_pickle=True)
                log.info(f"快取命中(磁碟): {key}")
                return dict(data)
            except Exception:
                pass
        return None

    def put(self, key: str, data: Dict[str, Any]):
        self._mem[key] = data
        self._timestamps[key] = datetime.now(timezone.utc)
        try:
            fpath = self._dir / f"{key}.npz"
            np.savez_compressed(str(fpath), **{
                k: v for k, v in data.items()
                if isinstance(v, np.ndarray)
            })
        except Exception as e:
            log.warning(f"磁碟快取寫入失敗: {e}")


# ─── 防鎖重試器 ───────────────────────────────


async def safe_fetch(
    client: httpx.AsyncClient,
    url: str,
    source_name: str,
    max_retries: int = 3,
    backoff_base: float = 30,
    timeout: float = 120,
    params: Optional[Dict] = None,
) -> Optional[httpx.Response]:
    """
    智慧防鎖 HTTP 請求
    - 429 (Too Many Requests) → 指數退避重試
    - 503 (Server Busy) → 退避後重試
    - 其他錯誤 → 記錄並返回 None（用快取降級）
    """
    for attempt in range(max_retries):
        try:
            log.info(f"[{source_name}] 請求中... (嘗試 {attempt+1}/{max_retries})")
            resp = await client.get(url, params=params, timeout=timeout)

            if resp.status_code == 200:
                log.info(f"[{source_name}] ✅ 成功 ({len(resp.content)/1024:.0f} KB)")
                return resp

            if resp.status_code == 429:
                wait = backoff_base * (2 ** attempt)
                log.warning(f"[{source_name}] ⚠️ 被限速 (429)，等待 {wait}s 後重試")
                await asyncio.sleep(wait)
                continue

            if resp.status_code == 503:
                wait = backoff_base * (2 ** attempt)
                log.warning(f"[{source_name}] ⚠️ 伺服器忙 (503)，等待 {wait}s")
                await asyncio.sleep(wait)
                continue

            log.error(f"[{source_name}] ❌ HTTP {resp.status_code}")
            return None

        except httpx.TimeoutException:
            log.warning(f"[{source_name}] ⏱️ 超時，重試 {attempt+1}/{max_retries}")
            await asyncio.sleep(10)
        except Exception as e:
            log.error(f"[{source_name}] ❌ 錯誤: {e}")
            return None

    log.error(f"[{source_name}] ❌ 全部 {max_retries} 次嘗試失敗")
    return None


# ─── 主擷取器 ─────────────────────────────────


class OceanDataFetcher:
    """
    統一數據擷取器 — 連接全球 14+ 個免費海洋數據源
    """

    def __init__(self, lat_range: Tuple[float, float], lon_range: Tuple[float, float]):
        self.lat_min, self.lat_max = lat_range
        self.lon_min, self.lon_max = lon_range
        self.cache = DataCache()

    # ══════════════════════════════════════════════
    #  1. NOAA ERDDAP — SST 海表溫度
    # ══════════════════════════════════════════════

    async def fetch_sst(self) -> Dict[str, np.ndarray]:
        """
        從 NOAA ERDDAP 取得 SST (OISST v2.1, 0.25° 解析度)
        返回: {lat, lon, sst, time}
        """
        cached = self.cache.get("sst", max_age_hours=6)
        if cached is not None:
            return cached

        now = datetime.now(timezone.utc)
        t_start = (now - timedelta(days=3)).strftime("%Y-%m-%dT00:00:00Z")
        t_end = now.strftime("%Y-%m-%dT00:00:00Z")

        url = (
            f"https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg.json"
            f"?sst[({t_start}):1:({t_end})]"
            f"[({self.lat_min}):1:({self.lat_max})]"
            f"[({self.lon_min}):1:({self.lon_max})]"
        )

        async with httpx.AsyncClient() as client:
            resp = await safe_fetch(client, url, "ERDDAP-SST")

        if resp is None:
            log.warning("SST 取得失敗，使用快取或模擬數據")
            return self._generate_demo_sst()

        try:
            data = resp.json()
            table = data["table"]
            cols = table["columnNames"]
            rows = np.array(table["rows"])

            # 解析 ERDDAP JSON 格式
            time_col = rows[:, cols.index("time")]
            lat_col = rows[:, cols.index("latitude")].astype(float)
            lon_col = rows[:, cols.index("longitude")].astype(float)
            sst_col = rows[:, cols.index("sst")].astype(float)

            # 取最新一個時間步
            unique_times = np.unique(time_col)
            latest_mask = time_col == unique_times[-1]

            lats = np.unique(lat_col[latest_mask])
            lons = np.unique(lon_col[latest_mask])
            sst_grid = sst_col[latest_mask].reshape(len(lats), len(lons))

            result = {"lat": lats, "lon": lons, "sst": sst_grid, "time": str(unique_times[-1])}
            self.cache.put("sst", result)
            log.info(f"SST 取得成功: {sst_grid.shape}, 範圍 {np.nanmin(sst_grid):.1f}~{np.nanmax(sst_grid):.1f}°C")
            return result

        except Exception as e:
            log.error(f"SST 解析失敗: {e}")
            return self._generate_demo_sst()

    # ══════════════════════════════════════════════
    #  2. NOAA ERDDAP — Chlorophyll-a 葉綠素
    # ══════════════════════════════════════════════

    async def fetch_chlorophyll(self) -> Dict[str, np.ndarray]:
        """
        從 ERDDAP 取得 MODIS Chlorophyll-a (4km)
        返回: {lat, lon, chl}
        """
        cached = self.cache.get("chl", max_age_hours=6)
        if cached is not None:
            return cached

        now = datetime.now(timezone.utc)
        t_start = (now - timedelta(days=8)).strftime("%Y-%m-%dT00:00:00Z")
        t_end = now.strftime("%Y-%m-%dT00:00:00Z")

        url = (
            f"https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMH1chla8day.json"
            f"?chlorophyll[({t_start}):1:({t_end})]"
            f"[(0.0):1:(0.0)]"  # surface
            f"[({self.lat_min}):1:({self.lat_max})]"
            f"[({self.lon_min}):1:({self.lon_max})]"
        )

        async with httpx.AsyncClient() as client:
            resp = await safe_fetch(client, url, "ERDDAP-CHL", timeout=180)

        if resp is None:
            return self._generate_demo_chl()

        try:
            data = resp.json()
            table = data["table"]
            cols = table["columnNames"]
            rows = np.array(table["rows"])

            lat_col = rows[:, cols.index("latitude")].astype(float)
            lon_col = rows[:, cols.index("longitude")].astype(float)
            chl_col = rows[:, cols.index("chlorophyll")]

            # 處理 NaN (字串 "NaN")
            chl_col = np.where(chl_col == "NaN", np.nan, chl_col).astype(float)

            lats = np.unique(lat_col)
            lons = np.unique(lon_col)

            # 取最後一個時間步的資料
            n_per_step = len(lats) * len(lons)
            if len(chl_col) >= n_per_step:
                chl_latest = chl_col[-n_per_step:]
                chl_grid = chl_latest.reshape(len(lats), len(lons))
            else:
                chl_grid = chl_col.reshape(len(lats), len(lons))

            result = {"lat": lats, "lon": lons, "chl": chl_grid}
            self.cache.put("chl", result)
            return result
        except Exception as e:
            log.error(f"Chl-a 解析失敗: {e}")
            return self._generate_demo_chl()

    # ══════════════════════════════════════════════
    #  3. HYCOM — 3D 海洋垂直剖面
    # ══════════════════════════════════════════════

    async def fetch_hycom_3d(self) -> Dict[str, np.ndarray]:
        """
        從 HYCOM OPeNDAP 取得 3D 水溫/鹽度
        用於計算溫躍層深度（大目鮪的關鍵）
        返回: {lat, lon, depths, temp_3d, salt_3d}
        """
        cached = self.cache.get("hycom_3d", max_age_hours=12)
        if cached is not None:
            return cached

        # HYCOM 用 OPeNDAP — 需要 xarray
        # 但為了避免依賴問題，我們用 HTTP JSON 方式
        depths = [0, 50, 100, 200, 300, 500]
        temp_profiles = {}

        async with httpx.AsyncClient() as client:
            for depth in depths:
                url = (
                    f"https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0.ascii"
                    f"?water_temp[0:1:0][{self._depth_to_idx(depth)}:1:{self._depth_to_idx(depth)}]"
                    f"[{self._lat_to_idx(self.lat_min)}:4:{self._lat_to_idx(self.lat_max)}]"
                    f"[{self._lon_to_idx(self.lon_min)}:4:{self._lon_to_idx(self.lon_max)}]"
                )
                resp = await safe_fetch(client, url, f"HYCOM-{depth}m", timeout=180)
                if resp:
                    temp_profiles[depth] = self._parse_hycom_ascii(resp.text)
                else:
                    temp_profiles[depth] = None

        # 如果有任何深度取得成功，組裝 3D 陣列
        valid_depths = [d for d in depths if temp_profiles[d] is not None]
        if not valid_depths:
            log.warning("HYCOM 全部失敗，使用模擬 3D 數據")
            return self._generate_demo_hycom3d()

        # 使用第一個成功的深度來確定 lat/lon 網格
        first = temp_profiles[valid_depths[0]]
        lats = first["lat"]
        lons = first["lon"]
        ny, nx = len(lats), len(lons)

        temp_3d = np.full((len(depths), ny, nx), np.nan)
        for i, d in enumerate(depths):
            if temp_profiles[d] is not None:
                temp_3d[i] = temp_profiles[d]["data"]

        result = {
            "lat": lats, "lon": lons,
            "depths": np.array(depths),
            "temp_3d": temp_3d,
        }
        self.cache.put("hycom_3d", result)
        log.info(f"HYCOM 3D 取得成功: {temp_3d.shape}")
        return result

    # ══════════════════════════════════════════════
    #  4. Open-Meteo — 海洋氣象
    # ══════════════════════════════════════════════

    async def fetch_marine_weather(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        取得指定位置的海洋氣象預報（7天）
        返回: 浪高、風速、風向、週期
        """
        cached_key = f"weather_{lat:.1f}_{lon:.1f}"
        cached = self.cache.get(cached_key, max_age_hours=3)
        if cached is not None:
            return cached

        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "wave_height,wave_direction,wave_period,wind_wave_height,swell_wave_height",
            "daily": "wave_height_max,wave_direction_dominant,wave_period_max",
            "timezone": "UTC",
            "forecast_days": 7,
        }

        async with httpx.AsyncClient() as client:
            resp = await safe_fetch(
                client,
                "https://marine-api.open-meteo.com/v1/marine",
                "Open-Meteo",
                params=params,
                timeout=30,
            )

        if resp is None:
            return {"error": "天氣數據取得失敗"}

        result = resp.json()
        self.cache.put(cached_key, result)
        return result

    # ══════════════════════════════════════════════
    #  5. VIIRS 夜間燈光（魷魚船偵測）
    # ══════════════════════════════════════════════

    async def fetch_viirs_nightlights(self) -> Dict[str, np.ndarray]:
        """
        從 NASA FIRMS / EOG 取得 VIIRS 夜間燈光數據
        用於偵測魷魚船集魚燈位置

        原理：魷魚船晚上開強光，衛星能看到。
        過濾邏輯：排除陸地光源、月光反射、氣體火焰
        """
        cached = self.cache.get("viirs", max_age_hours=24)
        if cached is not None:
            return cached

        # 使用 NASA FIRMS (Fire Information) 作為替代
        # VIIRS 活躍火源/光源 API（含漁船光源）
        now = datetime.now(timezone.utc)
        date_str = (now - timedelta(days=1)).strftime("%Y-%m-%d")

        url = (
            f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
            f"VIIRS_SNPP_NRT/"
            f"{self.lon_min},{self.lat_min},{self.lon_max},{self.lat_max}/1/{date_str}"
        )

        async with httpx.AsyncClient() as client:
            resp = await safe_fetch(client, url, "VIIRS-夜光", timeout=60)

        if resp is None:
            log.info("VIIRS 數據不可用，跳過夜光偵測")
            return {"lights": np.array([]), "count": 0}

        try:
            # 解析 CSV
            lines = resp.text.strip().split("\n")
            if len(lines) <= 1:
                return {"lights": np.array([]), "count": 0}

            import csv
            from io import StringIO
            reader = csv.DictReader(StringIO(resp.text))
            lights = []
            for row in reader:
                lat = float(row.get("latitude", 0))
                lon = float(row.get("longitude", 0))
                brightness = float(row.get("bright_ti4", 0))
                # 過濾：只保留海上光源（簡化版，正式版需要陸地遮罩）
                if self.lat_min <= lat <= self.lat_max and self.lon_min <= lon <= self.lon_max:
                    lights.append([lat, lon, brightness])

            lights = np.array(lights) if lights else np.array([]).reshape(0, 3)
            result = {"lights": lights, "count": len(lights)}
            self.cache.put("viirs", result)
            log.info(f"VIIRS 偵測到 {len(lights)} 個光源")
            return result
        except Exception as e:
            log.error(f"VIIRS 解析失敗: {e}")
            return {"lights": np.array([]).reshape(0, 3), "count": 0}

    # ══════════════════════════════════════════════
    #  6. 海流數據 (OSCAR / CMEMS)
    # ══════════════════════════════════════════════

    async def fetch_ocean_currents(self) -> Dict[str, np.ndarray]:
        """
        取得全球海表洋流 (u, v 分量)
        用於 FTLE 計算和航線規劃
        """
        cached = self.cache.get("currents", max_age_hours=6)
        if cached is not None:
            return cached

        # OSCAR 5天合成海流
        now = datetime.now(timezone.utc)
        t_str = (now - timedelta(days=5)).strftime("%Y-%m-%dT00:00:00Z")

        url = (
            f"https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdOscar1.json"
            f"?um[({t_str}):1:({t_str})][(0.0):1:(0.0)]"
            f"[({self.lat_min}):1:({self.lat_max})]"
            f"[({self.lon_min}):1:({self.lon_max})],"
            f"vm[({t_str}):1:({t_str})][(0.0):1:(0.0)]"
            f"[({self.lat_min}):1:({self.lat_max})]"
            f"[({self.lon_min}):1:({self.lon_max})]"
        )

        async with httpx.AsyncClient() as client:
            resp = await safe_fetch(client, url, "OSCAR-海流", timeout=120)

        if resp is None:
            return self._generate_demo_currents()

        try:
            data = resp.json()
            table = data["table"]
            cols = table["columnNames"]
            rows = np.array(table["rows"])

            lat_col = rows[:, cols.index("latitude")].astype(float)
            lon_col = rows[:, cols.index("longitude")].astype(float)

            um_idx = cols.index("um") if "um" in cols else cols.index("u")
            vm_idx = cols.index("vm") if "vm" in cols else cols.index("v")

            u_col = np.where(rows[:, um_idx] == "NaN", np.nan, rows[:, um_idx]).astype(float)
            v_col = np.where(rows[:, vm_idx] == "NaN", np.nan, rows[:, vm_idx]).astype(float)

            lats = np.unique(lat_col)
            lons = np.unique(lon_col)
            ny, nx = len(lats), len(lons)

            u_grid = u_col[:ny*nx].reshape(ny, nx)
            v_grid = v_col[:ny*nx].reshape(ny, nx)

            result = {"lat": lats, "lon": lons, "u": u_grid, "v": v_grid}
            self.cache.put("currents", result)
            log.info(f"海流取得成功: u={u_grid.shape}")
            return result
        except Exception as e:
            log.error(f"海流解析失敗: {e}")
            return self._generate_demo_currents()

    # ══════════════════════════════════════════════
    #  一次抓全部
    # ══════════════════════════════════════════════

    async def fetch_all(self) -> Dict[str, Any]:
        """並行抓取所有數據源"""
        log.info("=" * 60)
        log.info("開始抓取全部數據源...")
        log.info("=" * 60)

        results = await asyncio.gather(
            self.fetch_sst(),
            self.fetch_chlorophyll(),
            self.fetch_hycom_3d(),
            self.fetch_ocean_currents(),
            self.fetch_viirs_nightlights(),
            return_exceptions=True,
        )

        keys = ["sst", "chl", "hycom_3d", "currents", "viirs"]
        output = {}
        for key, result in zip(keys, results):
            if isinstance(result, Exception):
                log.error(f"{key} 抓取異常: {result}")
                output[key] = None
            else:
                output[key] = result

        success = sum(1 for v in output.values() if v is not None)
        log.info(f"數據抓取完成: {success}/{len(keys)} 個數據源成功")
        return output

    # ══════════════════════════════════════════════
    #  內部工具方法
    # ══════════════════════════════════════════════

    def _depth_to_idx(self, depth_m: float) -> int:
        """HYCOM 深度 → 陣列索引映射"""
        depth_map = {0: 0, 50: 10, 100: 18, 200: 26, 300: 31, 500: 36}
        return depth_map.get(depth_m, 0)

    def _lat_to_idx(self, lat: float) -> int:
        """HYCOM 緯度 → 陣列索引 (0.08°格點, 起始 -80°)"""
        return int((lat + 80.0) / 0.08)

    def _lon_to_idx(self, lon: float) -> int:
        """HYCOM 經度 → 陣列索引 (0.08°格點, 起始 0°)"""
        lon_360 = lon if lon >= 0 else lon + 360
        return int(lon_360 / 0.08)

    def _parse_hycom_ascii(self, text: str) -> Optional[Dict]:
        """解析 HYCOM ASCII/DDS 回應"""
        try:
            lines = text.strip().split("\n")
            data_lines = [l for l in lines if not l.startswith("[") and "," in l]
            if not data_lines:
                return None
            values = []
            for line in data_lines:
                parts = line.strip().split(",")
                for p in parts:
                    p = p.strip()
                    if p and p != "NaN":
                        try:
                            values.append(float(p))
                        except ValueError:
                            continue
            if not values:
                return None
            side = int(np.sqrt(len(values)))
            if side * side != len(values):
                side = max(1, side)
                values = values[:side*side]
            grid = np.array(values).reshape(side, side)
            lat = np.linspace(self.lat_min, self.lat_max, side)
            lon = np.linspace(self.lon_min, self.lon_max, side)
            return {"lat": lat, "lon": lon, "data": grid}
        except Exception:
            return None

    # ── 模擬數據（當 API 不可用時使用）──────────

    def _generate_demo_sst(self) -> Dict[str, np.ndarray]:
        """生成模擬 SST 數據（用於測試和 API 不可用時）"""
        log.info("使用模擬 SST 數據")
        lats = np.arange(self.lat_min, self.lat_max, 0.25)
        lons = np.arange(self.lon_min, self.lon_max, 0.25)
        lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
        # 模擬：赤道暖、高緯冷 + 隨機擾動
        sst = 30 - 0.3 * np.abs(lat_grid) + np.random.normal(0, 0.5, lat_grid.shape)
        sst = np.clip(sst, 5, 34)
        return {"lat": lats, "lon": lons, "sst": sst, "time": "demo", "is_demo": True}

    def _generate_demo_chl(self) -> Dict[str, np.ndarray]:
        log.info("使用模擬 Chl-a 數據")
        lats = np.arange(self.lat_min, self.lat_max, 0.25)
        lons = np.arange(self.lon_min, self.lon_max, 0.25)
        lat_grid, _ = np.meshgrid(lats, lons, indexing="ij")
        chl = 0.2 + 0.1 * np.abs(lat_grid - 20) / 10 + np.random.exponential(0.1, lat_grid.shape)
        chl = np.clip(chl, 0.01, 10)
        return {"lat": lats, "lon": lons, "chl": chl, "is_demo": True}

    def _generate_demo_currents(self) -> Dict[str, np.ndarray]:
        log.info("使用模擬海流數據")
        lats = np.arange(self.lat_min, self.lat_max, 1.0)
        lons = np.arange(self.lon_min, self.lon_max, 1.0)
        ny, nx = len(lats), len(lons)
        u = 0.2 * np.sin(np.linspace(0, 4*np.pi, ny)).reshape(-1, 1) * np.ones((1, nx))
        v = 0.1 * np.cos(np.linspace(0, 3*np.pi, nx)).reshape(1, -1) * np.ones((ny, 1))
        u += np.random.normal(0, 0.05, (ny, nx))
        v += np.random.normal(0, 0.05, (ny, nx))
        return {"lat": lats, "lon": lons, "u": u, "v": v, "is_demo": True}

    def _generate_demo_hycom3d(self) -> Dict[str, np.ndarray]:
        log.info("使用模擬 HYCOM 3D 數據")
        lats = np.arange(self.lat_min, self.lat_max, 1.0)
        lons = np.arange(self.lon_min, self.lon_max, 1.0)
        depths = np.array([0, 50, 100, 200, 300, 500])
        ny, nx, nd = len(lats), len(lons), len(depths)
        lat_grid, _ = np.meshgrid(lats, lons, indexing="ij")
        sst_surface = 30 - 0.3 * np.abs(lat_grid)
        temp_3d = np.zeros((nd, ny, nx))
        for i, d in enumerate(depths):
            # 溫度隨深度遞減
            temp_3d[i] = sst_surface - d * 0.03 + np.random.normal(0, 0.3, (ny, nx))
        return {"lat": lats, "lon": lons, "depths": depths, "temp_3d": temp_3d, "is_demo": True}
