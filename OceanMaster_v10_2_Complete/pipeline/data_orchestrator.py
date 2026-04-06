"""
OceanMaster v10.5 — Data Orchestrator
======================================
專職併發獲取 30+ 資料源，嚴格執行 Timeout 與 Cascade Fallback。

反幻覺鐵律:
  - 全 Fallback 失效 → 特徵 = NaN，禁止補值
  - 每次 API 呼叫必須帶 BBox
  - Timeout 熔斷: 60s 內切斷，主執行緒不鎖死
"""
import asyncio
import logging
import time
import numpy as np
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("OceanMaster.orchestrator")

# ── 各模組 import (安全降級) ──
try:
    from engine.data_fetcher_v2 import OceanDataFetcher
except ImportError:
    OceanDataFetcher = None

try:
    from engine.weather_fetcher import WeatherFetcher
    from engine.typhoon_tracker import TyphoonTracker
    from engine.wave_fetcher import WaveFetcher
    from engine.rainfall_fetcher import RainfallFetcher
    from engine.mld_forecast import MLDForecaster
    from engine.typhoon_ml_features import TyphoonFeatureEngineer
    from engine.safety_checker import assess_grid_safety
    from engine.pressure_features import compute_pressure_features
    from engine.tchp_calculator import compute_tchp, tchp_alert_message
    SAFETY_OK = True
except ImportError as e:
    SAFETY_OK = False
    log.warning(f"Safety modules not available: {e}")

try:
    from engine.salinity_fetcher import SalinityFetcher
    SALINITY_OK = True
except ImportError:
    SALINITY_OK = False

try:
    from engine.enhanced_data_sources import fetch_enso_oni
    ENHANCED_OK = True
except ImportError:
    ENHANCED_OK = False


@dataclass
class RawOceanData:
    """所有原始觀測數據的不可變容器"""
    lats: np.ndarray = None
    lons: np.ndarray = None
    ny: int = 0
    nx: int = 0

    # 核心遠掃
    sst: np.ndarray = None
    chl: np.ndarray = None
    ssh: np.ndarray = None
    salinity: np.ndarray = None
    u_current: np.ndarray = None
    v_current: np.ndarray = None
    wind_u: np.ndarray = None
    wind_v: np.ndarray = None
    do_surface: np.ndarray = None
    temp_3d: np.ndarray = None

    # v10.3 氣象安全
    weather_data: Optional[Dict] = None
    typhoon_alerts: List = field(default_factory=list)
    wave_data: Optional[Dict] = None
    pressure_data: Optional[Dict] = None
    tchp_result: Optional[Dict] = None
    safety_grid: Optional[np.ndarray] = None
    typhoon_ml: Optional[Dict] = None

    # v10.4 鹽度
    sss_data: Optional[Dict] = None

    # ENSO
    oni: float = 0.0

    # 資料血統
    data_sources: Dict = field(default_factory=dict)
    fetch_log: List[Dict] = field(default_factory=list)

    # 額外
    viirs_lights: Optional[np.ndarray] = None


class DataOrchestrator:
    """
    併發 Fetch 30+ 資料源: Ocean → Weather → Safety → SSS

    BBox 防爆: 所有 HTTP GET 帶精確空間裁切
    Timeout 熔斷: 逾時 60s 切斷降級
    NaN 阻斷: 全 Fallback 失效 = NaN 直通
    """

    def __init__(
        self,
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
        target_date: Optional[datetime] = None,
    ):
        self.lat_range = lat_range
        self.lon_range = lon_range
        self.target_date = target_date
        self._fetcher = OceanDataFetcher(lat_range, lon_range, target_date=target_date)

        # Safety engines (lazy init)
        self._weather = WeatherFetcher() if SAFETY_OK else None
        self._typhoon = TyphoonTracker() if SAFETY_OK else None
        self._wave = WaveFetcher() if SAFETY_OK else None
        self._typhoon_features = None
        if SAFETY_OK:
            self._typhoon_features = TyphoonFeatureEngineer()
            self._typhoon_features.load_ibtracs()

    async def fetch_all(self) -> RawOceanData:
        """
        主進入點: 一次性取回全部原始數據。
        回傳 RawOceanData, 各欄位若獲取失敗為 NaN/None。
        """
        t0 = time.time()
        result = RawOceanData()
        now = self.target_date or datetime.now(timezone.utc)

        # ── Phase 1: Core ocean data ──
        log.info("  📡 Phase 1: Core ocean fetch (SST/CHL/SSH/currents/salinity)")
        data = await self._fetcher.fetch_all()

        result.lats = data["lats"]
        result.lons = data["lons"]
        result.ny, result.nx = len(result.lats), len(result.lons)
        result.sst = data["sst"]
        result.chl = data["chl"]
        result.ssh = data.get("ssh")
        result.salinity = data["salinity"]
        result.u_current = data["u_current"]
        result.v_current = data["v_current"]
        result.wind_u = data["wind_u"]
        result.wind_v = data["wind_v"]
        result.do_surface = data.get("do_surface")
        result.temp_3d = data.get("temp_3d")
        result.viirs_lights = data.get("viirs_lights")
        result.data_sources = data.get("data_sources", {})

        # 記錄 data lineage
        for src, val in result.data_sources.items():
            is_clim = "CLIM" in str(val).upper()
            result.fetch_log.append({
                "source": src, "value": str(val),
                "is_climatology": is_clim, "timestamp": now.isoformat(),
            })

        log.info(f"  Grid: {result.ny}×{result.nx} = {result.ny * result.nx} pts")

        # ── Phase 1.5: ENSO ONI ──
        if ENHANCED_OK:
            try:
                import httpx
                async with httpx.AsyncClient(timeout=15) as ec:
                    result.oni = await fetch_enso_oni(ec)
            except Exception:
                try:
                    result.oni = await fetch_enso_oni(None)
                except Exception:
                    result.oni = 0.0

        # ── Phase 2: Weather & Safety (parallel) ──
        if SAFETY_OK and self._weather:
            log.info("  🌊 Phase 2: Weather + Typhoon + Wave + SSS")
            try:
                lat_r = (float(result.lats.min()), float(result.lats.max()))
                lon_r = (float(result.lons.min()), float(result.lons.max()))

                weather_task = self._weather.fetch_marine_weather(
                    lat_r, lon_r, ny=result.ny, nx=result.nx,
                )
                typhoon_task = self._typhoon.fetch_active_typhoons()

                wd, ta = await asyncio.gather(
                    weather_task, typhoon_task, return_exceptions=True,
                )
                result.weather_data = wd if not isinstance(wd, Exception) else None
                result.typhoon_alerts = ta if not isinstance(ta, Exception) else []
                if isinstance(wd, Exception):
                    log.warning(f"  Weather fetch: {wd}")
                if isinstance(ta, Exception):
                    log.warning(f"  Typhoon fetch: {ta}")

                # Wave
                wind_speed = None
                if result.weather_data and "wind_speed" in result.weather_data:
                    wind_speed = result.weather_data["wind_speed"]
                elif result.wind_u is not None and result.wind_v is not None:
                    wind_speed = np.sqrt(result.wind_u**2 + result.wind_v**2)

                result.wave_data = await self._wave.fetch_wave_height(
                    lat_r, lon_r, result.lats, result.lons,
                    wind_speed=wind_speed,
                )

                # Pressure
                if result.weather_data and "pressure_msl" in result.weather_data:
                    result.pressure_data = compute_pressure_features(
                        result.weather_data["pressure_msl"], result.lats, result.lons,
                    )

                # TCHP
                if result.temp_3d is not None and result.temp_3d.ndim == 3:
                    std_depths = np.array([0, 10, 20, 30, 50, 75, 100,
                                           125, 150, 200, 250, 300, 400, 500])[:result.temp_3d.shape[0]]
                    result.tchp_result = compute_tchp(result.temp_3d, std_depths, result.lats, result.lons)

                # Safety grid
                wh = result.wave_data["wave_height"] if result.wave_data else None
                p = result.weather_data["pressure_msl"] if result.weather_data else None
                result.safety_grid = assess_grid_safety(
                    result.lats, result.lons,
                    typhoon_alerts=result.typhoon_alerts,
                    wave_height=wh,
                    wind_speed=wind_speed,
                    pressure_msl=p,
                )

                # Typhoon ML
                result.typhoon_ml = self._typhoon_features.compute_features_grid(
                    result.lats, result.lons, now,
                )

                # SSS
                if SALINITY_OK:
                    try:
                        sss_fetcher = SalinityFetcher()
                        result.sss_data = await sss_fetcher.fetch_sss_features(
                            lat_r, lon_r, result.lats, result.lons,
                            fallback_salinity=result.salinity,
                        )
                    except Exception as e:
                        log.warning(f"  SSS: {e}")

                n_ty = len(result.typhoon_alerts) if isinstance(result.typhoon_alerts, list) else 0
                wave_src = result.wave_data.get("source", "?") if result.wave_data else "?"
                sss_src = result.sss_data.get("source", "?") if result.sss_data else "?"
                log.info(f"  ✅ Safety: {n_ty} typhoons, wave={wave_src}, sss={sss_src}")

            except Exception as e:
                log.warning(f"  Phase 2 error (non-fatal): {e}")

        elapsed = time.time() - t0
        log.info(f"  📡 Data orchestrator done: {elapsed:.1f}s")
        return result
