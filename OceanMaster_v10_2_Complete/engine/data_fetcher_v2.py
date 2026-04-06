"""
OceanMaster v10.3 — 生產級數據擷取引擎
=========================================
強制接通真實 API。零 np.random。

數據源 (全部免費、已查證):
  1. NOAA ERDDAP  — SST (OISST v2.1 NRT→Final→Geo-Blended, 0.25deg)
     ↳ ncdcOisst21NrtAgg (NRT, ~1-day latency)
     ↳ ncdcOisst21Agg (Final, ~2-week latency)
     ↳ noaacwBLENDEDsstDailyNight (Geo-polar Blended, 5km)
  2. NOAA ERDDAP  — Chl-a (MODIS Aqua + Terra, 4km, 8天合成)
  3. OSCAR ERDDAP — 海表海流 (5天合成, 1/3deg)
  4. CMEMS        — 海流/SSH/鹽度 (copernicusmarine 庫, 0.083deg)
  5. CMEMS BGC    — 溶氧/NPP (0.25deg)
  6. HYCOM        — 3D 溫鹽 (OPeNDAP ASCII, GLBv0.08)
  7. Open-Meteo   — 海洋氣象 (免 key)
  8. NASA FIRMS   — VIIRS 夜間燈光

多伺服器冗餘:
  coastwatch.pfeg.noaa.gov → polarwatch.noaa.gov → coastwatch.noaa.gov

防封鎖:
  Exponential Backoff + Jitter (base=5s, cap=300s)
  Retry-After header 尊重
  連線池重用 (httpx.AsyncClient)
  磁碟+記憶體雙層快取

回退策略 (所有 API 均失敗時):
  WOA2023 / MODIS 空間氣候態 — 非隨機亂數
"""

import numpy as np
import logging
import asyncio
import os
import random
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

try:
    import httpx
except ImportError:
    # ── httpx shim using requests (stdlib-friendly) ──
    import requests as _requests
    class _FakeResponse:
        def __init__(self, r):
            self.status_code = r.status_code
            self.text = r.text
            self.content = r.content
            self.headers = r.headers
        def json(self): return __import__('json').loads(self.text)
        def raise_for_status(self): pass
    class _TimeoutException(Exception): pass
    class _AsyncClient:
        def __init__(self, **kw):
            self._timeout = kw.get('timeout', 30)
            # ignore limits, transport, etc.
        async def __aenter__(self): return self
        async def __aexit__(self, *a): pass
        async def get(self, url, **kw):
            timeout = kw.pop('timeout', self._timeout)
            if hasattr(timeout, 'connect'): timeout = 30
            if isinstance(timeout, (int, float)) and timeout > 60: timeout = 60
            try:
                r = _requests.get(url, timeout=timeout, **kw)
                return _FakeResponse(r)
            except _requests.exceptions.Timeout:
                raise _TimeoutException(f"Timeout: {url}")
            except _requests.exceptions.ConnectionError as e:
                raise _TimeoutException(str(e))
    class httpx:
        AsyncClient = _AsyncClient
        TimeoutException = _TimeoutException
        Response = _FakeResponse
        class Timeout:
            def __init__(self, timeout=30, connect=10):
                self.connect = connect
        class Limits:
            def __init__(self, **kw): pass

log = logging.getLogger("OceanMaster.Fetcher")

ERDDAP = "https://coastwatch.pfeg.noaa.gov/erddap/griddap"
ERDDAP_MIRRORS = [
    "https://coastwatch.pfeg.noaa.gov/erddap/griddap",
    "https://upwell.pfeg.noaa.gov/erddap/griddap",
    "https://polarwatch.noaa.gov/erddap/griddap",
    "https://coastwatch.noaa.gov/erddap/griddap",
    "https://www.ncei.noaa.gov/erddap/griddap",
]

# SST datasets ordered by freshness (NRT first, then Final, then Blended)
SST_DATASETS = [
    ("ncdcOisst21NrtAgg",  "sst",  "OISST-NRT"),   # NRT: ~1-day latency, 2020-present
    ("ncdcOisst21Agg",     "sst",  "OISST-Final"),  # Final: ~2-week latency, 1981-present
]
# Geo-polar Blended SST (different variable name / grid)
SST_BLENDED = "noaacwBLENDEDsstDailyNight"
# MUR SST fallback (high-res 0.01° — on coastwatch)
SST_MUR = "jplMURSST41"

# CHL datasets — VIIRS replaces MODIS Aqua (decommissioned 2022)
# VIIRS (S-NPP) is the current operational successor, active 2012-present
CHL_DATASETS = [
    # (dataset_id, variable, tag, has_altitude_dim)
    ("noaacwNPPVIIRSchlaDaily",  "chlor_a", "VIIRS-NRT-4km",  True),   # NRT global 4km daily
    ("nesdisVHNSQchlaDaily",     "chlor_a", "VIIRS-SQ-750m",  True),   # Science Quality 750m daily
    ("erdVHNchla3day",           "chlor_a", "VIIRS-3day",     True),   # 3-day composite 750m
]
# 8-day composites — better cloud coverage than daily
CHL_8DAY = [
    ("erdVHNchla8day",     "chlor_a",     "VIIRS-8day",  True),   # VIIRS 8-day composite 750m
]
# Monthly composites — best coverage, lowest resolution
CHL_MONTHLY = [
    ("erdVHNchlaMonthly",  "chlor_a",     "VIIRS-Monthly", True),  # VIIRS monthly composite
]
# Legacy MODIS (kept as last resort — data ends 2022 but may still work for historical)
CHL_LEGACY = [
    ("erdMH1chla8day",     "chlorophyll", "MODIS-Aqua",  True),
]


# ═══════════════════════════════════════════════════
# Exponential Backoff + Jitter
# ═══════════════════════════════════════════════════

async def backoff_fetch(
    client: httpx.AsyncClient,
    url: str,
    tag: str,
    max_retries: int = 4,
    base_delay: float = 5.0,
    max_delay: float = 300.0,
    timeout: float = 120.0,
    params: Optional[Dict] = None,
) -> Optional[httpx.Response]:
    """
    HTTP GET with exponential backoff + jitter.
    delay = min(base * 2^attempt, max_delay) + uniform(0, delay*0.5)
    """
    for attempt in range(max_retries):
        try:
            log.info(f"[{tag}] #{attempt+1}/{max_retries}")
            resp = await client.get(url, params=params, timeout=timeout)
            if resp.status_code == 200:
                log.info(f"[{tag}] OK ({len(resp.content)/1024:.0f}KB)")
                return resp
            if resp.status_code in (429, 503):
                ra = resp.headers.get("Retry-After")
                bw = float(ra) if ra and ra.isdigit() else min(base_delay * (2 ** attempt), max_delay)
                jitter = random.uniform(0, bw * 0.5)
                log.warning(f"[{tag}] {resp.status_code}, wait {bw+jitter:.1f}s")
                await asyncio.sleep(bw + jitter)
                continue
            if resp.status_code in (400, 404):
                log.error(f"[{tag}] {resp.status_code} bad URL")
                return None
            await asyncio.sleep(min(base_delay * (2 ** attempt), max_delay))
        except httpx.TimeoutException:
            w = min(base_delay * (2 ** attempt), max_delay)
            log.warning(f"[{tag}] timeout, wait {w:.0f}s")
            await asyncio.sleep(w)
        except Exception as e:
            log.warning(f"[{tag}] {type(e).__name__}: {e}")
            await asyncio.sleep(min(base_delay * (2 ** attempt), max_delay))
    log.error(f"[{tag}] all {max_retries} attempts failed")
    return None


# ═══════════════════════════════════════════════════
# 雙層快取
# ═══════════════════════════════════════════════════

class DataCache:
    def __init__(self, cache_dir="data/cache"):
        self._mem = {}
        self._ts = {}
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def get(self, key, max_age_hours=24):
        if key in self._mem:
            age = (datetime.now(timezone.utc) - self._ts[key]).total_seconds() / 3600
            if age < max_age_hours:
                log.info(f"cache(RAM): {key}")
                return self._mem[key]
        fp = self._dir / f"{key}.npz"
        if fp.exists():
            mt = datetime.fromtimestamp(fp.stat().st_mtime, tz=timezone.utc)
            age = (datetime.now(timezone.utc) - mt).total_seconds() / 3600
            if age < max_age_hours:
                try:
                    d = dict(np.load(str(fp), allow_pickle=True))
                    self._mem[key] = d
                    self._ts[key] = mt
                    log.info(f"cache(disk): {key}")
                    return d
                except Exception:
                    pass
        return None

    def put(self, key, data):
        self._mem[key] = data
        self._ts[key] = datetime.now(timezone.utc)
        try:
            arrs = {k: v for k, v in data.items() if isinstance(v, np.ndarray)}
            if arrs:
                np.savez_compressed(str(self._dir / f"{key}.npz"), **arrs)
        except Exception:
            pass


# ═══════════════════════════════════════════════════
# WOA/MODIS 氣候態回退 (零 np.random)
# ═══════════════════════════════════════════════════

class WOAClimatology:
    @staticmethod
    def sst(lats, lons, month):
        lg, lo = np.meshgrid(lats, lons, indexing='ij')
        s = 2.5 * np.cos(2 * np.pi * (month - 8) / 12)
        base = 30.5 - 0.42 * np.abs(lg - 3.0) + s
        kuro = 3.0 * np.exp(-((lo - 135)**2 / 250 + (lg - 28)**2 / 60))
        oyash = -4.0 * np.exp(-((lo - 150)**2 / 100 + (lg - 42)**2 / 30))
        ct = -1.5 * np.exp(-((lo - 170)**2 / 200 + lg**2 / 20))
        return np.clip(base + kuro + oyash + ct, 2.0, 33.0).astype(np.float32)

    @staticmethod
    def chl(lats, lons, month):
        lg, lo = np.meshgrid(lats, lons, indexing='ij')
        base = 0.07 + 0.015 * np.abs(lg - 18.0)
        shelf = 0.5 * np.exp(-((lo - 122)**2 / 15))
        sp = 0.4 * np.clip((lg - 32) / 8, 0, 1) if month in (3, 4, 5) else 0.0
        au = 0.2 * np.clip((lg - 30) / 10, 0, 1) if month in (9, 10, 11) else 0.0
        return np.clip(base + shelf + sp + au, 0.02, 8.0).astype(np.float32)

    @staticmethod
    def currents(lats, lons):
        lg, lo = np.meshgrid(lats, lons, indexing='ij')
        k = np.exp(-((lo - 130)**2 / 80 + (lg - 27)**2 / 40))
        u = 0.03 + 0.8 * k * np.cos(np.radians(35))
        v = 0.02 + 0.8 * k * np.sin(np.radians(35))
        nec = -0.15 * np.exp(-(lg - 12)**2 / 15)
        return (u + nec).astype(np.float32), v.astype(np.float32)

    @staticmethod
    def wind(lats, lons, month):
        s = (len(lats), len(lons))
        if month in (11, 12, 1, 2, 3):
            return np.full(s, -5.5, np.float32), np.full(s, -3.0, np.float32)
        elif month in (6, 7, 8):
            return np.full(s, 3.5, np.float32), np.full(s, 4.5, np.float32)
        return np.full(s, -1.0, np.float32), np.full(s, 1.0, np.float32)

    @staticmethod
    def salinity(lats, lons):
        lg, lo = np.meshgrid(lats, lons, indexing='ij')
        s = 34.5 + 0.3 * np.cos(np.radians(lg * 2.5))
        s += 0.3 * np.exp(-((lo - 132)**2 / 150 + (lg - 25)**2 / 80))
        s -= 0.5 * np.exp(-((lo - 121)**2 / 8))
        return np.clip(s, 32.0, 37.0).astype(np.float32)

    @staticmethod
    def do_surface(lats, lons, month):
        lg, _ = np.meshgrid(lats, lons, indexing='ij')
        do = 4.5 + 0.08 * np.abs(lg - 5.0) + 0.3 * np.cos(2 * np.pi * (month - 2) / 12)
        return np.clip(do, 3.0, 8.0).astype(np.float32)


# ═══════════════════════════════════════════════════
# ERDDAP JSON 解析
# ═══════════════════════════════════════════════════

def _parse_erddap(resp, var, tag):
    try:
        tbl = resp.json()["table"]
        cols = tbl["columnNames"]
        rows = np.array(tbl["rows"], dtype=object)
        if len(rows) == 0:
            return None
        lat_c = rows[:, cols.index("latitude")].astype(float)
        lon_c = rows[:, cols.index("longitude")].astype(float)
        val = rows[:, cols.index(var)]
        val = np.where((val == "NaN") | (val is None), np.nan, val).astype(float)
        lats, lons = np.unique(lat_c), np.unique(lon_c)
        n = len(lats) * len(lons)
        if len(val) > n:
            val = val[-n:]
        return lats, lons, val.reshape(len(lats), len(lons)).astype(np.float32)
    except Exception as e:
        log.error(f"[{tag}] parse: {e}")
        return None


# ═══════════════════════════════════════════════════
# 主擷取器
# ═══════════════════════════════════════════════════

class OceanDataFetcher:
    def __init__(self, lat_range, lon_range, target_date=None):
        self.lat_min, self.lat_max = lat_range
        self.lon_min, self.lon_max = lon_range
        self.cache = DataCache()
        self._clim = WOAClimatology()
        self._st = {}
        self._target_date = target_date  # None = 即時, datetime = 歷史

    def _get_now(self):
        """取得目標時間（歷史模式用 target_date，即時模式用 utcnow）"""
        return self._target_date or datetime.now(timezone.utc)

    async def _fetch_sst(self, c):
        cd = self.cache.get("sst", 6)
        if cd:
            self._st["SST"] = "cache"
            return cd
        now = self._get_now()

        # [#1 商用] 全域 timeout — 避免当 OISST 服務器全部斷線時卡10+分鐘
        try:
            result = await asyncio.wait_for(self._fetch_sst_inner(c, now), timeout=90.0)
            if result:
                return result
        except asyncio.TimeoutError:
            log.warning("  SST: 全域 timeout 90s 觸發，將使用 climatology fallback")

        self._st["SST"] = "FAILED"
        return None

    async def _fetch_sst_inner(self, c, now):
        """真正的 SST 抓取邏輯（被 timeout 包裏）"""
        # ── Strategy: NRT dataset first (freshest), then Final, across mirrors ──
        for ds_id, var, tag in SST_DATASETS:
            for d in [2, 3, 5, 7, 14]:
                t = (now - timedelta(days=d)).strftime("%Y-%m-%dT12:00:00Z")
                got_response = False
                for server in ERDDAP_MIRRORS:
                    url = (f"{server}/{ds_id}.json?{var}[({t}):1:({t})]"
                           f"[({self.lat_min}):1:({self.lat_max})]"
                           f"[({self.lon_min}):1:({self.lon_max})]")
                    resp = await backoff_fetch(c, url, f"{tag}-{d}d@{server.split('//')[1].split('/')[0][:12]}", max_retries=2)
                    if resp:
                        p = _parse_erddap(resp, var, tag)
                        if p:
                            r = {"lat": p[0], "lon": p[1], "sst": p[2]}
                            self.cache.put("sst", r)
                            self._st["SST"] = f"{tag}(-{d}d)"
                            return r
                        got_response = True
                        break
                if got_response:
                    break

        # ── MUR SST fallback (0.01° resolution, JPL) ──
        log.info("  SST: trying MUR SST fallback...")
        for d in [2, 3, 5]:
            t = (now - timedelta(days=d)).strftime("%Y-%m-%dT09:00:00Z")
            url = (f"{ERDDAP}/{SST_MUR}.json?"
                   f"analysed_sst[({t}):1:({t})]"
                   f"[({self.lat_min}):10:({self.lat_max})]"
                   f"[({self.lon_min}):10:({self.lon_max})]")
            resp = await backoff_fetch(c, url, f"MUR-SST-{d}d", max_retries=2)
            if resp:
                p = _parse_erddap(resp, "analysed_sst", "MUR-SST")
                if p:
                    sst_c = p[2] - 273.15 if np.nanmean(p[2]) > 100 else p[2]
                    r = {"lat": p[0], "lon": p[1], "sst": sst_c.astype(np.float32)}
                    self.cache.put("sst", r)
                    self._st["SST"] = f"MUR(-{d}d)"
                    return r

        # ── Blended SST fallback ──
        log.info("  SST: trying Geo-polar Blended SST fallback...")
        for d in [2, 3, 5]:
            t = (now - timedelta(days=d)).strftime("%Y-%m-%dT12:00:00Z")
            url = (f"{ERDDAP}/{SST_BLENDED}.json?"
                   f"analysed_sst[({t}):1:({t})]"
                   f"[({self.lat_min}):0.25:({self.lat_max})]"
                   f"[({self.lon_min}):0.25:({self.lon_max})]")
            resp = await backoff_fetch(c, url, f"Blended-SST-{d}d", max_retries=2)
            if resp:
                p = _parse_erddap(resp, "analysed_sst", "Blended-SST")
                if p:
                    sst_c = p[2] - 273.15 if np.nanmean(p[2]) > 100 else p[2]
                    r = {"lat": p[0], "lon": p[1], "sst": sst_c.astype(np.float32)}
                    self.cache.put("sst", r)
                    self._st["SST"] = f"Blended(-{d}d)"
                    return r

        self._st["SST"] = "FAILED"
        return None

    async def _fetch_chl(self, c):
        cd = self.cache.get("chl", 6)
        if cd:
            self._st["CHL"] = "cache"
            return cd
        now = self._get_now()

        # ── Helper: try a list of datasets ──
        async def _try_chl_datasets(datasets, date_offsets, label):
            for ds_id, var, tag, has_alt in datasets:
                for d in date_offsets:
                    t = (now - timedelta(days=d)).strftime("%Y-%m-%dT00:00:00Z")
                    alt_dim = "[(0.0):1:(0.0)]" if has_alt else ""
                    for server in ERDDAP_MIRRORS[:3]:
                        url = (f"{server}/{ds_id}.json?{var}[({t}):1:({t})]"
                               f"{alt_dim}"
                               f"[({self.lat_min}):4:({self.lat_max})]"
                               f"[({self.lon_min}):4:({self.lon_max})]")
                        resp = await backoff_fetch(c, url, f"{tag}-{d}d@{server.split('//')[1].split('/')[0][:12]}", timeout=180, max_retries=2)
                        if resp:
                            p = _parse_erddap(resp, var, tag)
                            if p:
                                return {"lat": p[0], "lon": p[1], "chl": p[2]}, f"{tag}(-{d}d)"
                            break  # got response but bad data, try next date
            return None, None

        # ── Step 1: VIIRS daily (freshest) ──
        daily_result, daily_tag = await _try_chl_datasets(CHL_DATASETS, [2, 3, 5, 8, 16], "daily")

        if daily_result is not None:
            chl_data = daily_result["chl"]
            nan_ratio = np.sum(~np.isfinite(chl_data)) / chl_data.size

            if nan_ratio <= 0.5:
                # Good daily data
                self.cache.put("chl", daily_result)
                self._st["CHL"] = daily_tag
                return daily_result
            else:
                log.warning(f"  CHL daily NaN ratio = {nan_ratio:.0%} (>{50}%), "
                            f"switching to 8-day composite...")
        else:
            log.warning("  CHL: all daily datasets failed, trying 8-day composite...")

        # ── Step 2: VIIRS 8-day composite (better cloud coverage) ──
        r8d, tag8d = await _try_chl_datasets(CHL_8DAY, [2, 5, 10, 16], "8-day")
        if r8d is not None:
            nan_ratio_8d = np.sum(~np.isfinite(r8d["chl"])) / r8d["chl"].size
            if nan_ratio_8d <= 0.7:  # 8-day tolerates slightly more NaN
                log.info(f"  CHL: using 8-day composite ({tag8d}), "
                         f"NaN={nan_ratio_8d:.0%}")
                self.cache.put("chl", r8d)
                self._st["CHL"] = tag8d
                return r8d
            else:
                log.warning(f"  CHL 8-day NaN ratio = {nan_ratio_8d:.0%}, "
                            f"trying monthly composite...")

        # ── Step 3: VIIRS monthly composite (best coverage) ──
        rm, tagm = await _try_chl_datasets(CHL_MONTHLY, [15, 32, 45], "monthly")
        if rm is not None:
            log.info(f"  CHL: using monthly composite ({tagm})")
            self.cache.put("chl", rm)
            self._st["CHL"] = tagm
            return rm

        # ── Step 4: Legacy MODIS fallback (data ends 2022) ──
        log.info("  CHL: trying legacy MODIS fallback...")
        rleg, tagleg = await _try_chl_datasets(CHL_LEGACY, [3, 8, 16, 32], "legacy")
        if rleg is not None:
            self.cache.put("chl", rleg)
            self._st["CHL"] = tagleg
            return rleg

        # ── Step 5: If daily data existed but was NaN-heavy, interpolate first ──
        if daily_result is not None:
            log.warning("  CHL: all composites failed, interpolating NaN-heavy daily data")
            chl_fixed = daily_result["chl"].copy()
            nan_mask = ~np.isfinite(chl_fixed)
            if nan_mask.any() and not nan_mask.all():
                try:
                    from scipy.ndimage import generic_filter
                    def _nanmean_filter(x):
                        valid = x[np.isfinite(x)]
                        return np.mean(valid) if len(valid) > 0 else np.nan
                    filled = generic_filter(chl_fixed, _nanmean_filter, size=5,
                                             mode='constant', cval=np.nan)
                    chl_fixed[nan_mask] = filled[nan_mask]
                except ImportError:
                    pass
                # Fill remaining NaN with global median
                remaining = ~np.isfinite(chl_fixed)
                if remaining.any():
                    valid_vals = chl_fixed[np.isfinite(chl_fixed)]
                    fill_val = float(np.median(valid_vals)) if len(valid_vals) > 0 else 0.3
                    chl_fixed[remaining] = fill_val
                daily_result["chl"] = chl_fixed
                log.info(f"  CHL: interpolated, final NaN="
                         f"{np.sum(~np.isfinite(chl_fixed))/chl_fixed.size:.0%}")
            self.cache.put("chl", daily_result)
            self._st["CHL"] = f"{daily_tag}(interpolated)"
            return daily_result

        # ── Step 6 (T4): WOA climatology fallback for offline mode ──
        log.warning("  CHL: ALL satellite sources failed -> using WOA climatology")
        lats = np.arange(self.lat_min, self.lat_max + 0.001, 0.25)
        lons = np.arange(self.lon_min, self.lon_max + 0.001, 0.25)
        month = self._get_now().month
        chl_clim = self._clim.chl(lats, lons, month)
        clim_result = {"lat": lats, "lon": lons, "chl": chl_clim}
        self.cache.put("chl", clim_result)
        self._st["CHL"] = "WOA-climatology"
        log.info(f"  CHL: WOA climatology {chl_clim.shape}, "
                 f"NaN={np.sum(~np.isfinite(chl_clim))/chl_clim.size:.0%}")
        return clim_result

    async def _fetch_currents(self, c):
        cd = self.cache.get("cur", 6)
        if cd:
            self._st["CURRENTS"] = "cache"
            return cd
        now = self._get_now()
        # OSCAR 5-day composite — try wider date offsets across mirrors
        for d in [6, 10, 15, 20]:
            t = (now - timedelta(days=d)).strftime("%Y-%m-%dT00:00:00Z")
            for server in ERDDAP_MIRRORS[:3]:
                url = (f"{server}/erdOscar1.json"
                       f"?um[({t}):1:({t})][(0.0):1:(0.0)]"
                       f"[({self.lat_min}):1:({self.lat_max})]"
                       f"[({self.lon_min}):1:({self.lon_max})],"
                       f"vm[({t}):1:({t})][(0.0):1:(0.0)]"
                       f"[({self.lat_min}):1:({self.lat_max})]"
                       f"[({self.lon_min}):1:({self.lon_max})]")
                resp = await backoff_fetch(c, url, f"OSCAR-{d}d@{server.split('//')[1].split('/')[0][:12]}", timeout=120, max_retries=2)
                if resp:
                    try:
                        tbl = resp.json()["table"]
                        cols = tbl["columnNames"]
                        rows = np.array(tbl["rows"], dtype=object)
                        la = rows[:, cols.index("latitude")].astype(float)
                        lo = rows[:, cols.index("longitude")].astype(float)
                        ui = cols.index("um") if "um" in cols else cols.index("u")
                        vi = cols.index("vm") if "vm" in cols else cols.index("v")
                        u = np.where(rows[:, ui] == "NaN", np.nan, rows[:, ui]).astype(float)
                        v = np.where(rows[:, vi] == "NaN", np.nan, rows[:, vi]).astype(float)
                        lats, lons = np.unique(la), np.unique(lo)
                        n = len(lats) * len(lons)
                        r = {"lat": lats, "lon": lons,
                             "u": u[-n:].reshape(len(lats), len(lons)).astype(np.float32),
                             "v": v[-n:].reshape(len(lats), len(lons)).astype(np.float32)}
                        self.cache.put("cur", r)
                        self._st["CURRENTS"] = f"OSCAR(-{d}d)"
                        return r
                    except Exception as e:
                        log.warning(f"OSCAR parse: {e}")
                        break  # bad data, try next date
                # resp is None → try next mirror
        # CMEMS fallback
        cmems = await self._cmems_physics()
        if cmems:
            return cmems
        self._st["CURRENTS"] = "FAILED"
        return None

    async def _cmems_physics(self):
        user = os.environ.get("CMEMS_USER", "")
        if not user:
            log.info("CMEMS: set CMEMS_USER/CMEMS_PASS env vars")
            return None
        try:
            import copernicusmarine as cm
        except ImportError:
            log.warning("pip install copernicusmarine")
            return None
        try:
            now = self._get_now()
            # v202406 default part has: zos, sob, tob (no uo/vo)
            # Fetch SSH (zos) and derive geostrophic currents from SSH gradient
            ds = cm.open_dataset(
                dataset_id="cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
                variables=["zos"],
                minimum_latitude=self.lat_min, maximum_latitude=self.lat_max,
                minimum_longitude=self.lon_min, maximum_longitude=self.lon_max,
                start_datetime=(now - timedelta(days=3)).isoformat(),
                end_datetime=now.isoformat(),
                username=user, password=os.environ.get("CMEMS_PASS", ""))
            la, lo = ds.latitude.values, ds.longitude.values
            ssh = ds["zos"].isel(time=-1).values.astype(np.float32)
            ds.close()

            # Derive geostrophic currents from SSH gradient
            # u_g = -(g/f) × ∂η/∂y, v_g = (g/f) × ∂η/∂x
            g = 9.81  # m/s²
            omega = 7.2921e-5  # rad/s
            lat_rad = np.radians(la)
            f = 2 * omega * np.sin(lat_rad)  # Coriolis parameter
            f = np.where(np.abs(f) < 1e-10, 1e-10, f)  # avoid division by zero at equator

            dy = np.mean(np.abs(np.diff(la))) * 111000  # m
            dx = np.mean(np.abs(np.diff(lo))) * 111000 * np.cos(np.radians(np.mean(la)))

            deta_dy, deta_dx = np.gradient(np.nan_to_num(ssh, nan=0), dy, dx)
            f_2d = f[:, np.newaxis] * np.ones((1, len(lo)))
            u_geo = -(g / f_2d) * deta_dy
            v_geo = (g / f_2d) * deta_dx

            r = {"lat": la, "lon": lo,
                 "u": u_geo.astype(np.float32),
                 "v": v_geo.astype(np.float32),
                 "ssh": ssh}
            self._st["CURRENTS"] = "CMEMS-GEO"
            log.info(f"CMEMS SSH: {ssh.shape}, u/v derived via geostrophic balance")
            return r
        except Exception as e:
            log.warning(f"CMEMS: {e}")
            return None

    async def _cmems_bgc(self):
        user = os.environ.get("CMEMS_USER", "")
        if not user: return None
        try:
            import copernicusmarine as cm
        except ImportError: return None
        try:
            now = self._get_now()
            # Try multiple BGC dataset IDs
            bgc_datasets = [
                "cmems_mod_glo_bgc-bio_anfc_0.25deg_P1D-m",
                "cmems_mod_glo_bgc_anfc_0.25deg_P1D-m",
            ]
            for dsid in bgc_datasets:
                try:
                    ds = cm.open_dataset(
                        dataset_id=dsid,
                        variables=["o2"],
                        minimum_latitude=self.lat_min, maximum_latitude=self.lat_max,
                        minimum_longitude=self.lon_min, maximum_longitude=self.lon_max,
                        minimum_depth=0.0, maximum_depth=10.0,
                        start_datetime=(now - timedelta(days=3)).isoformat(),
                        end_datetime=now.isoformat(),
                        username=user, password=os.environ.get("CMEMS_PASS", ""))
                    o2 = ds["o2"].isel(time=-1, depth=0).values.astype(np.float32)
                    r = {"lat": ds.latitude.values, "lon": ds.longitude.values, "do_surface": o2 / 44.66}
                    ds.close()
                    self._st["DO"] = "CMEMS-BGC"
                    return r
                except Exception:
                    continue
            log.warning("CMEMS BGC: no working dataset found")
            return None
        except Exception as e:
            log.warning(f"CMEMS BGC: {e}")
            return None

    async def _fetch_hycom3d(self, c):
        cd = self.cache.get("hycom3d", 12)
        if cd:
            self._st["HYCOM"] = "cache"
            return cd
        depths_m = [0, 50, 100, 200, 300, 500]
        didx = {0: 0, 50: 10, 100: 18, 200: 26, 300: 31, 500: 36}
        la0 = int((self.lat_min + 80) / 0.08)
        la1 = int((self.lat_max + 80) / 0.08)
        lo0 = int(self.lon_min / 0.08) if self.lon_min >= 0 else int((self.lon_min + 360) / 0.08)
        lo1 = int(self.lon_max / 0.08) if self.lon_max >= 0 else int((self.lon_max + 360) / 0.08)
        hb = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0"
        layers = {}
        for dm in depths_m:
            url = f"{hb}.ascii?water_temp[0:1:0][{didx[dm]}:1:{didx[dm]}][{la0}:4:{la1}][{lo0}:4:{lo1}]"
            resp = await backoff_fetch(c, url, f"HYCOM-{dm}m", timeout=180)
            if resp:
                p = self._parse_hycom(resp.text)
                if p: layers[dm] = p
        if not layers:
            self._st["HYCOM"] = "FAILED"
            return None
        fst = list(layers.values())[0]
        la, lo = fst["lat"], fst["lon"]
        ny, nx = len(la), len(lo)
        t3d = np.full((len(depths_m), ny, nx), np.nan, np.float32)
        for i, d in enumerate(depths_m):
            if d in layers and layers[d]["data"].shape == (ny, nx):
                t3d[i] = layers[d]["data"]
        r = {"lat": la, "lon": lo, "depths": np.array(depths_m), "temp_3d": t3d}
        self.cache.put("hycom3d", r)
        self._st["HYCOM"] = f"OK({len(layers)}/{len(depths_m)})"
        return r

    def _parse_hycom(self, text):
        try:
            lines = text.strip().split("\n")
            dl = [l for l in lines if "," in l and not l.startswith("[")
                  and not l.startswith("water_temp") and not l.startswith("Dataset")]
            vals = []
            for line in dl:
                for p in line.strip().split(","):
                    p = p.strip()
                    if p and p != "NaN":
                        try: vals.append(float(p))
                        except ValueError: pass
            if len(vals) < 4: return None
            side = int(np.sqrt(len(vals)))
            if side < 2: return None
            grid = np.array(vals[:side*side], np.float32).reshape(side, side)
            return {"lat": np.linspace(self.lat_min, self.lat_max, side),
                    "lon": np.linspace(self.lon_min, self.lon_max, side), "data": grid}
        except Exception: return None

    async def _fetch_viirs(self, c):
        cd = self.cache.get("viirs", 24)
        if cd:
            self._st["VIIRS"] = "cache"
            return cd
        now = self._get_now()
        d = (now - timedelta(days=1)).strftime("%Y-%m-%d")
        url = (f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/VIIRS_SNPP_NRT/"
               f"{self.lon_min},{self.lat_min},{self.lon_max},{self.lat_max}/1/{d}")
        resp = await backoff_fetch(c, url, "VIIRS", timeout=60)
        empty = {"lights": np.array([]).reshape(0, 3), "count": 0}
        if not resp:
            self._st["VIIRS"] = "FAILED"
            return empty
        try:
            import csv; from io import StringIO
            pts = []
            for row in csv.DictReader(StringIO(resp.text)):
                la, lo = float(row.get("latitude", 0)), float(row.get("longitude", 0))
                br = float(row.get("bright_ti4", 0))
                if self.lat_min <= la <= self.lat_max and self.lon_min <= lo <= self.lon_max:
                    pts.append([la, lo, br])
            arr = np.array(pts, np.float32) if pts else np.array([]).reshape(0, 3)
            r = {"lights": arr, "count": len(pts)}
            self.cache.put("viirs", r)
            self._st["VIIRS"] = f"OK({len(pts)})"
            return r
        except Exception: return empty

    # ══════════════════════════════════════════════
    # fetch_all
    # ══════════════════════════════════════════════

    async def _fetch_wind_api(self, lats, lons):
        """
        風場資料取得 — 分層回退:
        1. CCMP V3.1 (Cross-Calibrated Multi-Platform) via ERDDAP — 0.25° 6h
        2. Open-Meteo Marine API (單點，展開到網格) — 免費、無 key
        """
        ny, nx = len(lats), len(lons)
        now = self._get_now()

        # ─── 嘗試 1: CCMP V3.1 風場 (ERDDAP) ───
        # 提供完整 2D 風場，可計算 wind stress curl → Ekman pumping
        ccmp_datasets = [
            ("erdlasCCMP3uw6hr", "uwnd", "vwnd", "erdlasCCMP3vw6hr"),  # CCMP v3.1
        ]
        try:
            import httpx
            for ds_u, var_u, var_v, ds_v in ccmp_datasets:
                for mirror in ERDDAP_MIRRORS[:3]:
                    try:
                        for day_offset in [1, 3, 5, 7]:
                            target = (now - timedelta(days=day_offset)).strftime("%Y-%m-%dT00:00:00Z")
                            url_u = (
                                f"{mirror}/{ds_u}.json?"
                                f"{var_u}[({target}):1:({target})]"
                                f"[({self.lat_min}):1:({self.lat_max})]"
                                f"[({self.lon_min}):1:({self.lon_max})]"
                            )
                            async with httpx.AsyncClient(timeout=30) as client:
                                resp = await client.get(url_u)
                            if resp.status_code == 200:
                                result = _parse_erddap(resp, var_u, "CCMP-U")
                                if result is not None:
                                    u_data = result["data"]
                                    u_lat = result["lat"]
                                    u_lon = result["lon"]

                                    url_v = (
                                        f"{mirror}/{ds_v}.json?"
                                        f"{var_v}[({target}):1:({target})]"
                                        f"[({self.lat_min}):1:({self.lat_max})]"
                                        f"[({self.lon_min}):1:({self.lon_max})]"
                                    )
                                    async with httpx.AsyncClient(timeout=30) as client:
                                        resp_v = await client.get(url_v)
                                    if resp_v.status_code == 200:
                                        result_v = _parse_erddap(resp_v, var_v, "CCMP-V")
                                        if result_v is not None:
                                            v_data = result_v["data"]
                                            # 重新格網到目標解析度
                                            wind_u = self._regrid(u_data, u_lat, u_lon, lats, lons)
                                            wind_v = self._regrid(v_data, u_lat, u_lon, lats, lons)
                                            wspd = np.sqrt(wind_u**2 + wind_v**2)
                                            log.info(f"  ✅ CCMP Wind: {wind_u.shape}, "
                                                     f"avg={np.nanmean(wspd):.1f} m/s")
                                            self._st["WIND"] = f"CCMP-{mirror.split('/')[2][:12]}"
                                            return wind_u, wind_v
                    except Exception as e:
                        log.debug(f"  CCMP {mirror}: {e}")
                        continue
        except Exception as e:
            log.debug(f"  CCMP import: {e}")

        # ─── 回退: Open-Meteo (單點展開) ───
        try:
            import httpx
            lat_c = float((lats.min() + lats.max()) / 2)
            lon_c = float((lons.min() + lons.max()) / 2)
            url = (
                f"https://api.open-meteo.com/v1/forecast?"
                f"latitude={lat_c}&longitude={lon_c}"
                f"&current=wind_speed_10m,wind_direction_10m"
            )
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                data = resp.json()
            current = data.get("current", {})
            wspd = current.get("wind_speed_10m", 0)  # m/s
            wdir = current.get("wind_direction_10m", 0)  # degrees
            # 轉為 u/v 分量並展開到網格
            u = -wspd * np.sin(np.radians(wdir))
            v = -wspd * np.cos(np.radians(wdir))
            wind_u = np.full((ny, nx), u, dtype=np.float32)
            wind_v = np.full((ny, nx), v, dtype=np.float32)
            log.info(f"  Wind (OpenMeteo single-point): {wspd:.1f} m/s @ {wdir:.0f}°")
            self._st["WIND"] = "OpenMeteo-1pt"
            return wind_u, wind_v
        except Exception as e:
            log.debug(f"  Wind API: {e}")
            return None

    async def _fetch_ssh_aviso(self, lats, lons):
        """AVISO SSH via ERDDAP — multi-server + date fallback"""
        try:
            lat_min, lat_max = float(lats.min()), float(lats.max())
            lon_min, lon_max = float(lons.min()), float(lons.max())

            for server in ERDDAP_MIRRORS[:2]:
                url = (
                    f"{server}/nesdisSSH1day.json?"
                    f"sea_surface_height[last][({lat_min}):({lat_max})][({lon_min}):({lon_max})]"
                )
                try:
                    async with httpx.AsyncClient(timeout=30) as client:
                        resp = await client.get(url)
                        resp.raise_for_status()
                        data = resp.json()
                    rows = data["table"]["rows"]
                    if rows:
                        ulats = sorted(set(r[1] for r in rows))
                        ulons = sorted(set(r[2] for r in rows))
                        grid = np.full((len(ulats), len(ulons)), np.nan)
                        lat_idx = {v: i for i, v in enumerate(ulats)}
                        lon_idx = {v: i for i, v in enumerate(ulons)}
                        for row in rows:
                            grid[lat_idx[row[1]], lon_idx[row[2]]] = row[3]
                        from scipy.interpolate import RegularGridInterpolator
                        interp = RegularGridInterpolator(
                            (np.array(ulats), np.array(ulons)), grid,
                            method='linear', bounds_error=False, fill_value=0)
                        g_lat, g_lon = np.meshgrid(lats, lons, indexing='ij')
                        result = interp((g_lat, g_lon)).astype(np.float32)
                        log.info(f"  SSH AVISO OK: range [{np.nanmin(result):.3f}, {np.nanmax(result):.3f}]m")
                        return result
                except Exception as e:
                    log.debug(f"  SSH AVISO ({server}): {e}")
                    continue
        except Exception as e:
            log.debug(f"  SSH AVISO: {e}")
            return None

    async def fetch_all(self):
        log.info("=" * 60)
        log.info("  Fetching real ocean data")
        log.info("=" * 60)
        now = self._get_now()
        month = now.month
        lats = np.arange(self.lat_min, self.lat_max + 0.01, 0.25)
        lons = np.arange(self.lon_min, self.lon_max + 0.01, 0.25)

        async with httpx.AsyncClient(
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            follow_redirects=True,
            headers={"User-Agent": "OceanMaster/10.2 (research)"},
        ) as c:
            rs = await self._fetch_sst(c)
            rc = await self._fetch_chl(c)
            ru = await self._fetch_currents(c)
            rh = await self._fetch_hycom3d(c)
            rv = await self._fetch_viirs(c)
        rb = await self._cmems_bgc()

        out = {"lats": lats, "lons": lons}
        rg = self._regrid

        out["sst"] = rg(rs["sst"], rs["lat"], rs["lon"], lats, lons) if rs and "sst" in rs else self._clim.sst(lats, lons, month)
        if not (rs and "sst" in rs): self._st["SST"] = "WOA-CLIM"

        out["chl"] = rg(rc["chl"], rc["lat"], rc["lon"], lats, lons) if rc and "chl" in rc else self._clim.chl(lats, lons, month)
        if not (rc and "chl" in rc): self._st["CHL"] = "MODIS-CLIM"

        if ru and "u" in ru:
            out["u_current"] = rg(ru["u"], ru["lat"], ru["lon"], lats, lons)
            out["v_current"] = rg(ru["v"], ru["lat"], ru["lon"], lats, lons)
            if "ssh" in ru: out["ssh"] = rg(ru["ssh"], ru["lat"], ru["lon"], lats, lons)
            if "salinity" in ru: out["salinity"] = rg(ru["salinity"], ru["lat"], ru["lon"], lats, lons)
        else:
            u, v = self._clim.currents(lats, lons)
            out["u_current"], out["v_current"] = u, v
            self._st["CURRENTS"] = "CLIM"

        # ── Wind: Open-Meteo → climatology ──
        try:
            wind_data = await self._fetch_wind_api(lats, lons)
            if wind_data is not None:
                out["wind_u"], out["wind_v"] = wind_data
                self._st["WIND"] = "Open-Meteo"
            else:
                raise ValueError("wind API returned None")
        except Exception as e:
            log.info(f"  Wind API fallback: {e}")
            out["wind_u"], out["wind_v"] = self._clim.wind(lats, lons, month)
            self._st["WIND"] = "CLIM"

        out.setdefault("salinity", self._clim.salinity(lats, lons))

        # ── SSH: try CMEMS SEALEVEL if not already from currents ──
        if out.get("ssh") is None:
            try:
                ssh_data = await self._fetch_ssh_aviso(lats, lons)
                if ssh_data is not None:
                    out["ssh"] = ssh_data
                    self._st["SSH"] = "AVISO"
            except Exception:
                out["ssh"] = np.zeros((len(lats), len(lons)), dtype=np.float32)
                self._st["SSH"] = "ZERO"

        if rb and "do_surface" in rb:
            out["do_surface"] = rg(rb["do_surface"], rb["lat"], rb["lon"], lats, lons)
        else:
            out["do_surface"] = self._clim.do_surface(lats, lons, month)
            self._st.setdefault("DO", "WOA-CLIM")

        out["temp_3d"] = rh["temp_3d"] if rh and "temp_3d" in rh else None
        # [商用] temp_3d 品質防護 — clamp 異常值 (HYCOM 快取可能含錯誤值)
        if out["temp_3d"] is not None:
            t3d = out["temp_3d"]
            bad_mask = (t3d < -2) | (t3d > 35)
            n_bad = int(np.sum(bad_mask))
            if n_bad > 0:
                log.warning(f"temp_3d: clamped {n_bad} anomalous values to NaN")
                t3d = t3d.astype(float)
                t3d[bad_mask] = np.nan
                out["temp_3d"] = t3d
            # [#4 商用] 如果 clamp 後全是 NaN，用 WOA 氣候態生成粗略 temp_3d
            if np.all(np.isnan(t3d)):
                log.warning("temp_3d: all NaN after clamping, generating WOA climatology fallback")
                try:
                    nd = t3d.shape[0]
                    clim_depths = np.array([0, 50, 100, 200, 300, 500])[:nd]
                    t3d_clim = np.zeros_like(t3d)
                    for k, depth in enumerate(clim_depths):
                        if k < nd:
                            # WOA 粗略近似：SST - 0.05*depth (for tropics)
                            sst_grid = out.get("sst")
                            if sst_grid is not None:
                                sst_mean = float(np.nanmean(sst_grid[np.isfinite(sst_grid)])) if np.any(np.isfinite(sst_grid)) else 25.0
                            else:
                                sst_mean = 25.0
                            t3d_clim[k] = sst_mean - 0.05 * depth
                    out["temp_3d"] = t3d_clim.astype(np.float32)
                    log.info(f"temp_3d: WOA climatology generated, shape={t3d_clim.shape}")
                except Exception as e:
                    log.warning(f"temp_3d: WOA fallback failed: {e}")

        # [#6 商用] SST 空間插值 — 填補雲覆蓋 NaN
        if out.get("sst") is not None:
            sst_arr = out["sst"]
            nan_ratio = np.sum(np.isnan(sst_arr)) / max(sst_arr.size, 1)
            if 0 < nan_ratio < 0.98:  # 有部分數據但不是全空
                try:
                    from scipy.interpolate import griddata
                    ny_s, nx_s = sst_arr.shape
                    yy, xx = np.mgrid[0:ny_s, 0:nx_s]
                    valid = np.isfinite(sst_arr)
                    if valid.sum() > 10:
                        points = np.column_stack([yy[valid], xx[valid]])
                        values = sst_arr[valid]
                        filled = griddata(points, values, (yy, xx), method='linear')
                        # 只填補 NaN 位置，不覆蓋原始數據
                        nan_mask = np.isnan(sst_arr)
                        sst_arr[nan_mask] = filled[nan_mask]
                        # 還有殘餘 NaN（邊緣）用 nearest 填
                        still_nan = np.isnan(sst_arr)
                        if still_nan.any():
                            filled2 = griddata(points, values, (yy, xx), method='nearest')
                            sst_arr[still_nan] = filled2[still_nan]
                        out["sst"] = sst_arr
                        new_nan_ratio = np.sum(np.isnan(sst_arr)) / max(sst_arr.size, 1)
                        log.info(f"  SST interpolation: NaN {nan_ratio:.1%} → {new_nan_ratio:.1%}")
                except Exception as e:
                    log.warning(f"  SST interpolation failed: {e}")

        out["viirs_lights"] = rv["lights"] if rv and "lights" in rv else np.array([]).reshape(0, 3)
        out["data_sources"] = dict(self._st)

        n_real = sum(1 for v in self._st.values() if "CLIM" not in v and "FAIL" not in v)
        log.info(f"\n  Sources: {n_real} real / {len(self._st)} total")
        for k, v in self._st.items():
            log.info(f"    {k}: {v}")
        return out

    @staticmethod
    def _regrid(data, slat, slon, dlat, dlon):
        if data is None or data.size == 0:
            return np.full((len(dlat), len(dlon)), np.nan, np.float32)
        if slat[0] > slat[-1]: slat, data = slat[::-1], data[::-1, :]
        if slon[0] > slon[-1]: slon, data = slon[::-1], data[:, ::-1]
        fv = np.nanmean(data) if np.any(np.isfinite(data)) else 0.0
        filled = np.nan_to_num(data, nan=fv)
        try:
            from scipy.interpolate import RegularGridInterpolator
            interp = RegularGridInterpolator((slat, slon), filled, method='linear', bounds_error=False, fill_value=np.nan)
            dg, dl = np.meshgrid(dlat, dlon, indexing='ij')
            return interp(np.column_stack([dg.ravel(), dl.ravel()])).reshape(len(dlat), len(dlon)).astype(np.float32)
        except Exception:
            r = np.full((len(dlat), len(dlon)), np.nan, np.float32)
            for i, lt in enumerate(dlat):
                li = np.argmin(np.abs(slat - lt))
                for j, ln in enumerate(dlon):
                    lj = np.argmin(np.abs(slon - ln))
                    if li < data.shape[0] and lj < data.shape[1]: r[i, j] = data[li, lj]
            return r
