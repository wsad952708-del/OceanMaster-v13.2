"""
OceanMaster v12 — 基礎架構品質模組  # [v12-enhance]
====================================
[v11] Cache TTL Strategy
[v11] Tenacity Retry Decorators
[v11] Unified Pipeline Bridge
[v12] CircuitBreaker — 外部 API 斷路保護
[v12] Cache Quality Check — 拒絕 all-NaN 快取

整合 main_v10_3.py 和 commercial_core_v2.py 的統一接口。
"""

import numpy as np
import logging
import time
import hashlib
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable, TypeVar
from functools import wraps
import threading  # [v12-enhance] for CircuitBreaker

log = logging.getLogger("OceanMaster.Infra")

T = TypeVar("T")


# ═══════════════════════════════════════════════════
#  Cache TTL Strategy
# ═══════════════════════════════════════════════════

class CacheTTL:
    """
    智能快取管理器 — 依數據類型設定不同 TTL

    TTL 策略:
      - SST (OISST): 24h (每日更新)
      - Chl-a (VIIRS): 24h (每日合成)
      - SSH/海流 (CMEMS): 48h (NRT 延遲 ~2 天)
      - 溶氧 (WOA): 30 天 (氣候態)
      - 地形 (GEBCO): 永久 (靜態)
      - ONI: 7 天 (月更新)
      - Argo: 6h (即時剖面)
      - 漁船 AIS: 1h (近即時)

    快取鍵 = hash(數據類型, 區域, 日期, 參數)
    """

    # TTL 配置 (秒)
    TTL_CONFIG = {
        "sst":          24 * 3600,      # 24h
        "chl":          24 * 3600,      # 24h
        "ssh":          48 * 3600,      # 48h
        "current":      48 * 3600,      # 48h
        "do":           30 * 24 * 3600, # 30 天
        "woa":          30 * 24 * 3600, # 30 天
        "gebco":        365 * 24 * 3600, # 1 年 (靜態)
        "oni":          7 * 24 * 3600,  # 7 天
        "argo":         6 * 3600,       # 6h
        "ais":          1 * 3600,       # 1h
        "cmems_bgc":    24 * 3600,      # 24h
        "lunar":        24 * 3600,      # 24h
        "default":      12 * 3600,      # 12h fallback
    }

    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: Dict[str, Any] = {}
        self._memory_timestamps: Dict[str, float] = {}

    def _make_key(self, data_type: str, **params) -> str:
        """生成快取鍵"""
        key_parts = [data_type] + [f"{k}={v}" for k, v in sorted(params.items())]
        key_str = "|".join(str(p) for p in key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()[:16]

    def get(self, data_type: str, **params) -> Optional[Any]:
        """
        從快取取得數據

        Returns:
            cached data if valid, None if expired/missing
        """
        key = self._make_key(data_type, **params)
        ttl = self.TTL_CONFIG.get(data_type, self.TTL_CONFIG["default"])

        # 1. 記憶體快取 (最快)
        if key in self._memory_cache:
            age = time.time() - self._memory_timestamps.get(key, 0)
            if age < ttl:
                log.debug(f"  Cache HIT (memory): {data_type} key={key} age={age:.0f}s")
                return self._memory_cache[key]
            else:
                del self._memory_cache[key]
                del self._memory_timestamps[key]

        # 2. 磁碟快取
        cache_file = self.cache_dir / f"{data_type}_{key}.npy"
        meta_file = self.cache_dir / f"{data_type}_{key}.json"

        if cache_file.exists() and meta_file.exists():
            with open(meta_file, "r") as f:
                meta = json.load(f)
            age = time.time() - meta.get("timestamp", 0)

            if age < ttl:
                data = np.load(str(cache_file), allow_pickle=True)
                # 更新記憶體快取
                self._memory_cache[key] = data
                self._memory_timestamps[key] = meta["timestamp"]
                log.debug(f"  Cache HIT (disk): {data_type} key={key} age={age:.0f}s")
                return data

        return None

    def put(self, data_type: str, data: Any, **params):
        """
        存入快取

        Parameters:
            data_type: 數據類型 (決定 TTL)
            data: numpy array or serializable data
            **params: 查詢參數
        """
        # [v12-fix] Cache quality check — reject all-NaN data
        if not self._cache_quality_ok(data, data_type):
            return

        key = self._make_key(data_type, **params)
        now = time.time()

        # 記憶體快取
        self._memory_cache[key] = data
        self._memory_timestamps[key] = now

        # 磁碟快取
        try:
            cache_file = self.cache_dir / f"{data_type}_{key}.npy"
            meta_file = self.cache_dir / f"{data_type}_{key}.json"

            np.save(str(cache_file), data, allow_pickle=True)
            with open(meta_file, "w") as f:
                json.dump({
                    "data_type": data_type,
                    "params": {k: str(v) for k, v in params.items()},
                    "timestamp": now,
                    "ttl_seconds": self.TTL_CONFIG.get(data_type, self.TTL_CONFIG["default"]),
                }, f)
        except Exception as e:
            log.warning(f"  Cache write failed: {e}")

    def invalidate(self, data_type: Optional[str] = None):
        """
        清除快取

        Parameters:
            data_type: 指定類型或 None=全部清除
        """
        if data_type:
            # 清除指定類型
            keys_to_remove = [k for k in self._memory_cache
                              if k.startswith(data_type)]
            for k in keys_to_remove:
                del self._memory_cache[k]
                self._memory_timestamps.pop(k, None)

            for f in self.cache_dir.glob(f"{data_type}_*"):
                f.unlink()
            log.info(f"  Cache invalidated: {data_type}")
        else:
            self._memory_cache.clear()
            self._memory_timestamps.clear()
            for f in self.cache_dir.glob("*"):
                if f.is_file():
                    f.unlink()
            log.info("  Cache fully cleared")

    def stats(self) -> Dict[str, Any]:
        """快取統計"""
        disk_files = list(self.cache_dir.glob("*.npy"))
        return {
            "memory_entries": len(self._memory_cache),
            "disk_entries": len(disk_files),
            "disk_size_mb": sum(f.stat().st_size for f in disk_files) / 1e6,
        }

    @staticmethod
    def _cache_quality_ok(data: Any, data_type: str) -> bool:
        """
        [v12-fix] 快取品質檢查

        拒絕:
          - all-NaN numpy arrays (完全無資料)
          - >95% NaN 的資料 (近乎無用)
        """
        if data is None:
            log.warning(f"  [v12-fix] Cache reject: {data_type} is None")
            return False

        if isinstance(data, np.ndarray):
            if data.size == 0:
                log.warning(f"  [v12-fix] Cache reject: {data_type} is empty array")
                return False

            try:
                nan_ratio = np.isnan(data.astype(float)).sum() / data.size
            except (ValueError, TypeError):
                return True  # non-numeric arrays pass

            if nan_ratio >= 1.0:
                log.warning(f"  [v12-fix] Cache reject: {data_type} is 100% NaN")
                return False
            if nan_ratio > 0.95:
                log.warning(
                    f"  [v12-fix] Cache reject: {data_type} is "
                    f"{nan_ratio*100:.1f}% NaN (threshold=95%)"
                )
                return False

        return True


# ═══════════════════════════════════════════════════
#  [v12-enhance] Circuit Breaker — 外部 API 斷路保護
# ═══════════════════════════════════════════════════

class CircuitBreaker:
    """
    三態斷路器 (CLOSED → OPEN → HALF_OPEN → CLOSED)

    - CLOSED:    正常轉發請求
    - OPEN:      連續失敗 ≥ threshold → 拒絕請求 (直接 fallback)
    - HALF_OPEN: recovery_timeout 後嘗試一次請求, 成功 → CLOSED

    用法:
        cb = CircuitBreaker("CMEMS", failure_threshold=5, recovery_timeout=120)
        result = cb.call(fetch_cmems_data, lat, lon)

    或者作為裝飾器:
        @CircuitBreaker.as_decorator("NOAA", failure_threshold=3)
        def fetch_noaa(...):
            ...
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 120.0,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._state = self.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0.0
        self._lock = threading.Lock()
        log.info(f"  CircuitBreaker [{name}]: initialized "
                 f"(threshold={failure_threshold}, timeout={recovery_timeout}s)")

    @property
    def state(self) -> str:
        with self._lock:
            if self._state == self.OPEN:
                elapsed = time.time() - self._last_failure_time
                if elapsed >= self.recovery_timeout:
                    self._state = self.HALF_OPEN
                    log.info(f"  CircuitBreaker [{self.name}]: OPEN → HALF_OPEN")
            return self._state

    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        通過斷路器調用 func。
        OPEN 狀態直接拋出 CircuitBreakerOpenError。
        """
        current_state = self.state

        if current_state == self.OPEN:
            raise CircuitBreakerOpenError(
                f"CircuitBreaker [{self.name}] is OPEN — "
                f"failing fast (failures={self._failure_count})"
            )

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        with self._lock:
            if self._state == self.HALF_OPEN:
                log.info(f"  CircuitBreaker [{self.name}]: HALF_OPEN → CLOSED (recovered)")
            self._state = self.CLOSED
            self._failure_count = 0

    def _on_failure(self):
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            if self._failure_count >= self.failure_threshold:
                self._state = self.OPEN
                log.warning(
                    f"  CircuitBreaker [{self.name}]: CLOSED → OPEN "
                    f"(failures={self._failure_count})"
                )

    def reset(self):
        """手動重設斷路器"""
        with self._lock:
            self._state = self.CLOSED
            self._failure_count = 0
            log.info(f"  CircuitBreaker [{self.name}]: manually reset to CLOSED")

    @classmethod
    def as_decorator(
        cls,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 120.0,
    ):
        """作為裝飾器使用"""
        cb = cls(name, failure_threshold, recovery_timeout)

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return cb.call(func, *args, **kwargs)
            wrapper._circuit_breaker = cb  # expose for testing
            return wrapper
        return decorator


class CircuitBreakerOpenError(Exception):
    """斷路器開啟時拋出的例外"""
    pass


# ═══════════════════════════════════════════════════
#  Retry Decorators (Tenacity-compatible)
# ═══════════════════════════════════════════════════

def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
    on_retry_log: bool = True,
):
    """
    重試裝飾器 — 帶指數退避

    用法:
        @retry_with_backoff(max_retries=3)
        def fetch_data():
            ...

    如果 tenacity 可用，使用 tenacity 實作;
    否則使用內建實作 (零依賴)。

    Parameters:
        max_retries: 最大重試次數
        initial_delay: 初始延遲 (秒)
        backoff_factor: 退避倍數
        exceptions: 要重試的例外類型
        on_retry_log: 是否記錄重試
    """
    try:
        import tenacity

        def decorator(func):
            @tenacity.retry(
                stop=tenacity.stop_after_attempt(max_retries + 1),
                wait=tenacity.wait_exponential(
                    multiplier=initial_delay,
                    exp_base=backoff_factor,
                    max=60,
                ),
                retry=tenacity.retry_if_exception_type(exceptions),
                before_sleep=tenacity.before_sleep_log(log, logging.WARNING)
                    if on_retry_log else None,
                reraise=True,
            )
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        return decorator

    except ImportError:
        # 純 Python 回退實作
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                delay = initial_delay
                last_exc = None

                for attempt in range(max_retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        last_exc = e
                        if attempt < max_retries:
                            if on_retry_log:
                                log.warning(
                                    f"  {func.__name__} 重試 {attempt+1}/{max_retries}: {e}"
                                )
                            time.sleep(delay)
                            delay *= backoff_factor
                        else:
                            raise
                raise last_exc  # type: ignore
            return wrapper
        return decorator


def timeout_guard(seconds: float = 30.0):
    """
    超時保護裝飾器

    用法:
        @timeout_guard(30)
        def slow_operation():
            ...

    注意: 在 Windows 上使用 threading 實作 (signal.alarm 僅限 Unix)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import threading

            result_container = [None]
            error_container = [None]

            def target():
                try:
                    result_container[0] = func(*args, **kwargs)
                except Exception as e:
                    error_container[0] = e

            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout=seconds)

            if thread.is_alive():
                log.error(f"  {func.__name__} 超時 ({seconds}s)")
                raise TimeoutError(f"{func.__name__} timed out after {seconds}s")

            if error_container[0] is not None:
                raise error_container[0]

            return result_container[0]
        return wrapper
    return decorator


# ═══════════════════════════════════════════════════
#  Unified Pipeline Bridge (v11)
# ═══════════════════════════════════════════════════

class V11PipelineBridge:
    """
    統一管線橋接器

    整合 main_v10_3.py 中的 OceanMasterPipeline 和
    commercial_core_v2.py 中的引擎，
    加上 v11 所有新模組。

    用法:
        bridge = V11PipelineBridge()
        bridge.enrich_ocean_data(ocean_data, species="bigeye")
    """

    def __init__(self):
        self._cache = CacheTTL()
        self._modules = {}
        self._load_v11_modules()

    def _load_v11_modules(self):
        """懶加載 v11 模組"""
        try:
            from engine.lunar_model import LunarPhaseEngine
            self._modules["lunar"] = LunarPhaseEngine()
        except ImportError:
            pass

        try:
            from engine.zooplankton_proxy import ZooplanktonProxy
            self._modules["zoo"] = ZooplanktonProxy()
        except ImportError:
            pass

        try:
            from engine.omz_model import OMZModel
            self._modules["omz"] = OMZModel()
        except ImportError:
            pass

        try:
            from engine.data_sources import ONIFetcher
            self._modules["oni"] = ONIFetcher()
        except ImportError:
            pass

        try:
            from engine.data_sources import GEBCOProximityCalculator
            self._modules["gebco"] = GEBCOProximityCalculator()
        except ImportError:
            pass

        log.info(f"  v11 Bridge: loaded {list(self._modules.keys())}")

    @retry_with_backoff(max_retries=2, initial_delay=0.5)
    def enrich_ocean_data(
        self,
        ocean_data: Dict[str, Any],
        species: str = "skipjack",
        date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        用 v11 新模組豐富 ocean_data 字典

        加入:
          - lunar_cpue_modifier
          - zooplankton_index
          - spawning_season
          - enso_oni
          - omz_compression (if DO available)
          - dist_to_seamount, dist_to_shelf_break

        Parameters:
            ocean_data: 原始海洋數據字典
            species: 目標物種
            date: 分析日期

        Returns:
            enriched ocean_data (原始 dict 被修改並返回)
        """
        if date is None:
            date = datetime.now()

        # 1. 月相
        lunar = self._modules.get("lunar")
        if lunar and "lunar_cpue_modifier" not in ocean_data:
            phase = lunar.compute_moon_phase(date)
            modifier = lunar.compute_lunar_cpue_modifier(
                species, "longline", phase["illumination"]
            )
            ocean_data["lunar_cpue_modifier"] = modifier
            ocean_data["moon_phase"] = phase["illumination"]

        # 2. 浮游動物
        zoo = self._modules.get("zoo")
        if zoo and "zooplankton_index" not in ocean_data:
            npp = ocean_data.get("npp")
            sst = ocean_data.get("sst")
            if npp is not None and sst is not None:
                zoo_idx = zoo.estimate_from_npp_lag(npp, sst=sst)
                ocean_data["zooplankton_index"] = zoo_idx

        # 3. 產卵季節
        try:
            from engine.species_params import SPECIES
            sp_info = SPECIES.get(species, {})
            spawning_months = sp_info.get("spawning_months", [])
            ocean_data["spawning_season"] = 1.0 if date.month in spawning_months else 0.0
        except ImportError:
            pass

        # 4. ENSO ONI
        oni_fetcher = self._modules.get("oni")
        if oni_fetcher and "enso_oni" not in ocean_data:
            oni_val = oni_fetcher.get_current_oni(date)
            ocean_data["enso_oni"] = oni_val

        # 5. GEBCO 地形
        gebco = self._modules.get("gebco")
        if gebco and "dist_to_seamount" not in ocean_data:
            sst = ocean_data.get("sst")
            if sst is not None:
                ny, nx = sst.shape
                lat_arr = ocean_data.get("lat", np.linspace(5, 35, ny))
                lon_arr = ocean_data.get("lon", np.linspace(120, 175, nx))
                lat_grid, lon_grid = np.meshgrid(lat_arr, lon_arr, indexing="ij")

                geo = gebco.compute_all_geomorphic_features(lat_grid, lon_grid)
                ocean_data["dist_to_seamount"] = geo["dist_to_seamount"]
                ocean_data["dist_to_shelf_break"] = geo["dist_to_shelf_break"]

        return ocean_data
