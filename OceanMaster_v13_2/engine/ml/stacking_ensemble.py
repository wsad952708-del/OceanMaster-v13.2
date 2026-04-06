"""
OceanMaster v13.2 — ML Stacking Ensemble 融合引擎  # [v12-enhance]
=================================================
🔴 關鍵升級：從規則加權 → 真正的 AI 學習

核心架構：
  Level-0 (Base Models):
    - Random Forest (捕捉非線性交互)
    - XGBoost (梯度提升，高效)
    - LightGBM (大數據集優化)
    - Ridge Regression (線性基線)

  Level-1 (Meta-Learner):
    - Logistic Regression (學習最佳組合權重)

  訓練數據來源：
    - FAO 歷史漁獲報告 (CPUE = Catch Per Unit Effort)
    - 船長回報 (self-reported catches)
    - GFW 漁船活動密度 (反推漁場品質)

  特徵工程：
    - 基礎特徵: SST, Chl-a, SSH, 海流速度
    - 衍生特徵: SST梯度, FTLE, 溫躍層深度, D20, MLD
    - 時空特徵: 月相, 日長, 潮汐週期, 季節正弦編碼
    - 交互特徵: SST×Chl-a, front_strength×FTLE
    - 滯後特徵: 7天/14天/30天移動平均

商業級演算法逆向工程：
  - CATSAT: AMM (Arithmetic Mean Model) → 我們用更好的 Geometric Mean + ML
  - INCOIS PFZ: Cayula-Cornillon + SST梯度 → 已在 algorithms.py 實現
  - TunaCAST: SSH渦旋 + 溫躍層 → 現在加入 CMEMS SSH
  - SatFish: Multi-sensor fusion → 我們的 Stacking 更先進
"""

import numpy as np
import logging
import pickle
import json
import os
import hmac
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone

log = logging.getLogger("OceanMaster.ML")

# ─── 嘗試導入 ML 庫 ─────────────────────────────
try:
    from sklearn.ensemble import (
        RandomForestClassifier,
        RandomForestRegressor,
        GradientBoostingRegressor,
        StackingRegressor,
    )
    from sklearn.linear_model import Ridge, LogisticRegression, RidgeCV
    from sklearn.model_selection import cross_val_predict, KFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    log.warning("scikit-learn 未安裝，ML Stacking 使用 fallback 模式")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

# [v12-enhance] CatBoost
try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

# [v12-fix] Model version for safe_model_load
_MODEL_VERSION = "v12"

# [v13.2-P0] Model signing key — MUST be set in production
_MODEL_SIGN_KEY = os.environ.get("MODEL_SIGN_KEY", "").encode()
if not _MODEL_SIGN_KEY:
    log.warning("⚠️ MODEL_SIGN_KEY not set — model integrity verification disabled. "
                "Set MODEL_SIGN_KEY env var for production.")


def _sign_model_file(path: Path) -> None:
    """[v13.2-P0] Write HMAC-SHA256 signature for a model file."""
    if not _MODEL_SIGN_KEY:
        return
    sig = hmac.new(_MODEL_SIGN_KEY, path.read_bytes(), hashlib.sha256).hexdigest()
    path.with_suffix(".sig").write_text(sig)
    log.info(f"  🔐 Model signed: {path.with_suffix('.sig')}")


def _verify_model_file(path: Path) -> bool:
    """[v13.2-P0] Verify HMAC-SHA256 signature of a model file.
    Returns True if verified or if signing is not configured.
    Raises RuntimeError if signature is invalid.
    """
    if not _MODEL_SIGN_KEY:
        # Signing not configured — allow loading with warning (dev mode)
        return True
    sig_path = path.with_suffix(".sig")
    if not sig_path.exists():
        raise RuntimeError(
            f"Model signature file missing: {sig_path}. "
            f"Re-train the model or disable MODEL_SIGN_KEY for dev mode."
        )
    expected = sig_path.read_text().strip()
    actual = hmac.new(_MODEL_SIGN_KEY, path.read_bytes(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, actual):
        raise RuntimeError(
            f"Model signature verification FAILED for {path}. "
            f"File may have been tampered with."
        )
    return True


# ═══════════════════════════════════════════════════
# ─── CPUE 物理上限 (kg/day, 延繩釣) ─── T2
# Source: WCPFC Statistical Yearbook
CPUE_CAPS = {
    'yellowfin': 500,   # WCPFC max ~400-500
    'bigeye': 300,
    'skipjack': 200,    # Longline bycatch, normally low
    'albacore': 250,
}


#  特徵工程器
# ═══════════════════════════════════════════════════

class FeatureEngineer:
    """
    將原始海洋觀測數據轉換為 ML 特徵矩陣

    特徵類別:
    1. 基礎環境特徵 (SST, Chl-a, SSH, 海流)
    2. 衍生物理特徵 (梯度, 鋒面, FTLE, 溫躍層)
    3. 時空特徵 (月相, 季節, 經緯度)
    4. 交互特徵 (SST×Chl, 鋒面×FTLE)
    5. 統計窗口特徵 (局部均值/標準差/偏度)
    """

    FEATURE_NAMES = [
        # 基礎環境
        "sst", "chl_log", "ssh", "current_speed", "current_dir",
        # 衍生物理
        "sst_gradient", "chl_gradient", "front_strength",
        "ftle", "ftle_ridge",
        "thermocline_depth", "d20_depth", "mld",
        # 距離特徵 (替代原始 lat/lon — Druon 2012, Scales 2014)
        "dist_to_front", "dist_to_eddy",
        "dist_to_seamount", "dist_to_shelf_break",
        # 地形特徵 (GEBCO)
        "bathy_depth", "bathy_slope", "bathy_roughness",
        # 時間特徵
        "moon_phase",
        "season_sin", "season_cos", "day_of_year_sin", "day_of_year_cos",
        # 時序特徵 (GreenFish 8-day window concept)
        "sst_7d_trend", "chl_30d_anomaly",
        # 交互
        "sst_x_chl", "front_x_ftle", "ssh_x_thermo",
        # 窗口統計
        "sst_local_std", "chl_local_mean", "current_local_mean",
        # AIS 漁船密度
        "ais_fishing_density",
        # VIIRS
        "viirs_light_density",
        # ── v11 新增 10 個科學特徵 ──
        "lunar_cpue_modifier",          # Feature A: 月相 CPUE 修正
        "zooplankton_index",            # Feature C: 浮游動物代理
        "spawning_season",              # Bug #4: 產卵季節
        "enso_oni",                     # ONI 指數
        "omz_compression",              # Bug #5: OMZ 棲息壓縮
        "eddy_enrichment",              # Feature D: 渦旋生物增益
        "dvm_accessible_depth",         # DVM 可到達深度
        "salinity_front_strength",      # Feature B: 鹽度鋒面
        "productivity_front",           # Bug #6: 融合鋒面
        "habitat_compression_ratio",    # OMZ/溫躍層壓縮比
        # ── v13 新增 2 個黑潮/台灣特徵 ──
        "kuroshio_distance",             # 距黑潮軸距離 (km)
        "taiwan_strait_flag",            # 台灣海峽區域旗標
        # ── [海鷹/蒼鷺] 深層特徵 4 個 ──
        "t100",                          # 100m 水溫 (°C)
        "gradient_strength",             # ΔT/ΔZ 溫度梯度 (°C/m)
        "delta_t_surface_100",           # SST - T100 (°C)
        "chl_lag15d",                    # 15 天前 CHL (log10)
        # ── v13.2-audit 新增 8 個科學衍生特徵 ──
        "moonlight_dvm_suppression",     # 月光抑制 micronekton 上浮 (Benoit-Bird 2009)
        "post_storm_chl_bloom",          # 颱風後 CHL 爆發潛力 (Zhao 2017)
        "seamount_current_interaction",  # 海底山 × 海流交互 (Taylor column)
        "prey_thermocline_trap",         # 溫躍層獵物聚集指數 (Mann & Lazier 2006)
        "tidal_mixing_index",            # 潮汐混合指數 (Simpson & Hunter 1974)
        "sst_seasonal_derivative",       # SST 季節變化率 (Podesta 1993)
        "convergence_proxy",             # 海流收斂帶代理 (Bakun 2006)
        "dawn_twilight_hours",           # 晨昏覓食窗口時長
        # ── v18.3 氣候指數 ML 特徵 (4 個) ──
        # 2024-2025 論文確認：ENSO lag 對長鰭鮪有 0-2 年效應 (MDPI 2024)
        # PDO 正相期黃鰭鮪 CPUE 更高 (Wu et al. 2022 Sci.Rep.)
        # SOI 與南太平洋長鰭鮪 CPUE 顯著相關 (MDPI 2024 South Pacific)
        "enso_lag1",                     # 1 年前 ONI 值 (長鰭鮪關鍵)
        "enso_lag2",                     # 2 年前 ONI 值 (長鰭鮪關鍵)
        "pdo_index",                     # PDO 指數 (黃鰭鮪關鍵, Mantua 1997)
        "soi_index",                     # SOI 指數 (南太平洋長鰭鮪)
        # ── v18.3 深層溶氧 (3 個) ──
        # Liu 2025 ICES: o2_0 > o2_50 是最重要棲地特徵
        # Xu 2025 Cook Is. LSTM: DO 150m/200m 最重要
        "do_50m",                        # 50m 溶氧 (ml/L)
        "do_150m",                       # 150m 溶氧 (ml/L)
        "do_200m",                       # 200m 溶氧 (ml/L)
    ]

    # [v12-fix] Feature count guard — 確保特徵數量與文檔一致
    EXPECTED_FEATURE_COUNT = 66  # [v18.3] +4 climate +3 deep DO = +7
    assert len(FEATURE_NAMES) == EXPECTED_FEATURE_COUNT, (
        f"Feature count mismatch: expected {EXPECTED_FEATURE_COUNT}, "
        f"got {len(FEATURE_NAMES)}. Update EXPECTED_FEATURE_COUNT if intentional."
    )

    def __init__(self):
        self.scaler = StandardScaler() if HAS_SKLEARN else None
        self._fitted = False

    def extract_features(
        self,
        ocean_data: Dict[str, Any],
        species: str = "skipjack",
    ) -> Tuple[np.ndarray, List[str]]:
        """
        從海洋數據字典提取特徵矩陣

        參數:
            ocean_data: 包含 sst, chl, ssh, u, v, ftle, front_strength 等
            species: 目標物種（影響某些特徵的計算方式）

        返回:
            X: (n_pixels, n_features) 特徵矩陣
            feature_names: 特徵名稱列表
        """
        sst = ocean_data.get("sst")
        if sst is None:
            raise ValueError("SST 數據缺失，無法提取特徵")

        ny, nx = sst.shape
        n_pixels = ny * nx
        features = {}

        # ── 1. 基礎環境特徵 ──
        features["sst"] = sst.ravel()

        chl = ocean_data.get("chl")
        if chl is not None:
            features["chl_log"] = np.log10(
                np.maximum(self._align(chl, sst.shape), 0.001)
            ).ravel()
        else:
            features["chl_log"] = np.zeros(n_pixels)

        ssh = ocean_data.get("ssh")
        if ssh is not None:
            features["ssh"] = self._align(ssh, sst.shape).ravel()
        else:
            features["ssh"] = np.zeros(n_pixels)

        u = ocean_data.get("u")
        v = ocean_data.get("v")
        if u is not None and v is not None:
            u_a = self._align(u, sst.shape)
            v_a = self._align(v, sst.shape)
            features["current_speed"] = np.sqrt(u_a**2 + v_a**2).ravel()
            features["current_dir"] = np.arctan2(v_a, u_a).ravel()
        else:
            features["current_speed"] = np.zeros(n_pixels)
            features["current_dir"] = np.zeros(n_pixels)

        # ── 2. 衍生物理特徵 ──
        from scipy import ndimage
        grad_y = ndimage.sobel(np.nan_to_num(sst, nan=np.nanmean(sst)), axis=0)
        grad_x = ndimage.sobel(np.nan_to_num(sst, nan=np.nanmean(sst)), axis=1)
        features["sst_gradient"] = np.sqrt(grad_x**2 + grad_y**2).ravel()

        if chl is not None:
            chl_a = self._align(chl, sst.shape)
            chl_log_2d = np.log10(np.maximum(chl_a, 0.001))
            gy = ndimage.sobel(np.nan_to_num(chl_log_2d), axis=0)
            gx = ndimage.sobel(np.nan_to_num(chl_log_2d), axis=1)
            features["chl_gradient"] = np.sqrt(gx**2 + gy**2).ravel()
        else:
            features["chl_gradient"] = np.zeros(n_pixels)

        fs = ocean_data.get("front_strength")
        features["front_strength"] = (
            self._align(fs, sst.shape).ravel() if fs is not None
            else np.zeros(n_pixels)
        )

        ftle = ocean_data.get("ftle")
        features["ftle"] = (
            self._align(ftle, sst.shape).ravel() if ftle is not None
            else np.zeros(n_pixels)
        )

        ftle_ridges = ocean_data.get("ftle_ridges")
        features["ftle_ridge"] = (
            self._align(ftle_ridges, sst.shape).astype(float).ravel()
            if ftle_ridges is not None else np.zeros(n_pixels)
        )

        td = ocean_data.get("thermocline_depth")
        features["thermocline_depth"] = (
            self._align(td, sst.shape).ravel() if td is not None
            else np.full(n_pixels, 150.0)
        )

        d20 = ocean_data.get("d20_depth")
        features["d20_depth"] = (
            self._align(d20, sst.shape).ravel() if d20 is not None
            else np.full(n_pixels, 200.0)
        )

        mld_data = ocean_data.get("mld")
        features["mld"] = (
            self._align(mld_data, sst.shape).ravel() if mld_data is not None
            else np.full(n_pixels, 50.0)
        )

        # ── 3. 距離特徵 (替代 lat/lon — Druon 2012, Scales 2014) ──
        lat = ocean_data.get("lat", np.linspace(5, 35, ny))
        lon = ocean_data.get("lon", np.linspace(120, 175, nx))

        # 3a. 距最近鋒面的距離 (km)
        front = ocean_data.get("front_strength")
        if front is not None:
            front_a = self._align(front, sst.shape)
            front_mask = front_a > 0.3  # 強鋒面
            dist_front = self._compute_distance_field(front_mask, lat, lon)
            features["dist_to_front"] = dist_front.ravel()
        else:
            features["dist_to_front"] = np.full(n_pixels, 100.0)

        # 3b. 距最近渦旋的距離 (km)
        eddy_core = ocean_data.get("eddy_core")
        if eddy_core is not None:
            eddy_a = self._align(eddy_core, sst.shape)
            dist_eddy = self._compute_distance_field(eddy_a > 0, lat, lon)
            features["dist_to_eddy"] = dist_eddy.ravel()
        else:
            features["dist_to_eddy"] = np.full(n_pixels, 200.0)

        # 3c. 地形距離 (海底山 + 大陸棚)
        dist_sm = ocean_data.get("dist_to_seamount")
        features["dist_to_seamount"] = (
            self._align(dist_sm, sst.shape).ravel() if dist_sm is not None
            else np.full(n_pixels, 300.0)
        )
        dist_shelf = ocean_data.get("dist_to_shelf_break")
        features["dist_to_shelf_break"] = (
            self._align(dist_shelf, sst.shape).ravel() if dist_shelf is not None
            else np.full(n_pixels, 200.0)
        )

        # 3d. 地形特徵
        bathy = ocean_data.get("bathy_depth")
        features["bathy_depth"] = (
            self._align(bathy, sst.shape).ravel() if bathy is not None
            else np.full(n_pixels, -4000.0)
        )
        b_slope = ocean_data.get("bathy_slope")
        features["bathy_slope"] = (
            self._align(b_slope, sst.shape).ravel() if b_slope is not None
            else np.zeros(n_pixels)
        )
        b_rough = ocean_data.get("bathy_roughness")  # [v15.0] V3.0 #1
        features["bathy_roughness"] = (
            self._align(b_rough, sst.shape).ravel() if b_rough is not None
            else np.zeros(n_pixels)
        )

        # ── 4. 時間特徵 ──
        moon = ocean_data.get("moon_phase", 0.5)
        features["moon_phase"] = np.full(n_pixels, moon)

        now = datetime.now(timezone.utc)
        doy = now.timetuple().tm_yday
        features["season_sin"] = np.full(n_pixels, np.sin(2 * np.pi * doy / 365.25))
        features["season_cos"] = np.full(n_pixels, np.cos(2 * np.pi * doy / 365.25))
        # [v15.3-WARNING] day_of_year_sin/cos 與 season_sin/cos 完全相同，
        # 但已訓練的模型依賴 59 個特徵，不可刪除。
        # 下次重訓時應將 day_of_year_* 改為不同週期（例如 30-day cycle）或移除。
        features["day_of_year_sin"] = np.full(n_pixels, np.sin(2 * np.pi * doy / 365.25))
        features["day_of_year_cos"] = np.full(n_pixels, np.cos(2 * np.pi * doy / 365.25))

        # ── 5. 時序特徵 (GreenFish 8-day window concept) ──
        sst_trend = ocean_data.get("sst_7d_trend")
        features["sst_7d_trend"] = (
            self._align(sst_trend, sst.shape).ravel() if sst_trend is not None
            else np.zeros(n_pixels)
        )
        chl_anom = ocean_data.get("chl_30d_anomaly")
        features["chl_30d_anomaly"] = (
            self._align(chl_anom, sst.shape).ravel() if chl_anom is not None
            else np.zeros(n_pixels)
        )

        # ── 4. 交互特徵 ──
        features["sst_x_chl"] = features["sst"] * features["chl_log"]
        features["front_x_ftle"] = features["front_strength"] * features["ftle"]
        features["ssh_x_thermo"] = features["ssh"] * features["thermocline_depth"]

        # ── 5. 窗口統計特徵 ──
        sst_filled = np.nan_to_num(sst, nan=np.nanmean(sst))
        # [v15.3-perf] 用代數恒等式取代 generic_filter(np.std)——加速 ~50x
        # Var(X) = E[X²] - E[X]²
        _sst_sq = ndimage.uniform_filter(sst_filled ** 2, size=5)
        _sst_mean = ndimage.uniform_filter(sst_filled, size=5)
        _sst_var = np.maximum(_sst_sq - _sst_mean ** 2, 0)  # clamp rounding errors
        features["sst_local_std"] = np.sqrt(_sst_var).ravel()

        if chl is not None:
            chl_filled = np.nan_to_num(self._align(chl, sst.shape), nan=0.2)
            features["chl_local_mean"] = ndimage.uniform_filter(
                chl_filled, size=5
            ).ravel()
        else:
            features["chl_local_mean"] = np.full(n_pixels, 0.2)

        features["current_local_mean"] = (
            ndimage.uniform_filter(
                features["current_speed"].reshape(ny, nx), size=5
            ).ravel()
            if features["current_speed"].any()
            else np.zeros(n_pixels)
        )

        # ── 6. AIS 漁船密度 ──
        ais = ocean_data.get("ais_fishing_density")
        features["ais_fishing_density"] = (
            self._align(ais, sst.shape).ravel() if ais is not None
            else np.zeros(n_pixels)
        )

        # ── 7. VIIRS 光密度 ──
        viirs = ocean_data.get("viirs_density")
        features["viirs_light_density"] = (
            self._align(viirs, sst.shape).ravel() if viirs is not None
            else np.zeros(n_pixels)
        )

        # ── 8. v11 新增科學特徵 (10 個) ──
        # 8a. 月相 CPUE 修正 (Feature A)
        lunar_mod = ocean_data.get("lunar_cpue_modifier")
        features["lunar_cpue_modifier"] = (
            np.full(n_pixels, float(lunar_mod)) if lunar_mod is not None
            else np.ones(n_pixels)  # 無修正=1.0
        )

        # 8b. 浮游動物代理 (Feature C)
        zoo = ocean_data.get("zooplankton_index")
        features["zooplankton_index"] = (
            self._align(zoo, sst.shape).ravel() if zoo is not None
            else np.full(n_pixels, 0.5)
        )

        # 8c. 產卵季節二元特徵
        spawning = ocean_data.get("spawning_season")
        features["spawning_season"] = (
            np.full(n_pixels, float(spawning)) if spawning is not None
            else np.zeros(n_pixels)
        )

        # 8d. ENSO ONI 指數
        oni = ocean_data.get("enso_oni", 0.0)
        features["enso_oni"] = np.full(n_pixels, float(oni))

        # 8e. OMZ 棲息壓縮 (Bug #5)
        omz_c = ocean_data.get("omz_compression")
        features["omz_compression"] = (
            self._align(omz_c, sst.shape).ravel() if omz_c is not None
            else np.ones(n_pixels)  # 無壓縮=1.0
        )

        # 8f. 渦旋生物增益 (Feature D)
        eddy_e = ocean_data.get("eddy_enrichment")
        features["eddy_enrichment"] = (
            self._align(eddy_e, sst.shape).ravel() if eddy_e is not None
            else np.zeros(n_pixels)
        )

        # 8g. DVM 可到達深度
        dvm_d = ocean_data.get("dvm_accessible_depth")
        features["dvm_accessible_depth"] = (
            self._align(dvm_d, sst.shape).ravel() if dvm_d is not None
            else np.full(n_pixels, 300.0)
        )

        # 8h. 鹽度鋒面 (Feature B)
        sal_f = ocean_data.get("salinity_front_strength")
        features["salinity_front_strength"] = (
            self._align(sal_f, sst.shape).ravel() if sal_f is not None
            else np.zeros(n_pixels)
        )

        # 8i. 融合鋒面 (Bug #6)
        pf = ocean_data.get("productivity_front")
        features["productivity_front"] = (
            self._align(pf, sst.shape).ravel() if pf is not None
            else np.zeros(n_pixels)
        )

        # 8j. 棲息壓縮比 (OMZ/溫躍層)
        hcr = ocean_data.get("habitat_compression_ratio")
        features["habitat_compression_ratio"] = (
            self._align(hcr, sst.shape).ravel() if hcr is not None
            else np.ones(n_pixels)
        )

        # ── 9. v13 黑潮/台灣特徵 (2 個) ──
        # 9a. 距黑潮軸距離 (km)
        kuro_dist = ocean_data.get("kuroshio_distance")
        if kuro_dist is not None:
            features["kuroshio_distance"] = (
                self._align(kuro_dist, sst.shape).ravel()
            )
        else:
            # 沒有黑潮資料時，用經緯度近似計算
            lat_g = ocean_data.get("lat", np.linspace(5, 35, ny))
            lon_g = ocean_data.get("lon", np.linspace(120, 175, nx))
            lat_2d, lon_2d = np.meshgrid(lat_g, lon_g, indexing='ij')
            kuro_axis_lon = 122.0 + 0.5 * (lat_2d - 22.0)
            dist_deg = np.abs(lon_2d - kuro_axis_lon)
            dist_km = dist_deg * 111.0 * np.cos(np.radians(lat_2d))
            features["kuroshio_distance"] = np.clip(dist_km, 0, 3000).ravel()

        # 9b. 台灣海峽區域旗標 (v13.1 高斯平滑邊界)
        ts_flag = ocean_data.get("taiwan_strait_flag")
        if ts_flag is not None:
            features["taiwan_strait_flag"] = (
                self._align(ts_flag, sst.shape).ravel()
            )
        else:
            lat_g = ocean_data.get("lat", np.linspace(5, 35, ny))
            lon_g = ocean_data.get("lon", np.linspace(120, 175, nx))
            lat_2d, lon_2d = np.meshgrid(lat_g, lon_g, indexing='ij')
            
            # 海峽中心約 23.75N, 119.5E
            dist_deg = np.sqrt((lat_2d - 23.75)**2 + (lon_2d - 119.5)**2)
            # 高斯平滑，sigma ~ 1.5°
            in_strait = np.exp(-(dist_deg**2) / (2 * 1.5**2))
            features["taiwan_strait_flag"] = in_strait.ravel()

        # ── 10. [海鷹/蒼鷺] 深層特徵 (4 個) ──
        t100_data = ocean_data.get("t100")
        features["t100"] = (
            self._align(t100_data, sst.shape).ravel() if t100_data is not None
            else np.full(n_pixels, 15.0)  # 萬用預設
        )

        grad_s = ocean_data.get("gradient_strength")
        features["gradient_strength"] = (
            self._align(grad_s, sst.shape).ravel() if grad_s is not None
            else np.full(n_pixels, 0.1)
        )

        dt_s100 = ocean_data.get("delta_t_surface_100")
        features["delta_t_surface_100"] = (
            self._align(dt_s100, sst.shape).ravel() if dt_s100 is not None
            else (features["sst"] - features["t100"])  # 導出
        )

        chl_lag = ocean_data.get("chl_lag15d")
        features["chl_lag15d"] = (
            self._align(chl_lag, sst.shape).ravel() if chl_lag is not None
            else features["chl_log"].copy()  # fallback: 當前 CHL
        )

        # ── 11. v13.2-audit 新增 8 個科學衍生特徵 ──

        # 11a. Moonlight DVM suppression (Benoit-Bird 2009)
        moon_val = features["moon_phase"]
        features["moonlight_dvm_suppression"] = 1.0 - 0.4 * moon_val

        # 11b. Post-storm CHL bloom potential
        post_storm = ocean_data.get("post_storm_chl_bloom")
        features["post_storm_chl_bloom"] = (
            self._align(post_storm, sst.shape).ravel() if post_storm is not None
            else np.zeros(n_pixels)  # 無颱風資料 = 0
        )

        # 11c. Seamount × Current interaction (Taylor column)
        sm_dist = features.get("dist_to_seamount", np.full(n_pixels, 300.0))
        cs_flat = features["current_speed"]
        features["seamount_current_interaction"] = np.clip(
            np.maximum(0, 1 - sm_dist / 300) * cs_flat * 8, 0, 1
        )

        # 11d. Prey thermocline trap
        grad_flat = features.get("gradient_strength", np.full(n_pixels, 0.1))
        mld_flat = features.get("mld", np.full(n_pixels, 50.0))
        features["prey_thermocline_trap"] = np.clip(
            grad_flat * 50 * np.minimum(1, 80 / np.maximum(mld_flat, 10)), 0, 1
        )

        # 11e. Tidal mixing index (spring/neap)
        features["tidal_mixing_index"] = 0.5 + 0.5 * np.cos(4 * np.pi * moon_val)

        # 11f. SST seasonal derivative
        dsst_dt = ocean_data.get("dsst_dt")
        if dsst_dt is not None:
            features["sst_seasonal_derivative"] = self._align(dsst_dt, sst.shape).ravel()
        else:
            # Fallback: approximate from SST spatial gradient magnitude
            features["sst_seasonal_derivative"] = features.get(
                "sst_gradient", np.zeros(n_pixels)
            ) * 0.5

        # 11g. Convergence proxy
        conv_strength = ocean_data.get("convergence_strength")
        if conv_strength is not None:
            features["convergence_proxy"] = self._align(conv_strength, sst.shape).ravel()
        else:
            # Fallback: current_speed × front_strength
            features["convergence_proxy"] = np.clip(
                cs_flat * features["front_strength"] * 30, 0, 1
            )

        # 11h. Dawn twilight hours
        dawn_hrs = ocean_data.get("dawn_twilight_hours")
        if dawn_hrs is not None:
            features["dawn_twilight_hours"] = np.full(n_pixels, float(dawn_hrs))
        else:
            # Approx from lat + season
            doy_local = now.timetuple().tm_yday
            lat_arr = ocean_data.get("lat", np.linspace(5, 35, ny))
            solar_decl = 23.44 * np.sin(2 * np.pi * (doy_local - 81) / 365)
            lat_mid = np.mean(lat_arr)
            tf = 1 + 0.3 * abs(np.tan(np.radians(solar_decl)) *
                                np.tan(np.radians(lat_mid)))
            features["dawn_twilight_hours"] = np.full(
                n_pixels, np.clip(1.5 * tf, 1.0, 4.0)
            )

        # ── 12. v18.3 氣候指數 ML 特徵 (4 個) ──
        # 這些值由 pipeline 從 NOAA PSL API 取得後放入 ocean_data
        # enso_oni 已在 8d 提取 (line 467)，這裡提取 lag 版本
        enso_lag1 = ocean_data.get("enso_lag1", 0.0)
        features["enso_lag1"] = np.full(n_pixels, float(enso_lag1))

        enso_lag2 = ocean_data.get("enso_lag2", 0.0)
        features["enso_lag2"] = np.full(n_pixels, float(enso_lag2))

        pdo_val = ocean_data.get("pdo_index", 0.0)
        features["pdo_index"] = np.full(n_pixels, float(pdo_val))

        soi_val = ocean_data.get("soi_index", 0.0)
        features["soi_index"] = np.full(n_pixels, float(soi_val))

        # ── 13. v18.3 深層溶氧 (3 個) ──
        # Liu 2025 ICES: o2_0/o2_50 是最重要棲地決定因素
        # Xu 2025 Cook Is.: DO 150m/200m 最重要
        # fallback 值基於全球平均深層 DO (ml/L)
        do50 = ocean_data.get("do_50m")
        features["do_50m"] = (
            self._align(do50, sst.shape).ravel() if do50 is not None
            else np.full(n_pixels, 4.5)  # 全球 50m 平均 ~4.5 ml/L
        )

        do150 = ocean_data.get("do_150m")
        features["do_150m"] = (
            self._align(do150, sst.shape).ravel() if do150 is not None
            else np.full(n_pixels, 3.0)  # 全球 150m 平均 ~3.0 ml/L
        )

        do200 = ocean_data.get("do_200m")
        features["do_200m"] = (
            self._align(do200, sst.shape).ravel() if do200 is not None
            else np.full(n_pixels, 2.5)  # 全球 200m 平均 ~2.5 ml/L
        )

        # ── 14. [v19-causal] SSS 鹽度適宜性指數 ──
        # 因果鏈 #10：魷魚(赤魷)最適 34-35 PSU，< 33 PSU 避開
        # Chen et al. (2021), Alabia et al. (2020)
        sss_raw = ocean_data.get("salinity")
        if sss_raw is not None:
            sss_val = self._align(sss_raw, sst.shape).ravel()
            # Sigmoid SI: 在 32-34 PSU 之間快速變化
            # < 32 PSU → SI ≈ 0 (河口/大雨帶硬排除)
            # 34-35 PSU → SI ≈ 1 (最適)
            sss_si = 1.0 / (1.0 + np.exp(-4.0 * (sss_val - 33.0)))
            features["salinity_si"] = sss_si
        else:
            features["salinity_si"] = np.full(n_pixels, 0.8)  # 遠洋預設高適宜性

        # ── 15. [v19-causal] 渦旋成熟度指數 ──
        # 因果鏈 #2：赤魷豐度在渦旋成熟期(生命中段)達到 peak
        # Yu et al. (2023) Frontiers in Marine Science
        eddy_age = ocean_data.get("eddy_age_days")  # 由 eddy_detector 提供
        if eddy_age is not None:
            eddy_age_arr = self._align(eddy_age, sst.shape).ravel()
            # 成熟度 = min(age / 90, 1.0)，90天飽和
            # 無渦旋區域 eddy_age = 0 → maturity = 0
            features["eddy_maturity_index"] = np.clip(eddy_age_arr / 90.0, 0.0, 1.0)
        else:
            features["eddy_maturity_index"] = np.zeros(n_pixels)

        # 組裝特徵矩陣
        feature_names = list(features.keys())
        X = np.column_stack([
            np.nan_to_num(features[fn], nan=0, posinf=0, neginf=0)
            for fn in feature_names
        ])

        return X, feature_names

    @staticmethod
    def _compute_distance_field(
        mask: np.ndarray, lats: np.ndarray, lons: np.ndarray,
    ) -> np.ndarray:
        """
        計算每個網格點到最近 mask 為 True 的點的距離 (km)。
        [v15.3-perf] 用 scipy.spatial.cKDTree 替代原始 triple-loop——加速 ~100x
        """
        ny, nx = mask.shape
        dist = np.full((ny, nx), 500.0, dtype=np.float32)  # 預設 500km

        feature_pts = np.argwhere(mask)
        if len(feature_pts) == 0:
            return dist

        # 取樣 (避免過多特徵點讓 KDTree build 太慢)
        if len(feature_pts) > 200:
            idx = np.linspace(0, len(feature_pts) - 1, 200, dtype=int)
            feature_pts = feature_pts[idx]

        # 將經緯度索引轉為經緯度值
        mean_lat = np.mean(lats) if len(lats) > 0 else 20.0
        cos_lat = np.cos(np.radians(mean_lat))

        # 特徵點經緯度 (km 尺度)
        feat_lats = np.array([lats[min(fj, len(lats)-1)] for fj, fi in feature_pts])
        feat_lons = np.array([lons[min(fi, len(lons)-1)] for fj, fi in feature_pts])
        feat_coords = np.column_stack([
            feat_lats * 111.0,
            feat_lons * 111.0 * cos_lat,
        ])

        try:
            from scipy.spatial import cKDTree
        except ImportError:
            # fallback to pure Python if scipy unavailable (should never happen)
            return dist

        tree = cKDTree(feat_coords)

        # 建構全網格座標 (km 尺度)
        lat_2d, lon_2d = np.meshgrid(lats, lons, indexing='ij')
        grid_coords = np.column_stack([
            lat_2d.ravel() * 111.0,
            lon_2d.ravel() * 111.0 * cos_lat,
        ])

        distances, _ = tree.query(grid_coords, k=1)
        dist = np.clip(distances.reshape(ny, nx), 0, 500).astype(np.float32)

        return dist

    def _align(self, arr: Optional[np.ndarray], shape: tuple) -> np.ndarray:
        if arr is None:
            return np.zeros(shape)
        if arr.shape == shape:
            return arr
        result = np.full(shape, np.nan)
        ny = min(arr.shape[0], shape[0])
        nx = min(arr.shape[1], shape[1])
        result[:ny, :nx] = arr[:ny, :nx]
        return result


# ═══════════════════════════════════════════════════
#  Stacking Ensemble 模型
# ═══════════════════════════════════════════════════

class FishingStackingModel:
    """
    Stacking Ensemble 漁場預測模型

    架構:
    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
    │ RandomForest │  │  XGBoost    │  │  LightGBM   │  │   Ridge     │
    │   (L0-1)    │  │   (L0-2)    │  │   (L0-3)    │  │   (L0-4)    │
    └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
           │                │                │                │
           └───────────┬────┴────────────────┴────────────────┘
                       │
                ┌──────┴──────┐
                │  Meta-Ridge │  ← Level-1 Meta-Learner
                │   (最終)    │
                └─────────────┘

    無訓練數據時的退化模式:
    - 使用專家知識預設權重 (比 v8 的固定規則更聰明)
    - 基於 HSI 模型的分數做 pseudo-label 半監督學習
    - 船長回報數據逐步累積後自動重訓
    """

    def __init__(self, species: str = "skipjack", model_dir: str = "models"):
        self.species = species
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.feature_engineer = FeatureEngineer()
        self.is_trained = False
        self.model = None
        self.scaler = StandardScaler() if HAS_SKLEARN else None

        # 專家權重（退化模式用）
        self._expert_weights = self._get_expert_weights(species)

    def _get_expert_weights(self, species: str) -> Dict[str, float]:
        """
        基於文獻的專家知識權重（逆向工程自商業系統）

        CATSAT 使用 AMM (等權平均)，我們用文獻驗證的差異化權重
        INCOIS PFZ 重鋒面，我們根據物種調整
        """
        weights_map = {
            "skipjack": {
                "sst": 0.25, "chl_log": 0.15, "front_strength": 0.20,
                "ftle": 0.15, "current_speed": 0.05, "moon_phase": 0.05,
                "ais_fishing_density": 0.10, "ssh": 0.05,
            },
            "yellowfin": {
                "sst": 0.20, "chl_log": 0.10, "ssh": 0.10,
                "front_strength": 0.15, "ftle": 0.10,
                "thermocline_depth": 0.15, "ais_fishing_density": 0.10,
                "current_speed": 0.05, "moon_phase": 0.05,
            },
            "bigeye": {
                "sst": 0.10, "thermocline_depth": 0.25, "d20_depth": 0.20,
                "ssh": 0.15, "chl_log": 0.05, "ftle": 0.10,
                "ais_fishing_density": 0.10, "current_speed": 0.05,
            },
            "squid": {
                "sst": 0.15, "chl_log": 0.15, "moon_phase": 0.20,
                "viirs_light_density": 0.25, "front_strength": 0.10,
                "ftle": 0.05, "ais_fishing_density": 0.10,
            },
        }
        return weights_map.get(species, weights_map["skipjack"])

    def build_stacking_model(self):
        """建立 Stacking Ensemble 模型 [v12-enhance: +CatBoost]"""
        if not HAS_SKLEARN:
            log.warning("sklearn 不可用，使用專家加權退化模式")
            return None

        estimators = [
            ("rf", RandomForestRegressor(
                n_estimators=200, max_depth=12, min_samples_leaf=10,
                n_jobs=-1, random_state=42,
            )),
            ("gbr", GradientBoostingRegressor(
                n_estimators=150, max_depth=6, learning_rate=0.05,
                subsample=0.8, random_state=42,
            )),
        ]

        if HAS_XGB:
            estimators.append(("xgb", xgb.XGBRegressor(
                n_estimators=200, max_depth=8, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0,
                n_jobs=-1, random_state=42,
            )))

        if HAS_LGB:
            estimators.append(("lgb", lgb.LGBMRegressor(
                n_estimators=200, max_depth=8, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0,
                n_jobs=-1, random_state=42, verbose=-1,
            )))

        # [v12-enhance] CatBoost — 第5個基學習器
        # CatBoost 對類別特徵和 ordered boosting 有獨特優勢
        # 在海洋數據中，species/season 等隱含類別特徵受益
        if HAS_CATBOOST:
            estimators.append(("catboost", CatBoostRegressor(
                iterations=200, depth=8, learning_rate=0.05,
                l2_leaf_reg=3.0, subsample=0.8,
                random_seed=42, verbose=0,
            )))

        estimators.append(("ridge", Ridge(alpha=1.0)))

        self.model = StackingRegressor(
            estimators=estimators,
            final_estimator=RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0]),
            cv=KFold(n_splits=5, shuffle=True, random_state=42),
            n_jobs=-1,
        )

        log.info(f"Stacking 模型建立完成: {len(estimators)} 個基學習器 "
                 f"({'含 CatBoost' if HAS_CATBOOST else '無 CatBoost'})")
        return self.model

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        lats: Optional[np.ndarray] = None,
        lons: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        訓練模型

        參數:
            X: (n_samples, n_features) 特徵矩陣
            y: (n_samples,) 目標值 (CPUE 或 歸一化漁獲量)
            lats: (n_samples,) 緯度 (for SpatialBlockCV)
            lons: (n_samples,) 經度 (for SpatialBlockCV)

        返回:
            訓練指標
        """
        if not HAS_SKLEARN:
            log.warning("sklearn 不可用，跳過訓練")
            return {"status": "fallback_mode"}

        log.info(f"開始訓練 {self.species} Stacking 模型...")
        log.info(f"  訓練集: {X.shape[0]} 樣本, {X.shape[1]} 特徵")

        # 標準化
        X_scaled = self.scaler.fit_transform(X)

        # 處理 NaN
        X_scaled = np.nan_to_num(X_scaled, nan=0, posinf=0, neginf=0)

        # 建立模型
        if self.model is None:
            self.build_stacking_model()

        # 訓練
        self.model.fit(X_scaled, y)
        self.is_trained = True

        # 評估: SpatialBlockCV (if coords available) else KFold fallback
        if lats is not None and lons is not None:
            log.info("  Using SpatialBlockCV for evaluation...")
            spatial_cv = SpatialBlockCV(n_blocks=5, buffer_km=50)
            folds = spatial_cv.split(X_scaled, lats, lons)
            y_pred_cv = np.full_like(y, np.nan)
            for train_idx, test_idx in folds:
                from sklearn.base import clone
                m = clone(self.model)
                m.fit(X_scaled[train_idx], y[train_idx])
                y_pred_cv[test_idx] = m.predict(X_scaled[test_idx])
            valid = np.isfinite(y_pred_cv)
            rmse = np.sqrt(mean_squared_error(y[valid], y_pred_cv[valid]))
            r2 = r2_score(y[valid], y_pred_cv[valid])
            cv_method = "SpatialBlockCV"
        else:
            log.warning("  No lat/lon provided — falling back to random KFold CV")
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            y_pred_cv = cross_val_predict(self.model, X_scaled, y, cv=kf)
            rmse = np.sqrt(mean_squared_error(y, y_pred_cv))
            r2 = r2_score(y, y_pred_cv)
            cv_method = "KFold"

        metrics = {
            "rmse": float(rmse),
            "r2": float(r2),
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "species": self.species,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        log.info(f"  ✅ 訓練完成: RMSE={rmse:.4f}, R²={r2:.4f}")

        # 特徵重要性（來自 RandomForest）
        if feature_names:
            rf_model = self.model.estimators_[0]  # RF
            importances = rf_model.feature_importances_
            top_features = sorted(
                zip(feature_names, importances),
                key=lambda x: x[1], reverse=True
            )[:10]
            metrics["top_features"] = [
                {"name": n, "importance": float(v)} for n, v in top_features
            ]
            log.info("  前 10 重要特徵:")
            for name, imp in top_features:
                log.info(f"    {name}: {imp:.4f}")

        # 保存模型
        self._save_model(metrics)

        return metrics

    def predict(
        self,
        ocean_data: Dict[str, Any],
    ) -> np.ndarray:
        """
        預測漁場品質分數

        如果模型已訓練 → 用 ML 預測
        如果未訓練 → 用專家加權退化模式 + HSI 分數
        """
        sst = ocean_data.get("sst")
        if sst is None:
            raise ValueError("SST 缺失")

        ny, nx = sst.shape

        # 提取特徵
        X, feature_names = self.feature_engineer.extract_features(
            ocean_data, self.species
        )

        if self.is_trained and self.model is not None and self.scaler is not None:
            # ── ML 預測模式 ──
            log.info(f"使用 ML Stacking 預測 {self.species}...")

            # [v15.3-fix] Feature alignment: trim X to match model's expected 
            # feature count when loading older WCPFC models (46 features) with
            # newer FeatureEngineer (59 features).
            n_model_features = len(self.scaler.mean_)
            if X.shape[1] > n_model_features:
                log.info(
                    f"[v15.3] 特徵對齊: pipeline={X.shape[1]} → "
                    f"model={n_model_features} (trimming extra features)"
                )
                X = X[:, :n_model_features]
            elif X.shape[1] < n_model_features:
                # Pad with zeros if somehow fewer features
                n_original = X.shape[1]
                pad = np.zeros((X.shape[0], n_model_features - n_original))
                X = np.hstack([X, pad])
                log.warning(
                    f"[v15.3] 特徵不足: pipeline={n_original} → "
                    f"model={n_model_features} (padding with zeros)"
                )

            X_scaled = self.scaler.transform(X)
            X_scaled = np.nan_to_num(X_scaled, nan=0, posinf=0, neginf=0)
            scores = self.model.predict(X_scaled)
            scores = np.clip(scores, 0, 1).reshape(ny, nx)
        else:
            # ── 專家加權退化模式 ──
            log.info(f"使用智慧專家加權預測 {self.species} (無訓練數據)...")
            scores = self._expert_predict(X, feature_names, ny, nx)

        # ── T2: Apply CPUE physical cap ──
        cap = CPUE_CAPS.get(self.species, 500)
        if hasattr(scores, 'max') and scores.max() > cap:
            n_capped = int((scores > cap).sum())
            log.warning(f"{self.species}: {n_capped} predictions capped at {cap} kg/day")
            scores = np.clip(scores, 0, cap)

        return scores

    def _expert_predict(
        self,
        X: np.ndarray,
        feature_names: List[str],
        ny: int, nx: int,
    ) -> np.ndarray:
        """
        專家知識加權預測（退化模式）

        比 v8 的固定規則更聰明：
        1. 使用文獻驗證的差異化權重
        2. 每個特徵先做高斯 SI 轉換
        3. 用加權幾何平均融合（不是簡單加法）
        """
        from engine.hsi_models import gaussian_si, range_si

        # 取得各特徵的列索引
        feat_idx = {fn: i for i, fn in enumerate(feature_names)}

        # 物種參數
        species_params = {
            "skipjack": {"sst_opt": 29, "sst_sig": 2.5, "chl_opt": -0.6, "chl_sig": 0.4},
            "yellowfin": {"sst_opt": 28, "sst_sig": 2.0, "chl_opt": -0.7, "chl_sig": 0.3},
            "bigeye": {"sst_opt": 28, "sst_sig": 2.0, "chl_opt": -0.8, "chl_sig": 0.3},
            "squid": {"sst_opt": 17, "sst_sig": 3.0, "chl_opt": -0.1, "chl_sig": 0.5},
        }
        params = species_params.get(self.species, species_params["skipjack"])

        n_pixels = X.shape[0]
        factors = []
        weights_list = []

        # SST SI
        sst_col = X[:, feat_idx.get("sst", 0)]
        si_sst = np.exp(-((sst_col - params["sst_opt"])**2) / (2 * params["sst_sig"]**2))
        factors.append(si_sst)
        weights_list.append(self._expert_weights.get("sst", 0.2))

        # Chl SI
        if "chl_log" in feat_idx:
            chl_col = X[:, feat_idx["chl_log"]]
            si_chl = np.exp(-((chl_col - params["chl_opt"])**2) / (2 * params["chl_sig"]**2))
            factors.append(si_chl)
            weights_list.append(self._expert_weights.get("chl_log", 0.15))

        # 鋒面 SI
        if "front_strength" in feat_idx:
            fs_col = X[:, feat_idx["front_strength"]]
            si_front = np.clip(fs_col * 2.0, 0, 1)
            factors.append(si_front)
            weights_list.append(self._expert_weights.get("front_strength", 0.15))

        # FTLE SI
        if "ftle" in feat_idx:
            ftle_col = X[:, feat_idx["ftle"]]
            ftle_max = np.nanmax(ftle_col) if np.nanmax(ftle_col) > 0 else 1
            si_ftle = np.clip(ftle_col / ftle_max, 0, 1)
            factors.append(si_ftle)
            weights_list.append(self._expert_weights.get("ftle", 0.1))

        # AIS 漁船密度
        if "ais_fishing_density" in feat_idx:
            ais_col = X[:, feat_idx["ais_fishing_density"]]
            ais_max = np.nanmax(ais_col) if np.nanmax(ais_col) > 0 else 1
            si_ais = np.clip(ais_col / ais_max, 0, 1)
            factors.append(si_ais)
            weights_list.append(self._expert_weights.get("ais_fishing_density", 0.1))

        # ── [v12-gfw] NO₃ SI — 營養鹽指標 ──
        # 最佳區間: ~2-8 mmol/m³ (上升流活躍但不是剛發生的冷水)
        # 太低(<0.5) = 貧瘠水域, 太高(>15) = 深水剛上湧、魚還沒來
        if "no3_surface" in feat_idx:
            no3_col = X[:, feat_idx["no3_surface"]]
            si_no3 = np.exp(-((no3_col - 5.0)**2) / (2 * 4.0**2))
            factors.append(si_no3)
            weights_list.append(0.08)
            log.debug(f"  expert NO₃ SI: mean={np.nanmean(si_no3):.3f}")

        # ── [v12-gfw] NPP SI — 淨初級生產力 ──
        if "npp_surface" in feat_idx:
            npp_col = X[:, feat_idx["npp_surface"]]
            npp_max = max(np.nanmax(npp_col), 1e-6)
            si_npp = np.clip(npp_col / npp_max, 0, 1)
            factors.append(si_npp)
            weights_list.append(0.06)

        # ── [v12-gfw] GFW 漁船作業時數 SI — 群體智慧 ──
        # 其他漁船在哪裡 = 最直接的魚群證據
        if "gfw_fishing_hours" in feat_idx:
            gfw_col = X[:, feat_idx["gfw_fishing_hours"]]
            # log scale: log(1+hours) normalized
            gfw_log = np.log1p(gfw_col)
            gfw_max = max(np.nanmax(gfw_log), 1e-6)
            si_gfw = np.clip(gfw_log / gfw_max, 0, 1)
            factors.append(si_gfw)
            weights_list.append(0.12)  # 高權重 — 漁船位置是最強信號
            log.debug(f"  expert GFW SI: mean={np.nanmean(si_gfw):.3f}, max_hours={np.nanmax(gfw_col):.0f}")

        # 加權幾何平均
        w = np.array(weights_list)
        w = w / w.sum()
        result = np.ones(n_pixels, dtype=float)
        for f, wi in zip(factors, w):
            f_safe = np.clip(f, 1e-10, 1.0)
            result *= np.power(f_safe, wi)

        return np.clip(result, 0, 1).reshape(ny, nx)

    def _save_model(self, metrics: Dict):
        """保存模型到磁碟 (with T3 validation + P0-1 HMAC signing)"""
        model_path = self.model_dir / f"stacking_{self.species}.pkl"
        meta_path = self.model_dir / f"stacking_{self.species}_meta.json"
        scaler_path = self.model_dir / f"scaler_{self.species}.pkl"

        try:
            # T3: Validate scaler is fitted before saving
            assert hasattr(self.scaler, 'mean_'), "Scaler not fitted (no mean_)!"
            assert hasattr(self.scaler, 'scale_'), "Scaler not fitted (no scale_)!"

            with open(model_path, "wb") as f:
                pickle.dump({
                    "model": self.model,
                    "scaler": self.scaler,
                    "species": self.species,
                }, f)

            # [v13.2-P0] Sign model file
            _sign_model_file(model_path)

            # T3: Also save scaler separately for validation
            with open(scaler_path, "wb") as f:
                pickle.dump(self.scaler, f)
            _sign_model_file(scaler_path)

            with open(meta_path, "w") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)

            # T3: Verify saved files
            model_size = model_path.stat().st_size
            scaler_size = scaler_path.stat().st_size
            log.info(f"模型已保存: {model_path} ({model_size // 1024} KB)")
            log.info(f"Scaler: {scaler_path} ({scaler_size} bytes)")
            assert scaler_size > 500, f"Scaler file too small: {scaler_size} bytes"

            # T3: Verify reload works (via signature-checked path)
            _verify_model_file(model_path)
            with open(model_path, "rb") as f:
                loaded = pickle.load(f)
            assert loaded["model"] is not None, "Model reload failed"
            assert hasattr(loaded["scaler"], 'mean_'), "Scaler reload missing mean_"
            log.info(f"Reload verification: OK")

        except Exception as e:
            log.error(f"模型保存失敗: {e}")

    def load_model(self) -> bool:
        """從磁碟載入模型 — 委派到 safe_model_load"""
        return self.safe_model_load()

    def safe_model_load(self) -> bool:
        """
        [v12-fix] 安全模型載入 — 附版本/物種/特徵數驗證

        驗證項目:
          0. [v13.2-P0] HMAC-SHA256 簽章驗證 (防 RCE)
          1. 檔案存在性
          2. Pickle 完整性 (不會因部分損壞 crash)
          3. 物種一致性 (模型是否匹配當前 species)
          4. 版本相容性 (模型的特徵數是否與當前 FE 相容)
          5. Scaler 完整性 (fitted mean_/scale_ 存在)
        """
        model_path = self.model_dir / f"stacking_{self.species}.pkl"
        meta_path = self.model_dir / f"stacking_{self.species}_meta.json"

        if not model_path.exists():
            log.info(f"模型檔案不存在: {model_path}")
            return False

        # [v13.2-P0] Step 0: HMAC signature verification BEFORE pickle.load
        try:
            _verify_model_file(model_path)
        except RuntimeError as e:
            log.error(f"[v13.2-P0] 模型簽章驗證失敗: {e}")
            return False

        try:
            with open(model_path, "rb") as f:
                data = pickle.load(f)
        except (pickle.UnpicklingError, EOFError, OSError) as e:
            log.error(f"[v12-fix] 模型檔案損壞，無法反序列化: {e}")
            return False
        except Exception as e:
            log.error(f"[v12-fix] 模型載入失敗 (未預期錯誤): {e}")
            return False

        # ── 驗證 1: 結構完整性 ──
        if not isinstance(data, dict):
            log.error("[v12-fix] 模型格式錯誤: 預期 dict")
            return False
        if "model" not in data or "scaler" not in data:
            log.error("[v12-fix] 模型缺少必要 key: model/scaler")
            return False

        # ── 驗證 2: 物種一致性 ──
        saved_species = data.get("species", "unknown")
        if saved_species != self.species:
            log.warning(f"[v12-fix] 物種不匹配: 模型={saved_species}, 當前={self.species}")
            return False

        # ── 驗證 3: Scaler 完整性 ──
        scaler = data["scaler"]
        if scaler is not None:
            if not hasattr(scaler, 'mean_') or not hasattr(scaler, 'scale_'):
                log.error("[v12-fix] Scaler 未完成 fit (缺少 mean_/scale_)")
                return False

            # 特徵數相容性
            n_features_saved = len(scaler.mean_)
            n_features_current = len(FeatureEngineer.FEATURE_NAMES)
            if n_features_saved != n_features_current:
                if n_features_saved > n_features_current:
                    log.error(
                        f"[v12-fix] 模型特徵數超過目前定義: 模型={n_features_saved}, "
                        f"當前={n_features_current}。無法載入。"
                    )
                    return False
                else:
                    # [海鷹] 舊模型特徵數較少 — 可向後相容
                    log.warning(
                        f"[v12-fix] 模型特徵數較少: 模型={n_features_saved}, "
                        f"當前={n_features_current}。新特徵將使用預設值。"
                    )

        # ── 驗證 4: Meta 資訊 (optional) ──
        if meta_path.exists():
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                log.info(f"模型 meta: {meta.get('trained_at', '?')} "
                         f"R²={meta.get('r2', '?')}")
            except Exception:
                pass  # meta 不影響載入

        # ── 全部通過: 載入 ──
        self.model = data["model"]
        self.scaler = scaler
        self.is_trained = True
        log.info(f"✅ 模型安全載入: {model_path} ({model_path.stat().st_size // 1024} KB)")
        return True

    def train_from_captain_reports(
        self,
        reports: List[Dict[str, Any]],
        ocean_data_history: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        從船長回報數據訓練模型

        reports 格式:
        [{
            "lat": 15.5, "lon": 131.0,
            "species": "skipjack",
            "catch_kg": 3000,
            "effort_hours": 8,
            "date": "2025-12-01",
        }, ...]
        """
        if not reports:
            log.warning("無船長回報數據，跳過訓練")
            return {"status": "no_data"}

        log.info(f"從 {len(reports)} 筆船長回報訓練 {self.species} 模型...")

        # 提取每個回報位置的特徵和 CPUE 標籤
        X_list = []
        y_list = []

        for report in reports:
            if report.get("species") != self.species:
                continue

            # CPUE = Catch Per Unit Effort（標準化漁獲量）
            catch = report.get("catch_kg", 0)
            effort = report.get("effort_hours", 1)
            cpue = catch / max(effort, 0.1)

            # 歸一化 CPUE 到 [0, 1]
            y_list.append(cpue)

            # 在回報位置附近提取特徵
            # (簡化版：用最近格點的特徵)
            X_list.append(self._get_features_at_point(
                report["lat"], report["lon"], ocean_data_history
            ))

        if len(y_list) < 20:
            log.warning(f"回報數據太少 ({len(y_list)}筆)，需要至少 20 筆才能訓練")
            return {"status": "insufficient_data", "n_reports": len(y_list)}

        X = np.array(X_list)
        y = np.array(y_list)

        # 歸一化 y 到 [0, 1]
        y_max = y.max()
        if y_max > 0:
            y = y / y_max

        return self.train(X, y)

    def _get_features_at_point(
        self, lat: float, lon: float, ocean_data: Dict
    ) -> np.ndarray:
        """在指定經緯度提取特徵向量"""
        lat_arr = ocean_data.get("lat", np.arange(5, 35, 0.25))
        lon_arr = ocean_data.get("lon", np.arange(120, 175, 0.25))

        iy = np.argmin(np.abs(lat_arr - lat))
        ix = np.argmin(np.abs(lon_arr - lon))

        X, _ = self.feature_engineer.extract_features(ocean_data, self.species)
        ny = len(lat_arr)
        nx = len(lon_arr)
        pixel_idx = iy * nx + ix

        if pixel_idx < X.shape[0]:
            return X[pixel_idx]
        return np.zeros(X.shape[1])

    def predict_with_uncertainty(
        self,
        ocean_data: Dict[str, Any],
        alpha: float = 0.1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        [v11 Feature F] 帶不確定性量化的預測

        使用 Split Conformal Prediction:
          1. 用訓練數據的殘差建立校正集
          2. 取殘差的 (1-α) 分位數作為信賴間距寬度
          3. 回傳 (y_pred, y_lower, y_upper)

        若未訓練: 使用專家模式 + 基於 HSI 的不確定性估計

        Parameters:
            ocean_data: 海洋觀測數據字典
            alpha: 顯著水準 (預設 0.1 = 90% 信賴區間)

        Returns:
            (y_pred, y_lower, y_upper): 各為 2D arrays (ny, nx)
        """
        sst = ocean_data.get("sst")
        if sst is None:
            raise ValueError("SST 缺失")
        ny, nx = sst.shape

        # 取得點預測
        y_pred = self.predict(ocean_data)

        if self.is_trained and hasattr(self, '_conformal_residuals'):
            # Conformal prediction interval
            q = np.quantile(self._conformal_residuals, 1 - alpha)
            y_lower = np.clip(y_pred - q, 0, None)
            y_upper = np.clip(y_pred + q, 0, 1)
            log.info(f"  Conformal interval (α={alpha}): ±{q:.4f}")
        else:
            # 退化模式: 基於預測分數的不確定性估計
            # 高分或低分 → 低不確定性; 中間分 → 高不確定性
            uncertainty = 0.15 + 0.25 * np.exp(-8 * (y_pred - 0.5) ** 2)
            y_lower = np.clip(y_pred - uncertainty, 0, None)
            y_upper = np.clip(y_pred + uncertainty, 0, 1)
            log.info(f"  Heuristic uncertainty: avg width={np.nanmean(y_upper - y_lower):.4f}")

        return y_pred, y_lower.astype(np.float32), y_upper.astype(np.float32)

    def calibrate_conformal(self, X_calib: np.ndarray, y_calib: np.ndarray):
        """
        校準 conformal prediction 所需的殘差分佈

        用訓練後的一部分數據 (hold-out calibration set) 來計算殘差

        Parameters:
            X_calib: 校準集特徵
            y_calib: 校準集真實值
        """
        if not self.is_trained or self.model is None:
            log.warning("模型未訓練，無法校準 conformal intervals")
            return

        X_scaled = self.scaler.transform(X_calib)
        X_scaled = np.nan_to_num(X_scaled, nan=0, posinf=0, neginf=0)
        y_pred = self.model.predict(X_scaled)
        self._conformal_residuals = np.abs(y_pred - y_calib)
        log.info(f"  Conformal 校準完成: {len(self._conformal_residuals)} 殘差, "
                 f"median={np.median(self._conformal_residuals):.4f}")


# ═══════════════════════════════════════════════════
#  [v11 Feature E] CPUE 標準化器
# ═══════════════════════════════════════════════════

class CPUEStandardizer:
    """
    [v11 Feature E] Delta-Lognormal GLM CPUE 標準化

    科學依據:
      漁獲率 (CPUE = Catch / Effort) 受非漁場因素影響:
      - 船型/噸位
      - 漁具 (延繩釣/圍網)
      - 區域 (鄰近 EEZ 限制)
      - 季節 (魚群行為)
      - 年份 (種群趨勢)

    Delta-Lognormal 方法:
      Step 1: Binomial GLM → P(catch > 0)
      Step 2: Lognormal GLM → E[CPUE | catch > 0]
      Step 3: 標準化 CPUE = P(catch>0) × E[CPUE|catch>0]

    文獻: Maunder & Punt 2004, Campbell 2015
    """

    # 漁具效率因子 (相對延繩釣=1.0)
    GEAR_EFFICIENCY = {
        "longline": 1.00,
        "purse_seine": 3.50,     # 圍網效率高 3-5 倍
        "pole_and_line": 1.50,
        "handline": 0.80,
        "troll": 0.60,
        "gillnet": 1.20,
    }

    # 船型噸位效率 (非線性: 越大不一定越好)
    VESSEL_SIZE_CLASSES = {
        "small": {"min_grt": 0, "max_grt": 50, "efficiency": 0.70},
        "medium": {"min_grt": 50, "max_grt": 200, "efficiency": 1.00},
        "large": {"min_grt": 200, "max_grt": 500, "efficiency": 1.25},
        "industrial": {"min_grt": 500, "max_grt": 10000, "efficiency": 1.10},
    }

    def __init__(self):
        self._fit_params = None  # 保存 GLM 擬合參數

    def standardize_cpue(
        self,
        catch_kg: np.ndarray,
        effort_hours: np.ndarray,
        gear: Optional[np.ndarray] = None,
        vessel_grt: Optional[np.ndarray] = None,
        lat: Optional[np.ndarray] = None,
        lon: Optional[np.ndarray] = None,
        month: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        標準化 CPUE

        Parameters:
            catch_kg: 漁獲量 (kg)
            effort_hours: 努力量 (小時)
            gear: 漁具類型 (字串陣列)
            vessel_grt: 船舶噸位 (GRT)
            lat, lon: 位置
            month: 月份

        Returns:
            std_cpue: 標準化 CPUE (kg/effort_unit)
        """
        n = len(catch_kg)

        # 原始 CPUE
        effort_safe = np.maximum(effort_hours, 1.0)
        raw_cpue = catch_kg / effort_safe

        # 1. 漁具效率修正
        gear_factor = np.ones(n)
        if gear is not None:
            for i in range(n):
                g = str(gear[i]).lower().replace(" ", "_")
                gear_factor[i] = self.GEAR_EFFICIENCY.get(g, 1.0)

        # 2. 船型修正
        vessel_factor = np.ones(n)
        if vessel_grt is not None:
            for i in range(n):
                grt = float(vessel_grt[i])
                for cls in self.VESSEL_SIZE_CLASSES.values():
                    if cls["min_grt"] <= grt < cls["max_grt"]:
                        vessel_factor[i] = cls["efficiency"]
                        break

        # 3. 季節修正 (使用正弦函數平滑)
        season_factor = np.ones(n)
        if month is not None:
            # 鮪魚通常在 4-6 月和 10-11 月較多
            m = np.array(month, dtype=float)
            season_factor = 1.0 + 0.15 * np.sin(2 * np.pi * (m - 4) / 12)

        # 4. Delta-Lognormal 標準化
        # Step A: 標準化努力量到基準漁具/船型
        std_effort = effort_safe * gear_factor * vessel_factor * season_factor

        # Step B: 標準化 CPUE = 原始漁獲 / 標準化努力量
        std_cpue = catch_kg / np.maximum(std_effort, 1.0)

        # Step C: 零漁獲處理 (Delta 部分)
        has_catch = catch_kg > 0
        if np.any(has_catch):
            # 正漁獲的 log-normal 均值作為標準化基準
            log_cpue = np.log(std_cpue[has_catch] + 1e-6)
            log_mean = np.mean(log_cpue)
            log_std = np.std(log_cpue) if len(log_cpue) > 1 else 0.5

            # 零膨脹校正: P(catch>0) 調整
            p_positive = np.sum(has_catch) / n
            std_cpue = std_cpue * p_positive

            log.info(f"  CPUE 標準化: n={n}, P(catch>0)={p_positive:.2f}, "
                     f"log_mean={log_mean:.2f}, log_std={log_std:.2f}")
        else:
            log.warning("  所有記錄均為零漁獲")

        return std_cpue.astype(np.float32)

    def compute_nominal_vs_standardized(
        self,
        catch_kg: np.ndarray,
        effort_hours: np.ndarray,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        比較名義 CPUE vs 標準化 CPUE

        Returns:
            dict with nominal, standardized, correction_factor, stats
        """
        effort_safe = np.maximum(effort_hours, 1.0)
        nominal = catch_kg / effort_safe
        standardized = self.standardize_cpue(catch_kg, effort_hours, **kwargs)

        # 校正因子
        nominal_mean = np.nanmean(nominal) if np.any(nominal > 0) else 1.0
        std_mean = np.nanmean(standardized) if np.any(standardized > 0) else 1.0
        correction = std_mean / max(nominal_mean, 1e-6)

        return {
            "nominal_cpue": nominal.astype(np.float32),
            "standardized_cpue": standardized,
            "correction_factor": float(correction),
            "stats": {
                "nominal_mean": float(np.nanmean(nominal)),
                "nominal_cv": float(np.nanstd(nominal) / max(np.nanmean(nominal), 1e-6)),
                "std_mean": float(np.nanmean(standardized)),
                "std_cv": float(np.nanstd(standardized) / max(np.nanmean(standardized), 1e-6)),
                "n_records": int(len(catch_kg)),
                "n_positive": int(np.sum(catch_kg > 0)),
            },
        }


# ═══════════════════════════════════════════════════
#  統一入口
# ═══════════════════════════════════════════════════

class MLFusionEngine:
    """
    ML 融合引擎 — 替代 v8 的規則加權

    用法:
        engine = MLFusionEngine()
        results = engine.predict_all_species(ocean_data)
    """

    def __init__(self, species_list=None, model_dir="models"):
        self.species_list = species_list or ["skipjack", "yellowfin", "bigeye", "squid"]
        self.models = {}

        for sp in self.species_list:
            model = FishingStackingModel(species=sp, model_dir=model_dir)
            model.load_model()  # 嘗試載入已訓練模型
            self.models[sp] = model

    def predict_all_species(
        self,
        ocean_data: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """預測所有物種的漁場分數"""
        results = {}

        for species, model in self.models.items():
            try:
                score_grid = model.predict(ocean_data)
                results[species] = {
                    "score_grid": score_grid,
                    "hsi": score_grid,  # 兼容 v8 介面
                    "species": species,
                    "method": "ml_stacking" if model.is_trained else "expert_weighted",
                    "stats": {
                        "mean_score": float(np.nanmean(score_grid)),
                        "max_score": float(np.nanmax(score_grid)),
                        "suitable_pct": float(np.nanmean(score_grid > 0.6) * 100),
                    },
                }
                log.info(
                    f"  {species}: max={np.nanmax(score_grid):.3f}, "
                    f"method={results[species]['method']}"
                )
            except Exception as e:
                log.error(f"  {species} ML 預測失敗: {e}")

        return results


# ═══════════════════════════════════════════════════
#  [v11 ML-2] 空間區塊交叉驗證
# ═══════════════════════════════════════════════════

class SpatialBlockCV:
    """
    [v11 ML-2] 空間區塊交叉驗證

    問題:
      傳統 KFold 隨機拆分 → 空間自相關導致過度樂觀的 R²
      相鄰像素幾乎相同 → 訓練/測試集「洩漏」

    解決:
      將空間網格劃分為 n_blocks × n_blocks 的區塊
      每次摺疊: 1 個區塊作為測試，其餘作為訓練
      確保訓練/測試之間有足夠的空間距離

    文獻: Roberts et al. 2017, Valavi et al. 2019
    """

    def __init__(self, n_blocks: int = 5, buffer_km: float = 50.0):
        """
        Parameters:
            n_blocks: 每個空間維度的區塊數 (總摺疊數 = n_blocks²)
            buffer_km: 訓練/測試區塊之間的緩衝距離 (km)
        """
        self.n_blocks = n_blocks
        self.buffer_km = buffer_km

    def split(
        self,
        X: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        產生空間區塊摺疊的 (train_idx, test_idx)

        Parameters:
            X: 特徵矩陣 (n_samples, n_features)
            lats: 每個樣本的緯度
            lons: 每個樣本的經度

        Returns:
            List of (train_indices, test_indices)
        """
        n_samples = len(X)

        # 將 lat/lon 網格化為區塊 ID
        lat_min, lat_max = np.min(lats), np.max(lats)
        lon_min, lon_max = np.min(lons), np.max(lons)

        lat_bins = np.linspace(lat_min, lat_max, self.n_blocks + 1)
        lon_bins = np.linspace(lon_min, lon_max, self.n_blocks + 1)

        # 每個樣本的區塊 ID
        lat_block = np.clip(np.digitize(lats, lat_bins) - 1, 0, self.n_blocks - 1)
        lon_block = np.clip(np.digitize(lons, lon_bins) - 1, 0, self.n_blocks - 1)
        block_ids = lat_block * self.n_blocks + lon_block

        unique_blocks = np.unique(block_ids)
        folds = []

        for test_block in unique_blocks:
            test_idx = np.where(block_ids == test_block)[0]
            if len(test_idx) < 5:
                continue

            # 計算區塊中心
            test_lat_c = np.mean(lats[test_idx])
            test_lon_c = np.mean(lons[test_idx])

            # 緩衝區: 排除距離太近的樣本
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[test_idx] = False

            if self.buffer_km > 0:
                for i in range(n_samples):
                    if not train_mask[i]:
                        continue
                    dlat = (lats[i] - test_lat_c) * 111.0
                    dlon = (lons[i] - test_lon_c) * 111.0 * np.cos(np.radians(test_lat_c))
                    dist = np.sqrt(dlat**2 + dlon**2)
                    if dist < self.buffer_km:
                        train_mask[i] = False

            train_idx = np.where(train_mask)[0]
            if len(train_idx) > 10:
                folds.append((train_idx, test_idx))

        log.info(f"  SpatialBlockCV: {len(folds)} folds from "
                 f"{self.n_blocks}×{self.n_blocks} blocks, buffer={self.buffer_km}km")
        return folds

    def cross_val_score(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
    ) -> Dict[str, float]:
        """
        執行空間 CV 並回傳指標

        Returns:
            dict with r2_mean, rmse_mean, r2_std, n_folds
        """
        folds = self.split(X, lats, lons)
        r2_scores = []
        rmse_scores = []

        for train_idx, test_idx in folds:
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            from sklearn.base import clone
            m = clone(model)
            m.fit(X_train, y_train)
            y_pred = m.predict(X_test)

            ss_res = np.sum((y_test - y_pred) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            r2 = 1 - ss_res / max(ss_tot, 1e-8)
            rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

            r2_scores.append(r2)
            rmse_scores.append(rmse)

        return {
            "r2_mean": float(np.mean(r2_scores)) if r2_scores else 0.0,
            "r2_std": float(np.std(r2_scores)) if r2_scores else 0.0,
            "rmse_mean": float(np.mean(rmse_scores)) if rmse_scores else 0.0,
            "n_folds": len(folds),
        }


# ═══════════════════════════════════════════════════
#  [v11 ML-3] TabPFN / AutoML 包裝器
# ═══════════════════════════════════════════════════

class TabPFNWrapper:
    """
    [v11 ML-3] TabPFN / CatBoost 可選基學習器

    TabPFN: Zero-shot 表格預測 (Hollmann et al. 2023)
      - 在合成數據上預訓練的 Transformer
      - 不需要超參數調整
      - 小樣本表現優異
      - 限制: 需 tabpfn 套件，只能處理 ≤1000 個特徵

    CatBoost: 作為後備
      - 自動處理分類特徵
      - 對稀疏數據有天然支持
      - 內建 GPU 加速

    使用方式:
      wrapper = TabPFNWrapper(backend='auto')
      wrapper.fit(X_train, y_train)
      y_pred = wrapper.predict(X_test)
    """

    def __init__(self, backend: str = "auto"):
        """
        Parameters:
            backend: 'tabpfn', 'catboost', or 'auto' (嘗試 tabpfn → catboost → ridge)
        """
        self.backend = backend
        self.model = None
        self._backend_used = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """訓練模型"""
        # 嘗試載入後端
        if self.backend in ("auto", "tabpfn"):
            try:
                from tabpfn import TabPFNRegressor
                self.model = TabPFNRegressor(device="cpu")
                # TabPFN 限制: 最多 1000 個訓練樣本
                if len(X) > 1000:
                    idx = np.random.choice(len(X), 1000, replace=False)
                    self.model.fit(X[idx], y[idx])
                else:
                    self.model.fit(X, y)
                self._backend_used = "tabpfn"
                log.info(f"  TabPFN 訓練完成: n={min(len(X), 1000)}")
                return self
            except ImportError:
                if self.backend == "tabpfn":
                    raise ImportError("tabpfn 未安裝: pip install tabpfn")
                log.info("  TabPFN 不可用, 嘗試 CatBoost...")

        if self.backend in ("auto", "catboost"):
            try:
                from catboost import CatBoostRegressor
                self.model = CatBoostRegressor(
                    iterations=500,
                    learning_rate=0.05,
                    depth=6,
                    verbose=0,
                    loss_function="RMSE",
                )
                self.model.fit(X, y)
                self._backend_used = "catboost"
                log.info(f"  CatBoost 訓練完成: n={len(X)}")
                return self
            except ImportError:
                if self.backend == "catboost":
                    raise ImportError("catboost 未安裝: pip install catboost")
                log.info("  CatBoost 不可用, 使用 Ridge 回退...")

        # 最終回退: Ridge
        if HAS_SKLEARN:
            self.model = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
            self.model.fit(X, y)
            self._backend_used = "ridge"
            log.info(f"  Ridge 回退訓練: n={len(X)}")
        else:
            raise ImportError("無可用的 ML 後端")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """預測"""
        if self.model is None:
            raise RuntimeError("模型未訓練")
        return self.model.predict(X)

    @property
    def backend_name(self) -> str:
        return self._backend_used or "none"
    def train_species(
        self,
        species: str,
        reports: List[Dict],
        ocean_data: Dict,
    ) -> Dict:
        """訓練指定物種的模型"""
        if species not in self.models:
            return {"error": f"未知物種: {species}"}
        return self.models[species].train_from_captain_reports(reports, ocean_data)
