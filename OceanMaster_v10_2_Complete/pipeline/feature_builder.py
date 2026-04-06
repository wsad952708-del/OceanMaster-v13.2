"""
OceanMaster v10.5 — Feature Builder
=====================================
專職將原始觀測數據轉換為 ML 所需的特徵矩陣。

反幻覺鐵律:
  - 原始數據 NaN → 特徵 NaN，嚴禁補零
  - Gap-fill 僅限空間內插 (鄰域平均)，不可補歷史平均
  - EKE 使用科氏力 + Haversine 球面距離
"""
import logging
import numpy as np
from typing import Any, Dict, Optional
from dataclasses import dataclass, field

log = logging.getLogger("OceanMaster.features")

# ── 物理演算法 ──
from engine.algorithms import (
    detect_sst_fronts, compute_boa_gradient,
    compute_ftle, compute_thermocline, calculate_eke,
)
from engine.ocean_physics import BathymetryAnalyzer, OceanPhysicsEngine

try:
    from engine.gebco_features import GEBCOFeatures
    HAS_GEBCO = True
except ImportError:
    HAS_GEBCO = False

try:
    from engine.thermocline_fetcher import ThermoclineFetcher
    from engine.eddy_detector import EddyDetector as GFEddyDetector
    from engine.forage_engine import ForageEngine
    from engine.dvm_model import DVMModel
    GREENFISH_OK = True
except ImportError:
    GREENFISH_OK = False


@dataclass
class FeatureMatrix:
    """物理/生態特徵矩陣容器"""
    # SST/CHL (possibly gap-filled)
    sst: np.ndarray = None
    chl: np.ndarray = None

    # 鋒面
    sst_fronts: Dict = field(default_factory=dict)
    chl_fronts: Dict = field(default_factory=dict)
    front_strength: np.ndarray = None

    # 動力學
    ftle_field: np.ndarray = None
    eke_result: Dict = field(default_factory=dict)

    # 溫躍層
    thermocline_result: Optional[Dict] = None

    # 海底地形
    bathy: np.ndarray = None
    depth_si: Dict = field(default_factory=dict)
    gebco_feats: Dict = field(default_factory=dict)

    # GreenFish
    gf_z20: np.ndarray = None
    gf_mld: np.ndarray = None
    gf_eddy_result: Optional[Dict] = None
    gf_forage: Optional[Dict] = None
    gf_feeding: Dict = field(default_factory=dict)

    # Physics
    physics: Dict = field(default_factory=dict)

    # data quality
    nan_map: Dict = field(default_factory=dict)  # feature → NaN ratio


class FeatureBuilder:
    """
    原始數據 → ML 特徵矩陣

    - Gap-fill: 空間插補 (NaN < 70% 才填)
    - NaN 傳播: raw NaN → feature NaN → ML skip
    - EKE: geostrophic (Coriolis + Haversine)
    """

    def __init__(self, species: list):
        self.species = species
        self.physics_engine = OceanPhysicsEngine()
        self.gebco = GEBCOFeatures() if HAS_GEBCO else None

        if GREENFISH_OK:
            self.thermocline = ThermoclineFetcher()
            self.gf_eddy = GFEddyDetector()
            self.forage_engine = ForageEngine()
            self.dvm_model = DVMModel()

    def build(self, raw, month: int, now=None) -> FeatureMatrix:
        """
        raw: RawOceanData
        回傳: FeatureMatrix (NaN 保留, 不可補)
        """
        fm = FeatureMatrix()
        lats, lons = raw.lats, raw.lons
        ny, nx = raw.ny, raw.nx

        # ── Gap-fill SST/CHL (空間內插, 非幻覺) ──
        fm.sst = self._gap_fill(raw.sst, "SST", max_nan=0.70)
        fm.chl = self._gap_fill(raw.chl, "CHL", max_nan=0.70)

        # ── 鋒面偵測 ──
        fm.sst_fronts = detect_sst_fronts(fm.sst, lats, lons)
        fm.front_strength = fm.sst_fronts["front_strength"]
        fm.chl_fronts = compute_boa_gradient(fm.chl, lats, lons)

        # ── FTLE ──
        fm.ftle_field = compute_ftle(raw.u_current, raw.v_current, lats, lons)["ftle"]

        # ── EKE (v10.4 geostrophic) ──
        if raw.ssh is not None:
            fm.eke_result = calculate_eke(raw.ssh, lats, lons)
        else:
            fm.eke_result = {"eke": np.full((ny, nx), np.nan, dtype=np.float32)}

        # ── Thermocline ──
        if raw.temp_3d is not None:
            depths = np.array([0, 50, 100, 200, 300, 500])[:raw.temp_3d.shape[0]]
            fm.thermocline_result = compute_thermocline(raw.temp_3d, depths, lats, lons)

        # ── Bathymetry ──
        fm.bathy = BathymetryAnalyzer.generate_bathymetry(lats, lons)
        for sp in self.species:
            fm.depth_si[sp] = BathymetryAnalyzer.compute_depth_suitability(fm.bathy, sp)

        if self.gebco is not None:
            try:
                fm.gebco_feats = self.gebco.compute_features(lats, lons, bathy=fm.bathy)
            except Exception as e:
                log.debug(f"  GEBCO: {e}")

        # ── Physics ──
        fm.physics = self.physics_engine.analyze_full(
            raw.u_current, raw.v_current, raw.salinity,
            raw.wind_u, raw.wind_v, lats,
            species_list=self.species[:4],
        )

        # ── Data quality追蹤 ──
        data_sources = raw.data_sources
        currents_is_clim = "CLIM" in str(data_sources.get("CURRENTS", "")).upper()
        if currents_is_clim:
            log.warning("  ⚠️ 海流為氣候態 → EKE/渦旋偵測將受限")

        # ── GreenFish Lite ──
        if GREENFISH_OK:
            log.info("  🐟 GreenFish Lite: Z20/MLD/Eddy/Forage/DVM")
            self._build_greenfish(fm, raw, lats, lons, ny, nx, month, now, currents_is_clim)

        # ── NaN map ──
        for name, arr in [("sst", fm.sst), ("chl", fm.chl), ("ftle", fm.ftle_field),
                          ("front", fm.front_strength)]:
            if arr is not None:
                ratio = float(np.sum(np.isnan(arr)) / max(arr.size, 1))
                fm.nan_map[name] = ratio
                if ratio > 0.5:
                    log.warning(f"  ⚠️ {name}: NaN ratio = {ratio:.1%}")

        return fm

    def _gap_fill(self, data: np.ndarray, name: str, max_nan: float = 0.70) -> np.ndarray:
        """
        空間插補 gap-fill。
        若 NaN > max_nan → 保留 NaN (不可補值產生幻覺)
        """
        if data is None:
            return data
        nan_ratio = np.sum(np.isnan(data)) / max(data.size, 1)
        if nan_ratio <= 0.01:
            return data  # 乾淨

        if nan_ratio > max_nan:
            log.warning(f"  {name}: NaN={nan_ratio:.0%} > {max_nan:.0%} → 保留 NaN (禁止填補)")
            return data

        # 空間插補 (近鄰平均, 非幻覺)
        log.info(f"  {name}: gap-fill {nan_ratio:.0%} NaN via spatial interpolation")
        try:
            from scipy.ndimage import generic_filter

            def _nanmean(x):
                v = x[~np.isnan(x)]
                return np.nanmean(v) if len(v) > 0 else np.nan

            filled = data.copy()
            for radius in [3, 5, 9]:
                still_nan = np.isnan(filled)
                if not np.any(still_nan):
                    break
                smoothed = generic_filter(filled, _nanmean, size=radius,
                                          mode='constant', cval=np.nan)
                filled = np.where(still_nan, smoothed, filled)

            # 仍有 NaN → 保留 NaN (嚴禁全域中位數)
            new_ratio = np.sum(np.isnan(filled)) / max(filled.size, 1)
            log.info(f"  {name}: {nan_ratio:.1%} → {new_ratio:.1%} NaN")
            return filled
        except ImportError:
            return data

    def _build_greenfish(self, fm, raw, lats, lons, ny, nx, month, now, currents_is_clim):
        """GreenFish Lite 子系統特徵"""
        # Z20/MLD
        glorys = self.thermocline.fetch_glorys12(lats, lons, month)
        if glorys is not None:
            fm.gf_z20 = glorys["z20"]
            fm.gf_mld = glorys["mld"]
        elif raw.temp_3d is not None and raw.temp_3d.ndim == 3 and raw.temp_3d.shape[0] >= 3:
            hycom_coverage = (raw.temp_3d.shape[1] * raw.temp_3d.shape[2]) / max(ny * nx, 1)
            if hycom_coverage < 0.01:
                fm.gf_z20 = ThermoclineFetcher.climatology_z20(lats, lons, month)
                fm.gf_mld = ThermoclineFetcher.climatology_mld(lats, lons, month)
            else:
                tc = self.thermocline.compute_from_temp3d(raw.temp_3d, lats, lons)
                fm.gf_z20 = tc["z20"]
                fm.gf_mld = tc["mld"]
        else:
            fm.gf_z20 = ThermoclineFetcher.climatology_z20(lats, lons, month)
            fm.gf_mld = ThermoclineFetcher.climatology_mld(lats, lons, month)

        # Eddy
        if raw.ssh is not None and raw.u_current is not None and not currents_is_clim:
            fm.gf_eddy_result = self.gf_eddy.detect(raw.ssh, raw.u_current, raw.v_current, lats, lons)
        else:
            fm.gf_eddy_result = self.gf_eddy._empty_result(ny, nx)

        # Forage
        fm.gf_forage = self.forage_engine.compute(
            fm.chl, fm.sst, lats, z20=fm.gf_z20, mld=fm.gf_mld, month=month,
        )

        # DVM feeding
        for sp in self.species:
            sp_key = sp.split("_")[0] if "squid" not in sp else sp
            if sp_key in ("squid_todarodes", "squid_ommastrephes"):
                fm.gf_feeding[sp] = fm.gf_forage["forage_total"]
            else:
                try:
                    hour_utc = now.hour if now else 12
                    fm.gf_feeding[sp] = self.dvm_model.compute_feeding_index(
                        sp, fm.gf_forage["forage_surface"], fm.gf_forage["forage_deep"],
                        fm.gf_z20, fm.gf_mld, fm.sst, hour_utc=hour_utc,
                    )
                except Exception as e:
                    log.warning(f"  DVM {sp}: {e}")
                    fm.gf_feeding[sp] = fm.gf_forage["forage_total"].copy()
