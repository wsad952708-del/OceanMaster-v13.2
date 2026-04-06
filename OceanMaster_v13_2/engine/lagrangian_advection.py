"""
OceanMaster v13.2 — Lagrangian 生物量平流模擬模組
===================================================
利用流場預判食物鏈下游聚集位置

原理：
  海洋初級生產力 (Chl-a) 在 SST 鋒面、上升流區域高 →
  浮游植物被海流平流輸送 → 3-7 天後轉化為浮游動物 →
  聚集區 = 餌料魚密集 = 鮪魚覓食熱點

  本模組做的事：
  1. 在 Chl-a > 0.2 mg/m³ 的區域播種「虛擬粒子」
  2. 用 HYCOM/CMEMS 海流場 (u, v) 進行 RK4 前向積分 72hr
  3. 計算粒子聚集密度 → 預判未來 3 天的食物聚集熱區
  4. 輸出熱力圖圖層 → 疊加到 OceanMaster_Map.html

  等效於: 「如果今天的高 Chl-a 水塊隨海流漂，3 天後會在哪裡？」

學術依據：
  van Sebille et al. (2018). Lagrangian ocean analysis: fundamentals and practices.
    Ocean Modelling, 121, 49-75.
  Lévy et al. (2018). The role of submesoscale currents in structuring marine
    ecosystems. Nature Communications, 9, 4758.
  Chenillat et al. (2015). Plankton drifter — a new bio-Lagrangian model.
    Biogeosciences, 12, 3083-3105.
  OceanParcels — Delandmeter & van Sebille (2019). The Parcels v2.0 framework.
    Geosci. Model Dev., 12, 3571-3584.

與 OceanMaster 的關係：
  - 使用 data_fetcher_v2.fetch_all() 的 u_current, v_current, chl, lats, lons
  - RK4 積分器沿用 algorithms.py compute_ftle() 的座標轉換和向量化模式
  - 輸出格式可直接與 html_map_generator.py 整合 (Leaflet heatmap layer)
  - 輸出的 hotspot dict 與 ai_fusion.py 相容 (可傳入 kml_generator)
"""

import numpy as np
import logging
import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from dataclasses import dataclass, field

try:
    from scipy.interpolate import RegularGridInterpolator
except ImportError:
    RegularGridInterpolator = None

log = logging.getLogger("OceanMaster.BioAdvection")


# ═══════════════════════════════════════════════════════════
# 配置 — 與 config.py 的 SPECIES_PARAMS 不衝突
# ═══════════════════════════════════════════════════════════

@dataclass
class AdvectionConfig:
    """
    Lagrangian 平流模擬參數

    學術參考:
      - integration_hours=72: Lévy et al. (2018) 中尺度輸送時間尺度 ~3 天
      - dt_hours=1.0: CFL 穩定條件 Δt < Δx / max(|u|)
        0.25° ≈ 27.8 km, max |u| ≈ 2 m/s → Δt_max ≈ 3.9 hr → 1 hr 保守
      - chl_threshold=0.2: MODIS 全球均值 ~0.3, 寡營養海域 <0.1
        0.2 mg/m³ 是高生產力海域的合理下限
      - density_bandwidth_deg=0.5: KDE 頻寬 ≈ 55 km, 與 HSI 熱點 NMS 半徑一致
    """
    integration_hours: float = 72.0     # 平流積分時間
    dt_hours: float = 1.0              # RK4 時間步長
    chl_threshold: float = 0.2         # Chl-a 播種閾值 (mg/m³)
    particle_spacing_deg: float = 0.25  # 粒子播種間距 (度)
    max_particles: int = 50000         # 最大粒子數 (效能限制)
    density_bandwidth_deg: float = 0.5  # 密度估算頻寬
    density_grid_res: float = 0.25     # 輸出密度網格解析度
    decay_rate_per_day: float = 0.1    # 生物量衰減率 (d⁻¹)
    min_density_percentile: float = 70  # 熱點門檻百分位數
    convergence_boost: float = 1.5     # 匯聚區加分係數

    # ── 魚群主動游泳參數 (biased random walk / thermotaxis) ──
    # GreenFish 推測做法: 'species-specific models and parameters
    # which emulate the movements of schools'
    # Reference: Humston et al. (2004) Behavioral assumptions in models
    # of fish movement. Trans. Am. Fish. Soc. 133, 1304-1328.
    swim_enabled: bool = False           # 預設關閉 (純被動漂移)
    swim_speed_ms: float = 0.3           # 魚群游泳速度 (m/s)
                                         # 鮪魚巡航 ~0.3-0.5 m/s
                                         # 柔魚 ~0.1-0.3 m/s
    preferred_sst: float = 24.0          # 偏好 SST (°C)
    sst_sensitivity: float = 2.0         # SST 趨溫靈敏度 (°C)
                                         # 偏離 preferred_sst ± sensitivity
                                         # 以外才開始主動游泳
    random_walk_std: float = 0.01        # 隨機游泳擾動 (度/步)


# ═══════════════════════════════════════════════════════════
# 粒子集合體
# ═══════════════════════════════════════════════════════════

@dataclass
class ParticleSet:
    """
    Lagrangian 粒子集合

    類比 OceanParcels.ParticleSet, 但輕量化
    所有座標存為向量化 numpy 陣列 — 一次 RK4 更新所有粒子
    """
    lon: np.ndarray          # 經度 (度)
    lat: np.ndarray          # 緯度 (度)
    biomass: np.ndarray      # 生物量權重 (初始=Chl-a 值)
    active: np.ndarray       # 活躍狀態 (False=出界/擱淺)
    age_hours: np.ndarray    # 追蹤時間 (hr)
    source_lon: np.ndarray   # 起源經度 (不變)
    source_lat: np.ndarray   # 起源緯度 (不變)

    @property
    def n_active(self) -> int:
        return int(np.sum(self.active))

    @property
    def n_total(self) -> int:
        return len(self.lon)


# ═══════════════════════════════════════════════════════════
# 核心引擎
# ═══════════════════════════════════════════════════════════

class BioAdvectionEngine:
    """
    Lagrangian 生物量平流模擬引擎

    Pipeline:
      1. seed_particles()   — 在 Chl-a > threshold 播種
      2. advect_rk4()       — RK4 前向積分 72 hr
      3. compute_density()  — 粒子聚集密度 → 2D 網格
      4. extract_hotspots() — 從密度場提取漁場候選點

    整合點:
      - 輸入: data_fetcher_v2.fetch_all() 的回傳 dict
      - 輸出: advected_biomass (2D np.ndarray) + hotspot list
      - HTML: export_to_heatmap_layer() → Leaflet heatmap JSON

    用法:
      engine = BioAdvectionEngine()
      result = engine.run(env_data)       # env_data = fetch_all() 回傳
      hotspots = result["hotspots"]       # 可插入 ai_fusion
      heatmap_js = result["heatmap_js"]   # 可插入 HTML map
    """

    def __init__(self, config: Optional[AdvectionConfig] = None):
        self.config = config or AdvectionConfig()

    # ──────────────────────────────────────────────
    #  Step 1: 粒子播種
    # ──────────────────────────────────────────────

    def seed_particles(
        self,
        chl: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
    ) -> ParticleSet:
        """
        在 Chl-a > threshold 的格點播種粒子

        每個格點生成一顆粒子，權重 = Chl-a 值
        超過 max_particles 時，優先保留高 Chl-a 區域

        Parameters:
          chl:  2D (ny, nx) Chl-a 場 (mg/m³)
          lats: 1D (ny,) 緯度陣列
          lons: 1D (nx,) 經度陣列

        Returns:
          ParticleSet — 向量化粒子集合
        """
        cfg = self.config

        # 建立格點座標
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        # 填補 NaN (陸地/雲遮蔽)
        chl_filled = np.nan_to_num(chl, nan=0.0)

        # 播種條件: Chl-a > threshold
        mask = chl_filled > cfg.chl_threshold

        seed_lats = lat_grid[mask]
        seed_lons = lon_grid[mask]
        seed_biomass = chl_filled[mask]

        n_seeds = len(seed_lats)
        log.info(f"  播種: Chl-a > {cfg.chl_threshold} mg/m³ → "
                 f"{n_seeds} 個候選格點")

        # 粒子數限制 — 保留 Chl-a 最高的格點
        if n_seeds > cfg.max_particles:
            top_idx = np.argsort(seed_biomass)[::-1][:cfg.max_particles]
            seed_lats = seed_lats[top_idx]
            seed_lons = seed_lons[top_idx]
            seed_biomass = seed_biomass[top_idx]
            log.info(f"  限制: {n_seeds} → {cfg.max_particles} 粒子 "
                     f"(Chl-a ≥ {seed_biomass[-1]:.3f})")

        n = len(seed_lats)

        pset = ParticleSet(
            lon=seed_lons.astype(np.float64).copy(),
            lat=seed_lats.astype(np.float64).copy(),
            biomass=seed_biomass.astype(np.float64).copy(),
            active=np.ones(n, dtype=bool),
            age_hours=np.zeros(n, dtype=np.float64),
            source_lon=seed_lons.astype(np.float64).copy(),
            source_lat=seed_lats.astype(np.float64).copy(),
        )

        log.info(f"  ParticleSet: {n} 粒子, "
                 f"Chl-a 範圍 {pset.biomass.min():.3f} ~ "
                 f"{pset.biomass.max():.3f} mg/m³")

        return pset

    # ──────────────────────────────────────────────
    #  Step 2: RK4 前向積分
    # ──────────────────────────────────────────────

    def advect_rk4(
        self,
        pset: ParticleSet,
        u: np.ndarray,
        v: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
    ) -> ParticleSet:
        """
        4 階 Runge-Kutta 前向平流

        完全沿用 algorithms.py compute_ftle() 的座標轉換:
          deg_per_m_lon = 1.0 / (111320 × cos(lat))
          deg_per_m_lat = 1.0 / 110540

        Parameters:
          pset: 粒子集合
          u:    2D (ny, nx) 東向流速 (m/s)
          v:    2D (ny, nx) 北向流速 (m/s)
          lats: 1D (ny,) 緯度
          lons: 1D (nx,) 經度

        Returns:
          pset — 就地更新 (in-place) 後返回
        """
        cfg = self.config
        n_steps = int(cfg.integration_hours / cfg.dt_hours)
        dt_sec = cfg.dt_hours * 3600.0  # 轉秒

        log.info(f"  RK4 平流: {cfg.integration_hours}hr, "
                 f"dt={cfg.dt_hours}hr, {n_steps} 步")

        # ── 建立流場插值器 ──
        # 填補 NaN (陸地) — 與 algorithms.py 一致
        u_filled = np.nan_to_num(u, nan=0.0).astype(np.float64)
        v_filled = np.nan_to_num(v, nan=0.0).astype(np.float64)

        if RegularGridInterpolator is not None:
            u_interp = RegularGridInterpolator(
                (lats.astype(np.float64), lons.astype(np.float64)),
                u_filled, bounds_error=False, fill_value=0.0
            )
            v_interp = RegularGridInterpolator(
                (lats.astype(np.float64), lons.astype(np.float64)),
                v_filled, bounds_error=False, fill_value=0.0
            )
            interpolate = self._interp_scipy
        else:
            u_interp = None
            v_interp = None
            interpolate = self._interp_nearest
            log.warning("  scipy 不可用, 使用最近鄰插值 (精度降低)")

        # ── 邊界 ──
        lat_min, lat_max = float(lats.min()), float(lats.max())
        lon_min, lon_max = float(lons.min()), float(lons.max())

        # ── 生物量衰減參數 ──
        decay_factor = np.exp(-cfg.decay_rate_per_day * cfg.dt_hours / 24.0)

        # ── 主積分迴圈 ──
        for step in range(n_steps):
            active = pset.active
            if not np.any(active):
                log.warning(f"    Step {step}: 所有粒子失活, 提前終止")
                break

            px = pset.lon[active]
            py = pset.lat[active]

            # 座標轉換: m/s → deg/s
            # 與 algorithms.py L242-243 完全一致
            deg_per_m_lon = 1.0 / (111320.0 * np.cos(np.radians(py)))
            deg_per_m_lat = 1.0 / 110540.0

            # ── RK4 四階段 ──
            # k1
            k1u, k1v = interpolate(
                py, px, u_interp, v_interp,
                u_filled, v_filled, lats, lons
            )
            k1u *= deg_per_m_lon
            k1v *= deg_per_m_lat

            # k2
            py2 = py + 0.5 * dt_sec * k1v
            px2 = px + 0.5 * dt_sec * k1u
            deg_lon_2 = 1.0 / (111320.0 * np.cos(np.radians(py2)))
            k2u, k2v = interpolate(
                py2, px2, u_interp, v_interp,
                u_filled, v_filled, lats, lons
            )
            k2u *= deg_lon_2
            k2v *= deg_per_m_lat

            # k3
            py3 = py + 0.5 * dt_sec * k2v
            px3 = px + 0.5 * dt_sec * k2u
            deg_lon_3 = 1.0 / (111320.0 * np.cos(np.radians(py3)))
            k3u, k3v = interpolate(
                py3, px3, u_interp, v_interp,
                u_filled, v_filled, lats, lons
            )
            k3u *= deg_lon_3
            k3v *= deg_per_m_lat

            # k4
            py4 = py + dt_sec * k3v
            px4 = px + dt_sec * k3u
            deg_lon_4 = 1.0 / (111320.0 * np.cos(np.radians(py4)))
            k4u, k4v = interpolate(
                py4, px4, u_interp, v_interp,
                u_filled, v_filled, lats, lons
            )
            k4u *= deg_lon_4
            k4v *= deg_per_m_lat

            # RK4 合成 — 與 algorithms.py L263-264 一致
            pset.lon[active] += (dt_sec / 6.0) * (k1u + 2*k2u + 2*k3u + k4u)
            pset.lat[active] += (dt_sec / 6.0) * (k1v + 2*k2v + 2*k3v + k4v)

            # ── 魚群主動游泳 (biased random walk + thermotaxis) ──
            # GreenFish: 'emulate movements of schools' ≠ 純被動漂移
            # → 加入溫度趨向性 + 隨機擾動
            if cfg.swim_enabled:
                swim_dx, swim_dy = _compute_thermotaxis(
                    pset.lat[active], pset.lon[active],
                    u_interp, v_interp, u_filled, v_filled,
                    lats, lons, interpolate,
                    sst_field=None,  # TODO: pass SST field when available
                    swim_speed=cfg.swim_speed_ms,
                    preferred_sst=cfg.preferred_sst,
                    sst_sensitivity=cfg.sst_sensitivity,
                    random_walk_std=cfg.random_walk_std,
                    dt_sec=dt_sec,
                    deg_per_m_lon=deg_per_m_lon,
                    deg_per_m_lat=deg_per_m_lat,
                )
                pset.lon[active] += swim_dx
                pset.lat[active] += swim_dy

            # 更新時間
            pset.age_hours[active] += cfg.dt_hours

            # 生物量衰減 (指數衰減模擬降解/被捕食)
            pset.biomass[active] *= decay_factor

            # ── 邊界檢查: 出界 → 失活 ──
            out_of_bounds = (
                (pset.lat[active] < lat_min - 1.0) |
                (pset.lat[active] > lat_max + 1.0) |
                (pset.lon[active] < lon_min - 1.0) |
                (pset.lon[active] > lon_max + 1.0)
            )
            if np.any(out_of_bounds):
                active_idx = np.where(active)[0]
                pset.active[active_idx[out_of_bounds]] = False

            # 進度日誌 (每 12 步 = 12 hr)
            if (step + 1) % 12 == 0 or step == n_steps - 1:
                elapsed = (step + 1) * cfg.dt_hours
                log.info(f"    t={elapsed:.0f}hr: "
                         f"{pset.n_active}/{pset.n_total} 活躍 "
                         f"({pset.n_active/max(pset.n_total,1)*100:.0f}%)")

        # 最終統計
        if pset.n_active > 0:
            disp_km = np.sqrt(
                ((pset.lon[pset.active] - pset.source_lon[pset.active]) *
                 111.32 * np.cos(np.radians(pset.lat[pset.active])))**2 +
                ((pset.lat[pset.active] - pset.source_lat[pset.active]) * 110.54)**2
            )
            log.info(f"  平流完成: 平均位移 {np.mean(disp_km):.1f} km, "
                     f"最大 {np.max(disp_km):.1f} km")

        return pset

    # ──────────────────────────────────────────────
    #  Step 3: 密度估算
    # ──────────────────────────────────────────────

    def compute_density(
        self,
        pset: ParticleSet,
        lats: np.ndarray,
        lons: np.ndarray,
    ) -> Dict[str, Any]:
        """
        計算粒子聚集密度 → 2D 網格

        使用 2D 直方圖 + Gaussian 平滑 (KDE 近似)
        權重 = 殘餘生物量 (考慮衰減後)

        比 OceanParcels 的 kernel density 更高效 (O(N) vs O(N²))

        Parameters:
          pset: 平流後的粒子集合
          lats: 1D 目標緯度網格
          lons: 1D 目標經度網格

        Returns:
          {
            "advected_biomass": 2D (ny, nx) 歸一化密度場 (0-1),
            "raw_density": 2D 原始密度 (粒子數/格點),
            "convergence_mask": 2D bool 匯聚區遮罩,
            "lat": 1D, "lon": 1D,
            "stats": dict,
          }
        """
        cfg = self.config

        # 只計算活躍粒子
        active = pset.active
        if pset.n_active == 0:
            ny, nx = len(lats), len(lons)
            log.warning("  無活躍粒子, 返回零密度場")
            return {
                "advected_biomass": np.zeros((ny, nx), dtype=np.float32),
                "raw_density": np.zeros((ny, nx), dtype=np.float32),
                "convergence_mask": np.zeros((ny, nx), dtype=bool),
                "lat": lats, "lon": lons,
                "stats": {"n_active": 0},
            }

        px = pset.lon[active]
        py = pset.lat[active]
        pw = pset.biomass[active]

        # ── 2D 加權直方圖 ──
        ny, nx = len(lats), len(lons)
        lat_edges = _build_edges(lats)
        lon_edges = _build_edges(lons)

        density, _, _ = np.histogram2d(
            py, px,
            bins=[lat_edges, lon_edges],
            weights=pw,
        )
        density = density.astype(np.float64)

        # ── Gaussian 平滑 (σ = bandwidth / resolution) ──
        sigma_pix = cfg.density_bandwidth_deg / max(
            np.abs(lats[1] - lats[0]) if len(lats) > 1 else 0.25, 0.01
        )
        try:
            from scipy.ndimage import gaussian_filter
            density_smooth = gaussian_filter(density, sigma=sigma_pix)
        except ImportError:
            # 手動 box filter 回退
            density_smooth = _box_smooth(density, int(sigma_pix * 2 + 1))

        # ── 匯聚偵測 ──
        # 粒子密度遠高於均勻分布 → 匯聚區
        if np.nanmax(density_smooth) > 0:
            uniform_density = pset.n_active / max(ny * nx, 1)
            convergence_mask = density_smooth > (
                cfg.convergence_boost * uniform_density
            )
        else:
            convergence_mask = np.zeros_like(density_smooth, dtype=bool)

        # ── 歸一化到 0-1 ──
        dmax = np.nanmax(density_smooth)
        if dmax > 0:
            advected_biomass = (density_smooth / dmax).astype(np.float32)
        else:
            advected_biomass = np.zeros_like(density_smooth, dtype=np.float32)

        # 統計
        n_convergence = int(np.sum(convergence_mask))
        pct_area = n_convergence / max(ny * nx, 1) * 100

        log.info(f"  密度場: {ny}×{nx} 網格, "
                 f"峰值={dmax:.2f}, 匯聚區={n_convergence} 格點 ({pct_area:.1f}%)")

        return {
            "advected_biomass": advected_biomass,
            "raw_density": density_smooth.astype(np.float32),
            "convergence_mask": convergence_mask,
            "lat": lats,
            "lon": lons,
            "stats": {
                "n_active": pset.n_active,
                "n_total": pset.n_total,
                "peak_density": float(dmax),
                "convergence_cells": n_convergence,
                "convergence_pct": pct_area,
            },
        }

    # ──────────────────────────────────────────────
    #  Step 4: 熱點提取
    # ──────────────────────────────────────────────

    def extract_hotspots(
        self,
        density_result: Dict[str, Any],
        top_n: int = 20,
        min_score: float = 0.4,
        min_distance_deg: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        從密度場中提取漁場候選熱點

        使用與 ai_fusion._extract_hotspots() 相同的 NMS 邏輯:
        按分數降序，排除距離太近的點

        Returns:
          List[Dict] — 與 ai_fusion hotspot dict 格式完全相容
        """
        grid = density_result["advected_biomass"]
        lats = density_result["lat"]
        lons = density_result["lon"]
        conv_mask = density_result["convergence_mask"]

        # 只考慮匯聚區內的高密度格點
        candidates = []
        ny, nx = grid.shape

        for iy in range(ny):
            for ix in range(nx):
                score = float(grid[iy, ix])
                if score >= min_score and conv_mask[iy, ix]:
                    candidates.append({
                        "lat": float(lats[iy]),
                        "lon": float(lons[ix]),
                        "score": score,
                        "iy": iy, "ix": ix,
                    })

        # 如果匯聚區內候選不足，放寬到整個高密度場
        if len(candidates) < 3:
            for iy in range(ny):
                for ix in range(nx):
                    score = float(grid[iy, ix])
                    if score >= min_score:
                        already = any(c["iy"] == iy and c["ix"] == ix
                                      for c in candidates)
                        if not already:
                            candidates.append({
                                "lat": float(lats[iy]),
                                "lon": float(lons[ix]),
                                "score": score,
                                "iy": iy, "ix": ix,
                            })

        # 降序排列
        candidates.sort(key=lambda x: x["score"], reverse=True)

        # NMS (與 ai_fusion.py L170-188 一致)
        selected = []
        for c in candidates:
            too_close = False
            for s in selected:
                dist = np.sqrt(
                    (c["lat"] - s["lat"])**2 + (c["lon"] - s["lon"])**2
                )
                if dist < min_distance_deg:
                    too_close = True
                    break
            if not too_close:
                selected.append({
                    "lat": c["lat"],
                    "lon": c["lon"],
                    "score": c["score"],
                    "species": "multi",  # 平流模擬不特定物種
                    "rank": len(selected) + 1,
                    "source": "lagrangian_advection",
                    "advection_hours": self.config.integration_hours,
                    "explain": (
                        f"Lagrangian 生物量平流 {self.config.integration_hours:.0f}hr: "
                        f"食物匯聚密度 {c['score']*100:.0f}%"
                    ),
                })
                if len(selected) >= top_n:
                    break

        log.info(f"  熱點提取: {len(candidates)} 候選 → "
                 f"{len(selected)} 個 (NMS, d>{min_distance_deg}°)")

        return selected

    # ──────────────────────────────────────────────
    #  Step 5: HTML 熱力圖匯出
    # ──────────────────────────────────────────────

    @staticmethod
    def export_to_heatmap_layer(
        density_result: Dict[str, Any],
        intensity_scale: float = 1.0,
        min_opacity: float = 0.3,
    ) -> str:
        """
        將密度場轉換為 Leaflet.heat 相容的 JavaScript 資料

        輸出格式對應 html_map_generator.py 的 layer 系統:
        可直接插入 <script> 區塊作為新圖層

        Returns:
          JavaScript 程式碼字串, 包含:
          1. const advectionHeatData = [[lat, lon, intensity], ...];
          2. L.heatLayer 圖層建立
          3. 圖層控制 checkbox 綁定

        整合方式:
          將此字串插入到 html_map_generator._build_html() 的
          </script> 之前即可

        注意:
          需要 leaflet-heat 外掛:
          <script src="https://unpkg.com/leaflet.heat/dist/leaflet-heat.js">
        """
        grid = density_result["advected_biomass"]
        lats = density_result["lat"]
        lons = density_result["lon"]
        stats = density_result.get("stats", {})

        # 只輸出高於閾值的格點 (減少資料量)
        threshold = 0.2
        heat_points = []
        ny, nx = grid.shape

        for iy in range(ny):
            for ix in range(nx):
                val = float(grid[iy, ix])
                if val > threshold:
                    heat_points.append([
                        round(float(lats[iy]), 4),
                        round(float(lons[ix]), 4),
                        round(val * intensity_scale, 3),
                    ])

        log.info(f"  Heatmap 匯出: {len(heat_points)} 個熱點格點 "
                 f"(閾值>{threshold})")

        # 生成 JavaScript — 遵循 html_map_generator.py 的 coding style
        js = f"""
// ═══ OceanMaster Lagrangian 生物量平流圖層 ═══
// 粒子數: {stats.get('n_active', '?')}/{stats.get('n_total', '?')}
// 匯聚區: {stats.get('convergence_cells', '?')} 格點 ({stats.get('convergence_pct', 0):.1f}%)
const advectionHeatData = {json.dumps(heat_points, separators=(',', ':'))};

// Leaflet.heat 圖層
if(typeof L.heatLayer !== 'undefined') {{
  const advectionLayer = L.heatLayer(advectionHeatData, {{
    radius: 25,
    blur: 15,
    maxZoom: 10,
    max: {intensity_scale},
    minOpacity: {min_opacity},
    gradient: {{
      0.0: 'transparent',
      0.2: '#1a237e',
      0.4: '#0277bd',
      0.5: '#00838f',
      0.6: '#2e7d32',
      0.7: '#f9a825',
      0.8: '#ef6c00',
      0.9: '#d84315',
      1.0: '#b71c1c'
    }}
  }});

  // 加入圖層控制 — 與既有 layers 物件整合
  if(typeof layers !== 'undefined') {{
    layers.advection = advectionLayer;
  }}

  // 預設不顯示 (使用者自行開啟)
  // map.addLayer(advectionLayer);  // 取消註解以預設顯示
}} else {{
  console.warn('leaflet.heat not loaded — 需要: <script src="https://unpkg.com/leaflet.heat/dist/leaflet-heat.js">');
}}
"""
        return js

    @staticmethod
    def export_to_heatmap_json(
        density_result: Dict[str, Any],
    ) -> str:
        """
        匯出為純 JSON 格式 (供外部應用讀取)

        格式:
        {
          "type": "advected_biomass",
          "grid": { "lat_min": ..., "lat_max": ..., "lon_min": ..., "lon_max": ..., "resolution": ... },
          "data": [[lat, lon, intensity], ...],
          "stats": { ... }
        }
        """
        grid = density_result["advected_biomass"]
        lats = density_result["lat"]
        lons = density_result["lon"]

        data_points = []
        ny, nx = grid.shape
        for iy in range(ny):
            for ix in range(nx):
                val = float(grid[iy, ix])
                if val > 0.1:
                    data_points.append([
                        round(float(lats[iy]), 4),
                        round(float(lons[ix]), 4),
                        round(val, 4),
                    ])

        return json.dumps({
            "type": "advected_biomass",
            "engine": "OceanMaster_BioAdvection_v10.4",
            "integration_hours": 72,
            "grid": {
                "lat_min": float(lats.min()),
                "lat_max": float(lats.max()),
                "lon_min": float(lons.min()),
                "lon_max": float(lons.max()),
                "resolution": float(lats[1] - lats[0]) if len(lats) > 1 else 0.25,
            },
            "data": data_points,
            "stats": density_result.get("stats", {}),
        }, indent=2, ensure_ascii=False)

    # ──────────────────────────────────────────────
    #  主入口: 一鍵執行
    # ──────────────────────────────────────────────

    def run(
        self,
        env_data: Dict[str, Any],
        top_n: int = 20,
    ) -> Dict[str, Any]:
        """
        一鍵執行完整的 Lagrangian 生物量平流分析

        Parameters:
          env_data: data_fetcher_v2.fetch_all() 的回傳 dict
                    必須包含: lats, lons, chl, u_current, v_current
          top_n:    最多提取幾個熱點

        Returns:
          {
            "advected_biomass": 2D np.ndarray (0-1 歸一化密度),
            "density_result":   完整密度結果 dict,
            "particle_set":     ParticleSet 物件,
            "hotspots":         List[Dict] — ai_fusion 相容格式,
            "heatmap_js":       str — Leaflet JavaScript 程式碼,
            "heatmap_json":     str — JSON 資料,
            "html_layer_ctrl":  str — 圖層控制 HTML 片段,
            "html_script_tag":  str — leaflet-heat script tag,
            "stats":            dict,
          }
        """
        log.info("═══ Lagrangian 生物量平流模擬 開始 ═══")

        # ── 解包環境數據 (來自 fetch_all) ──
        lats = env_data.get("lats", env_data.get("lat"))
        lons = env_data.get("lons", env_data.get("lon"))

        chl = env_data.get("chl")
        u = env_data.get("u_current")
        v = env_data.get("v_current")

        if lats is None or lons is None:
            raise ValueError("env_data 缺少 lats/lons")
        if chl is None:
            raise ValueError("env_data 缺少 chl (葉綠素)")
        if u is None or v is None:
            raise ValueError("env_data 缺少 u_current/v_current (海流)")

        log.info(f"  環境場: {len(lats)}×{len(lons)} 網格, "
                 f"Chl-a: {np.nanmin(chl):.3f}~{np.nanmax(chl):.3f} mg/m³, "
                 f"|u|_max={np.nanmax(np.abs(u)):.2f} m/s")

        # ── Step 1: 播種 ──
        pset = self.seed_particles(chl, lats, lons)

        if pset.n_total == 0:
            log.warning("  無粒子可播種 (Chl-a 全部低於閾值)")
            empty_density = {
                "advected_biomass": np.zeros((len(lats), len(lons)), np.float32),
                "raw_density": np.zeros((len(lats), len(lons)), np.float32),
                "convergence_mask": np.zeros((len(lats), len(lons)), bool),
                "lat": lats, "lon": lons,
                "stats": {"n_active": 0, "n_total": 0},
            }
            return {
                "advected_biomass": empty_density["advected_biomass"],
                "density_result": empty_density,
                "particle_set": pset,
                "hotspots": [],
                "heatmap_js": "// No particles to advect",
                "heatmap_json": "{}",
                "html_layer_ctrl": "",
                "html_script_tag": "",
                "stats": {"n_particles": 0},
            }

        # ── Step 2: RK4 平流 ──
        pset = self.advect_rk4(pset, u, v, lats, lons)

        # ── Step 3: 密度計算 ──
        density_result = self.compute_density(pset, lats, lons)

        # ── Step 4: 熱點提取 ──
        hotspots = self.extract_hotspots(density_result, top_n=top_n)

        # ── Step 5: 匯出 ──
        heatmap_js = self.export_to_heatmap_layer(density_result)
        heatmap_json = self.export_to_heatmap_json(density_result)

        # HTML 插入片段
        html_script_tag = ('<script src="https://unpkg.com/'
                           'leaflet.heat/dist/leaflet-heat.js"></script>')

        html_layer_ctrl = (
            '<label><input type="checkbox" id="lyrAdvection"> '
            '🌊 生物量平流預測</label>'
        )

        stats = {
            "n_particles_seeded": pset.n_total,
            "n_particles_active": pset.n_active,
            "integration_hours": self.config.integration_hours,
            "convergence_cells": density_result["stats"].get(
                "convergence_cells", 0),
            "hotspots_found": len(hotspots),
        }

        log.info(f"═══ Lagrangian 生物量平流模擬 完成 ═══")
        log.info(f"  播種: {pset.n_total}, 存活: {pset.n_active}, "
                 f"熱點: {len(hotspots)}")

        return {
            "advected_biomass": density_result["advected_biomass"],
            "density_result": density_result,
            "particle_set": pset,
            "hotspots": hotspots,
            "heatmap_js": heatmap_js,
            "heatmap_json": heatmap_json,
            "html_layer_ctrl": html_layer_ctrl,
            "html_script_tag": html_script_tag,
            "stats": stats,
        }

    def predict_lcs_drift(
        self,
        ftle_field: np.ndarray,
        forecast_u: Dict[int, np.ndarray],
        forecast_v: Dict[int, np.ndarray],
        lats: np.ndarray,
        lons: np.ndarray,
        ridge_threshold_pct: float = 85.0,
        days: int = 3,
    ) -> Dict[str, Any]:
        """
        [v18] Predict where LCS ridges (ocean fronts) will be in 1-3 days.

        ** GLOBALLY UNIQUE FEATURE — no competitor has predictive LCS. **
        CLS/CATSAT shows where fronts ARE today.
        We predict where fronts WILL BE tomorrow.

        Algorithm:
          1. Extract FTLE ridge points (top 15 percentile)
          2. Seed particles on ridge locations
          3. Forward-advect using CMEMS forecast flow fields
          4. Output: future LCS positions per day

        Ref: CLS FSLE methodology (d'Ovidio 2004) + CMEMS forecast

        Args:
            ftle_field: 2D FTLE field (from algorithms.py compute_ftle)
            forecast_u: {day_offset: 2D u-current array}
            forecast_v: {day_offset: 2D v-current array}
            lats, lons: 1D coordinate arrays
            ridge_threshold_pct: percentile for ridge extraction (default 85)
            days: forecast days (1-3)

        Returns:
            {
                "day1_ridges": [(lat, lon), ...],
                "day2_ridges": [(lat, lon), ...],
                "day3_ridges": [(lat, lon), ...],
                "summary": str,
            }
        """
        log.info("═══ [v18] Predictive LCS Drift — 鋒面漂移預報 ═══")

        # 1. Extract FTLE ridges (top percentile)
        ftle_safe = np.nan_to_num(ftle_field, nan=0)
        threshold = np.percentile(ftle_safe[ftle_safe > 0], ridge_threshold_pct)
        ridge_mask = ftle_safe > threshold

        ridge_lats = []
        ridge_lons = []
        ny, nx = ftle_safe.shape
        for iy in range(ny):
            for ix in range(nx):
                if ridge_mask[iy, ix]:
                    ridge_lats.append(float(lats[iy]))
                    ridge_lons.append(float(lons[ix]))

        n_ridges = len(ridge_lats)
        if n_ridges == 0:
            log.warning("  No FTLE ridges found above threshold")
            return {"day1_ridges": [], "day2_ridges": [], "day3_ridges": [],
                    "summary": "No ridges detected"}

        # Subsample if too many (max 2000)
        if n_ridges > 2000:
            idx = np.linspace(0, n_ridges - 1, 2000, dtype=int)
            ridge_lats = [ridge_lats[i] for i in idx]
            ridge_lons = [ridge_lons[i] for i in idx]

        # 2. Create particles on ridges
        pset = ParticleSet(
            lon=np.array(ridge_lons),
            lat=np.array(ridge_lats),
            biomass=np.ones(len(ridge_lats)),
        )

        log.info(f"  LCS ridges: {len(ridge_lats)} points seeded")

        result = {}
        for day in range(1, min(days + 1, 4)):
            u_fc = forecast_u.get(day)
            v_fc = forecast_v.get(day)

            if u_fc is None or v_fc is None:
                log.debug(f"  Day{day}: no forecast flow field")
                result[f"day{day}_ridges"] = []
                continue

            # 3. Advect 24 hours with this day's forecast
            pset = self.advect_rk4(
                pset, u_fc, v_fc, lats, lons,
            )

            # 4. Record positions
            active = pset.active
            future_positions = list(zip(
                pset.lat[active].tolist(),
                pset.lon[active].tolist(),
            ))
            result[f"day{day}_ridges"] = [
                (round(lat, 3), round(lon, 3))
                for lat, lon in future_positions[:500]  # cap output
            ]
            log.info(f"  Day{day}: {len(future_positions)} ridge points predicted")

        result["summary"] = (
            f"LCS drift: {len(ridge_lats)} ridge points → "
            f"Day1: {len(result.get('day1_ridges', []))}, "
            f"Day2: {len(result.get('day2_ridges', []))}, "
            f"Day3: {len(result.get('day3_ridges', []))}"
        )
        log.info(f"  {result['summary']}")
        return result

    # ──────────────────────────────────────────────
    #  插值器 (內部)
    # ──────────────────────────────────────────────

    @staticmethod
    def _interp_scipy(
        lat_pts: np.ndarray,
        lon_pts: np.ndarray,
        u_interp: Any,
        v_interp: Any,
        u_grid: np.ndarray,
        v_grid: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        scipy RegularGridInterpolator 插值
        與 algorithms.py compute_ftle() L248 一致
        """
        pts = np.column_stack([lat_pts, lon_pts])
        return u_interp(pts), v_interp(pts)

    @staticmethod
    def _interp_nearest(
        lat_pts: np.ndarray,
        lon_pts: np.ndarray,
        u_interp: Any,
        v_interp: Any,
        u_grid: np.ndarray,
        v_grid: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        最近鄰插值回退 (無 scipy 時)
        與 data_fetcher_v2._regrid() 的回退邏輯一致
        """
        iy = np.clip(
            np.searchsorted(lats, lat_pts) - 1, 0, len(lats) - 1
        ).astype(int)
        ix = np.clip(
            np.searchsorted(lons, lon_pts) - 1, 0, len(lons) - 1
        ).astype(int)
        return u_grid[iy, ix], v_grid[iy, ix]


# ═══════════════════════════════════════════════════════════
# KML 匯出 — 與 kml_generator.py 格式相容
# ═══════════════════════════════════════════════════════════

def generate_advection_kml(
    density_result: Dict[str, Any],
    hotspots: List[Dict[str, Any]],
    output_path: str = "output/Bio_Advection_Prediction.kml",
    vessel_lat: float = 25.13,
    vessel_lon: float = 121.74,
) -> str:
    """
    生成生物量平流預測 KML

    格式與 kml_generator.py 完全相容:
    - 使用相同的 Style ID
    - 相同的 Placemark 描述結構
    - 可與主 KML 合併或獨立使用
    """
    filepath = Path(output_path)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    parts = [f'''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
<n>🌊 OceanMaster 生物量平流預測</n>
<description>Lagrangian 粒子追蹤 → 食物匯聚熱區
算法: RK4 前向積分 72hr (dt=1hr)
播種: Chl-a > 0.2 mg/m³
生成: {now_str}
⚠️ 本預測僅供參考</description>
<Style id="s_best"><IconStyle><scale>1.3</scale><Icon><href>https://maps.google.com/mapfiles/kml/paddle/red-circle.png</href></Icon></IconStyle></Style>
<Style id="s_good"><IconStyle><scale>1.1</scale><Icon><href>https://maps.google.com/mapfiles/kml/paddle/orange-circle.png</href></Icon></IconStyle></Style>
<Style id="s_fair"><IconStyle><scale>0.9</scale><Icon><href>https://maps.google.com/mapfiles/kml/paddle/ylw-circle.png</href></Icon></IconStyle></Style>
<Style id="s_low"><IconStyle><scale>0.7</scale><Icon><href>https://maps.google.com/mapfiles/kml/paddle/wht-circle.png</href></Icon></IconStyle></Style>
<Style id="s_vessel"><IconStyle><scale>1.2</scale><Icon><href>https://maps.google.com/mapfiles/kml/shapes/sailing.png</href></Icon></IconStyle></Style>
<LookAt><longitude>{vessel_lon}</longitude><latitude>{vessel_lat}</latitude><range>3000000</range></LookAt>''']

    # 匯聚區熱點
    parts.append(
        '<Folder><n>🌊 生物量匯聚預測 (72hr)</n><open>1</open>'
        '<description>Lagrangian 平流: Chl-a 高生產力水塊 → 72hr 後聚集位置\n'
        '= 未來 3 天的餌料聚集熱區</description>'
    )

    SP_ZH = {
        "skipjack": "鰹魚", "yellowfin": "黃鰭鮪", "bigeye": "大目鮪",
        "albacore": "長鰭鮪", "squid_todarodes": "赤魷", "multi": "多物種",
    }

    for h in hotspots:
        pct = int(round(h["score"] * 100))
        if h["score"] >= 0.75:
            style, label = "s_best", "🔴 極高密度"
        elif h["score"] >= 0.60:
            style, label = "s_good", "🟠 高密度"
        elif h["score"] >= 0.45:
            style, label = "s_fair", "🟡 中密度"
        else:
            style, label = "s_low", "⚪ 低密度"

        sp_zh = SP_ZH.get(h.get("species", ""), h.get("species", ""))

        desc = (
            f"【Lagrangian 生物量平流預測】{label}\n"
            f"═══ 匯聚密度: {pct}% ═══\n"
            f"🌊 算法: RK4 前向積分 {h.get('advection_hours', 72):.0f}hr\n"
            f"🌿 來源: 高 Chl-a 區域平流聚集\n"
            f"🐟 適用: {sp_zh}\n"
            f"📍 位置: {h['lat']:.3f}°N, {h['lon']:.3f}°E\n"
            f"📊 {h.get('explain', '')}"
        )

        parts.append(
            f'<Placemark><n>#{h["rank"]} 生物量匯聚 ({pct}%)</n>\n'
            f'<description><![CDATA[{desc}]]></description>\n'
            f'<styleUrl>#{style}</styleUrl>\n'
            f'<Point><coordinates>{h["lon"]},{h["lat"]},0</coordinates>'
            f'</Point></Placemark>'
        )

    parts.append('</Folder>')

    # 船位
    parts.append(
        f'<Folder><n>⛵ 船位</n>\n'
        f'<Placemark><n>⛵ 母港/船位</n>\n'
        f'<description>位置: {vessel_lat:.2f}N, {vessel_lon:.2f}E</description>\n'
        f'<styleUrl>#s_vessel</styleUrl>\n'
        f'<Point><coordinates>{vessel_lon},{vessel_lat},0</coordinates>'
        f'</Point></Placemark></Folder>'
    )

    parts.append('</Document></kml>')

    kml_content = "\n".join(parts)
    filepath.write_text(kml_content, encoding="utf-8")
    size_kb = filepath.stat().st_size / 1024
    log.info(f"  平流 KML: {filepath} ({size_kb:.1f} KB, "
             f"{len(hotspots)} hotspots)")

    return str(filepath)



# ═══════════════════════════════════════════════════════════
# [v16.0 F3] FAD 漂流物熟成時間模型
# ═══════════════════════════════════════════════════════════

def estimate_fad_maturity(
    sst: float,
    chl: float,
    days_deployed: float,
    current_speed_ms: float = 0.2,
) -> Dict[str, Any]:
    """
    [v16.0 F3] Estimate FAD (Fish Aggregating Device) maturity.

    FAD productivity depends on bio-fouling colonization time:
    - Warm water (SST > 28°C) + high CHL → faster colonization
    - Cold water or oligotrophic → slower colonization
    - Mature FAD (>15 days) attracts tuna schools

    Ref:
      Dagorn et al. (2013) Fish and Fisheries 14:491-509
      Moreno et al. (2007) Aquatic Living Resources 20:321-330

    Returns:
        {
            "maturity_pct": float (0-100% matured),
            "days_to_mature": float (remaining days to full maturity),
            "colonization_rate": float (% per day),
            "is_productive": bool,
            "advisory": str,
        }
    """
    # Base colonization rate: ~3-5% per day in tropics
    base_rate = 3.5  # %/day

    # SST factor: warmer = faster colonization
    sst_factor = np.clip((sst - 20) / 10, 0.3, 2.0)

    # CHL factor: higher productivity = faster fouling
    chl_factor = np.clip(chl / 0.3, 0.5, 2.0)

    # Current effect: moderate current brings nutrients
    current_factor = np.clip(0.8 + current_speed_ms, 0.5, 1.5)

    rate = float(base_rate * sst_factor * chl_factor * current_factor)
    maturity = min(100.0, rate * days_deployed)

    days_remaining = max(0, (100.0 - maturity) / max(rate, 0.1))
    is_productive = maturity >= 60.0  # 60% = minimum productive threshold

    if maturity >= 90:
        advisory = "🟢 FAD 完全成熟，高度吸引鮪魚群"
    elif maturity >= 60:
        advisory = "🟡 FAD 可用，開始聚魚"
    elif maturity >= 30:
        advisory = f"🟠 FAD 發育中，約 {days_remaining:.0f} 天成熟"
    else:
        advisory = f"🔴 FAD 未成熟 ({maturity:.0f}%)，等待 {days_remaining:.0f} 天"

    return {
        "maturity_pct": round(maturity, 1),
        "days_to_mature": round(days_remaining, 1),
        "colonization_rate": round(rate, 2),
        "is_productive": is_productive,
        "advisory": advisory,
    }


# ═══════════════════════════════════════════════════════════
# 魚群主動游泳 (thermotaxis + biased random walk)
# ═══════════════════════════════════════════════════════════

def _compute_thermotaxis(
    lat_pts, lon_pts,
    u_interp, v_interp, u_grid, v_grid, lats, lons, interpolate_fn,
    sst_field=None,
    swim_speed=0.3,
    preferred_sst=24.0,
    sst_sensitivity=2.0,
    random_walk_std=0.01,
    dt_sec=3600.0,
    deg_per_m_lon=None,
    deg_per_m_lat=1.0 / 110540.0,
):
    """
    Compute biased random walk displacement for fish school movement.

    GreenFish 推測做法: 'species-specific models and parameters
    which emulate the movements of schools' ≠ 純被動漂移

    Components:
    1. Thermotaxis: swim toward preferred_sst along SST gradient
       - Only activates when |SST - preferred| > sensitivity
       - Strength proportional to distance from preferred SST
    2. Random walk: stochastic perturbation (correlated random walk)

    Args:
        lat_pts, lon_pts: particle positions (1D arrays)
        sst_field: 2D SST field for gradient (optional, None = random only)
        swim_speed: m/s swimming speed
        preferred_sst: target temperature (°C)
        sst_sensitivity: °C threshold before directed swimming starts
        random_walk_std: std of random displacement per step (degrees)
        dt_sec: time step in seconds
        deg_per_m_lon: conversion factor (1D array matching lat_pts)
        deg_per_m_lat: conversion factor (scalar)

    Returns:
        (dx, dy): displacement in degrees
    """
    n = len(lat_pts)
    dx = np.zeros(n, dtype=np.float64)
    dy = np.zeros(n, dtype=np.float64)

    # ── Component 1: Thermotaxis (directed swimming) ──
    if sst_field is not None and RegularGridInterpolator is not None:
        try:
            sst_filled = np.nan_to_num(sst_field, nan=preferred_sst).astype(np.float64)

            # SST gradient: ∂T/∂lat, ∂T/∂lon
            if len(lats) > 2 and len(lons) > 2:
                dlat = float(lats[1] - lats[0]) * 110540.0  # meters
                dlon = float(lons[1] - lons[0]) * 111320.0 * np.cos(np.radians(np.mean(lats)))
                grad_lat = np.gradient(sst_filled, dlat, axis=0)  # °C/m
                grad_lon = np.gradient(sst_filled, dlon, axis=1)  # °C/m

                # Interpolate gradient at particle positions
                interp_glat = RegularGridInterpolator(
                    (lats.astype(np.float64), lons.astype(np.float64)),
                    grad_lat, bounds_error=False, fill_value=0.0
                )
                interp_glon = RegularGridInterpolator(
                    (lats.astype(np.float64), lons.astype(np.float64)),
                    grad_lon, bounds_error=False, fill_value=0.0
                )

                # SST at particle positions
                interp_sst = RegularGridInterpolator(
                    (lats.astype(np.float64), lons.astype(np.float64)),
                    sst_filled, bounds_error=False, fill_value=preferred_sst
                )

                points = np.column_stack([lat_pts, lon_pts])
                sst_at_particles = interp_sst(points)
                glat = interp_glat(points)
                glon = interp_glon(points)

                # Temperature deviation from preferred
                deviation = sst_at_particles - preferred_sst  # positive = too warm

                # Only swim when deviation exceeds sensitivity
                should_swim = np.abs(deviation) > sst_sensitivity
                swim_strength = np.where(
                    should_swim,
                    np.clip(np.abs(deviation) / (preferred_sst * 0.1 + 1e-8), 0, 1),
                    0.0
                )

                # Swim direction: toward preferred temperature
                # If too warm (deviation > 0): swim toward cooler water (against gradient)
                # If too cool (deviation < 0): swim toward warmer water (along gradient)
                grad_mag = np.sqrt(glat**2 + glon**2) + 1e-12
                sign = -np.sign(deviation)  # negative = swim against gradient direction

                # Normalize gradient direction
                swim_lat = sign * glat / grad_mag * swim_speed * swim_strength
                swim_lon = sign * glon / grad_mag * swim_speed * swim_strength

                # Convert m/s to degrees
                if deg_per_m_lon is not None:
                    dx += swim_lon * deg_per_m_lon * dt_sec
                dy += swim_lat * deg_per_m_lat * dt_sec

        except Exception as e:
            log.debug(f"  Thermotaxis fallback: {e}")

    # ── Component 2: Random walk (stochastic perturbation) ──
    dx += np.random.normal(0, random_walk_std, n)
    dy += np.random.normal(0, random_walk_std, n)

    return dx, dy


# ═══════════════════════════════════════════════════════════
# 內部工具函數
# ═══════════════════════════════════════════════════════════

def _build_edges(centers: np.ndarray) -> np.ndarray:
    """
    從格點中心座標建立 histogram bin 邊界
    用法: np.histogram2d 需要 edges 而非 centers
    """
    if len(centers) < 2:
        return np.array([centers[0] - 0.125, centers[0] + 0.125])
    dx = centers[1] - centers[0]
    edges = np.empty(len(centers) + 1)
    edges[:-1] = centers - dx / 2
    edges[-1] = centers[-1] + dx / 2
    return edges


def _box_smooth(data: np.ndarray, size: int) -> np.ndarray:
    """
    簡易 box filter (不需 scipy)
    用於 Gaussian 平滑的回退方案
    """
    if size < 1:
        return data
    size = max(size, 1) | 1  # 確保奇數
    kernel = np.ones((size, size)) / (size * size)
    ny, nx = data.shape
    result = np.zeros_like(data)
    pad = size // 2
    padded = np.pad(data, pad, mode='edge')
    for iy in range(ny):
        for ix in range(nx):
            result[iy, ix] = np.sum(
                padded[iy:iy+size, ix:ix+size] * kernel
            )
    return result


# ═══════════════════════════════════════════════════════════
# CLI / 獨立測試
# ═══════════════════════════════════════════════════════════

def _main():
    """獨立測試入口"""
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # 模擬 fetch_all() 回傳 — 使用 WOA 氣候態
    sys.path.insert(0, str(Path(__file__).parent.parent))
    try:
        from engine.data_fetcher_v2 import WOAClimatology
        clim = WOAClimatology()
    except ImportError:
        clim = None

    from config import AreaConfig
    area = AreaConfig()

    lats = np.arange(area.lat_min, area.lat_max + 0.01, area.resolution)
    lons = np.arange(area.lon_min, area.lon_max + 0.01, area.resolution)
    month = datetime.now().month

    if clim:
        chl = clim.chl(lats, lons, month)
        u, v = clim.currents(lats, lons)
        sst = clim.sst(lats, lons, month)
    else:
        ny, nx = len(lats), len(lons)
        lat_g, lon_g = np.meshgrid(lats, lons, indexing='ij')
        chl = 0.3 * np.exp(-((lat_g - 15)**2/100 + (lon_g - 150)**2/200))
        u = 0.2 * np.ones((ny, nx))
        v = 0.05 * np.ones((ny, nx))
        sst = 28.0 - 0.4 * np.abs(lat_g - 15)

    env_data = {
        "lats": lats, "lons": lons,
        "chl": chl, "u_current": u, "v_current": v,
        "sst": sst,
    }

    # 執行
    engine = BioAdvectionEngine()
    result = engine.run(env_data)

    print(f"\n{'='*60}")
    print("Lagrangian 生物量平流模擬結果")
    print(f"{'='*60}")
    print(f"播種粒子: {result['stats']['n_particles_seeded']}")
    print(f"存活粒子: {result['stats']['n_particles_active']}")
    print(f"匯聚格點: {result['stats']['convergence_cells']}")
    print(f"熱點數: {result['stats']['hotspots_found']}")
    print(f"密度場: {result['advected_biomass'].shape}")
    print()

    for h in result["hotspots"][:10]:
        print(f"  #{h['rank']} ({h['lat']:6.2f}N, {h['lon']:7.2f}E) "
              f"score={h['score']:.2f} | {h['explain']}")

    # KML 輸出
    kml_path = generate_advection_kml(
        result["density_result"],
        result["hotspots"],
    )
    print(f"\nKML: {kml_path}")

    # JSON 輸出
    json_path = Path("output/advected_biomass.json")
    json_path.write_text(result["heatmap_json"], encoding="utf-8")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    _main()
