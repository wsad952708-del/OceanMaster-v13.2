"""
OceanMaster v13.2 — 8-Day Rolling HSI Forecast
=============================================
模仿 GreenFish 的 8 天漁場位置預測。

使用已有的 CMEMS forecast SST 數據，逐日計算 HSI grid，
提取每日 top-N hotspot 座標 + 信心度分級。

數據來源: CMEMS analysis-forecast (已由 data_fetcher_v2 抓取)
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("OceanMaster.Forecast")

# Day weighting: Day1=1.0, Day8=0.3 (exponential decay)
_DAY_WEIGHTS = {i: max(0.3, 1.0 * math.exp(-0.15 * (i - 1))) for i in range(1, 10)}

# Species SST optima (same as commercial_core)
_SST_OPTIMA = {
    "yellowfin": (24, 30), "bigeye": (15, 25), "skipjack": (26, 32),
    "albacore": (15, 22), "japanese_flying_squid": (12, 20),
    "pacific_saury": (10, 18), "mahi_mahi": (24, 30),
    "blue_marlin": (24, 30), "mackerel_scad": (22, 28),
}


def _sst_hsi(sst: np.ndarray, species: str) -> np.ndarray:
    """計算 SST-based HSI (Gaussian envelope)"""
    lo, hi = _SST_OPTIMA.get(species, (20, 28))
    mid = (lo + hi) / 2
    sigma = (hi - lo) / 2.5
    hsi = np.exp(-0.5 * ((sst - mid) / max(sigma, 1)) ** 2)
    return np.nan_to_num(hsi, nan=0.0).astype(np.float32)


def _extract_daily_hotspots(
    hsi: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    species: str,
    day: int,
    top_n: int = 10,
    min_hsi: float = 0.3,
    min_dist_deg: float = 1.0,
) -> List[Dict]:
    """從單日 HSI grid 提取 top-N hotspots (NMS)"""
    filled = np.nan_to_num(hsi, nan=0)
    candidates = []
    ny, nx = filled.shape
    for iy in range(ny):
        for ix in range(nx):
            if filled[iy, ix] >= min_hsi:
                candidates.append({
                    "lat": float(lats[iy]),
                    "lon": float(lons[ix]),
                    "hsi": float(filled[iy, ix]),
                })
    candidates.sort(key=lambda x: x["hsi"], reverse=True)

    # NMS
    selected = []
    for c in candidates:
        if len(selected) >= top_n:
            break
        too_close = any(
            np.sqrt((c["lat"] - s["lat"])**2 + (c["lon"] - s["lon"])**2) < min_dist_deg
            for s in selected
        )
        if not too_close:
            conf = "High" if c["hsi"] >= 0.7 else "Medium" if c["hsi"] >= 0.5 else "Low"
            selected.append({
                "lat": c["lat"],
                "lon": c["lon"],
                "hsi": round(c["hsi"], 3),
                "confidence": conf,
                "species": species,
                "forecast_day": day,
                "weight": round(_DAY_WEIGHTS.get(day, 0.3), 2),
            })
    return selected


def generate_8day_forecast(
    forecast_data: Dict[int, Dict],
    lats: np.ndarray,
    lons: np.ndarray,
    species_list: List[str],
    current_hsi_results: Optional[Dict] = None,
    top_n_per_day: int = 10,
) -> Dict:
    """
    生成 8 天滾動 HSI 預報。

    Args:
        forecast_data: {day_offset: {"sst": ndarray, "u": ..., "v": ...}}
        lats, lons: 1D coordinate arrays
        species_list: 物種列表
        current_hsi_results: 當日 HSI 結果 (optional, 用於 Day0 基準)
        top_n_per_day: 每天每物種提取的 hotspot 數

    Returns:
        {
            "days": {1: [...hotspots...], 2: [...], ..., 8: [...]},
            "summary": {"total_hotspots": N, "species_count": M},
            "best_windows": [{"day": 3, "species": "bigeye", "hsi": 0.82}],
        }
    """
    if not forecast_data:
        log.warning("  Forecast: no CMEMS forecast data available")
        return {"days": {}, "summary": {"total_hotspots": 0}}

    days_result = {}
    best_windows = []
    total_hotspots = 0

    for day_offset in range(1, 9):
        fc = forecast_data.get(day_offset) or forecast_data.get(str(day_offset))
        if fc is None:
            log.debug(f"  Forecast Day{day_offset}: no data")
            continue

        fc_sst = fc.get("sst")
        if fc_sst is None:
            continue

        # Resize if needed
        target_shape = (len(lats), len(lons))
        if fc_sst.shape != target_shape:
            from scipy.ndimage import zoom
            zy = target_shape[0] / fc_sst.shape[0]
            zx = target_shape[1] / fc_sst.shape[1]
            fc_sst = zoom(fc_sst, (zy, zx), order=1)

        # [v18] Enhanced forecast: add CHL + current convergence if available
        # Ref: GreenFish uses multi-factor satellite ensemble, not just SST
        fc_chl = fc.get("chl")
        fc_u = fc.get("u_current") or fc.get("u")
        fc_v = fc.get("v_current") or fc.get("v")

        chl_bonus = np.zeros(target_shape, dtype=np.float32)
        conv_bonus = np.zeros(target_shape, dtype=np.float32)

        if fc_chl is not None:
            if fc_chl.shape != target_shape:
                from scipy.ndimage import zoom as _zoom
                _zy = target_shape[0] / fc_chl.shape[0]
                _zx = target_shape[1] / fc_chl.shape[1]
                fc_chl = _zoom(fc_chl, (_zy, _zx), order=1)
            # CHL productivity bonus: 0.1-1.0 mg/m³ optimal
            chl_safe = np.nan_to_num(fc_chl, nan=0)
            chl_score = np.where(
                (chl_safe >= 0.1) & (chl_safe <= 2.0),
                np.clip(chl_safe / 1.0, 0, 1),
                0
            )
            chl_bonus = 0.15 * chl_score.astype(np.float32)

        if fc_u is not None and fc_v is not None:
            try:
                if fc_u.shape != target_shape:
                    fc_u = zoom(fc_u, (target_shape[0]/fc_u.shape[0], target_shape[1]/fc_u.shape[1]), order=1)
                    fc_v = zoom(fc_v, (target_shape[0]/fc_v.shape[0], target_shape[1]/fc_v.shape[1]), order=1)
                # Convergence = negative divergence → upwelling → nutrient enrichment
                du_dx = np.gradient(np.nan_to_num(fc_u, nan=0), axis=1)
                dv_dy = np.gradient(np.nan_to_num(fc_v, nan=0), axis=0)
                div = du_dx + dv_dy
                conv = np.clip(-div * 1e4, 0, 1)  # scale + only convergence
                conv_bonus = 0.10 * conv.astype(np.float32)
            except Exception as e:
                log.debug(f"[降級] engine/forecast_hsi.py: {e}")

        day_hotspots = []
        for sp in species_list:
            hsi = _sst_hsi(fc_sst, sp)

            # [v18] Add multi-factor bonus
            hsi = np.clip(hsi + chl_bonus + conv_bonus, 0, 1)

            # Apply day weight (confidence decay)
            weight = _DAY_WEIGHTS.get(day_offset, 0.3)
            weighted_hsi = hsi * weight

            spots = _extract_daily_hotspots(
                weighted_hsi, lats, lons, sp, day_offset,
                top_n=top_n_per_day,
            )
            day_hotspots.extend(spots)

            # Track best window
            if spots:
                best = max(spots, key=lambda x: x["hsi"])
                best_windows.append({
                    "day": day_offset,
                    "species": sp,
                    "hsi": best["hsi"],
                    "lat": best["lat"],
                    "lon": best["lon"],
                })

        # Sort by HSI within day
        day_hotspots.sort(key=lambda x: x["hsi"], reverse=True)
        days_result[day_offset] = day_hotspots[:20]  # top-20 per day
        total_hotspots += len(days_result[day_offset])

    # Find overall best fishing windows
    best_windows.sort(key=lambda x: x["hsi"], reverse=True)

    log.info(
        f"  📅 8-Day Forecast: {total_hotspots} hotspots across "
        f"{len(days_result)} days, best={best_windows[0]['hsi']:.2f} "
        f"(Day{best_windows[0]['day']} {best_windows[0]['species']})"
        if best_windows else "  📅 8-Day Forecast: no valid hotspots"
    )

    return {
        "days": days_result,
        "summary": {
            "total_hotspots": total_hotspots,
            "forecast_days": len(days_result),
            "species_count": len(species_list),
        },
        "best_windows": best_windows[:5],
    }
