"""
OceanMaster v13.2 — SST Rate of Change + Upwelling Event Detector [#2]
========================================================================
Detect rapid SST changes that trigger fish aggregation events.

Key insight: rapid cooling (-1°C/day or more) indicates upwelling.
  → Nutrients surge → phytoplankton bloom in 3-5 days
  → Zooplankton bloom in 7-10 days → Fish aggregation

Ref: Zainuddin et al. (2006) Deep-Sea Res. 53:210-222
     Podesta et al. (1993) ICES J. Mar. Sci. 50:3-11
"""

import numpy as np
import logging
from typing import Dict, Any, Optional

log = logging.getLogger("OceanMaster.dSST")


class SSTChangeDetector:
    """
    [v16.0 #2] Detect SST rate of change and upwelling events.

    Uses current SST vs forecast/historical to compute dSST/dt.
    Identifies upwelling events (rapid cooling) and warm intrusions.
    """

    def __init__(
        self,
        cooling_threshold: float = -0.5,  # °C/day for mild upwelling
        strong_cooling: float = -1.0,      # °C/day for strong upwelling
        warming_threshold: float = 0.5,    # °C/day for warm intrusion
    ):
        self.cooling_threshold = cooling_threshold
        self.strong_cooling = strong_cooling
        self.warming_threshold = warming_threshold

    def compute(
        self,
        sst_current: np.ndarray,
        sst_previous: Optional[np.ndarray] = None,
        sst_forecast: Optional[Dict[int, Any]] = None,
        dt_days: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Compute SST rate of change.

        Args:
            sst_current: Current SST grid (°C)
            sst_previous: Previous SST grid (°C), e.g. yesterday
            sst_forecast: {day_offset: {"sst": grid}}, e.g. from CMEMS forecast
            dt_days: time difference in days between current and previous

        Returns:
            {
                "dsst_dt": 2D rate (°C/day),
                "upwelling_mask": 2D bool,
                "strong_upwelling_mask": 2D bool,
                "warming_mask": 2D bool,
                "event_type": 2D int (-2=strong_up, -1=mild_up, 0=stable, 1=warming),
                "upwelling_pct": float,
                "forecast_trend": Optional[2D] (°C over forecast period),
                "advisory": str,
            }
        """
        ny, nx = sst_current.shape

        # ── Method 1: Current vs Previous ──
        if sst_previous is not None and sst_previous.shape == sst_current.shape:
            dsst = (sst_current - sst_previous) / max(dt_days, 0.1)
        # ── Method 2: Use forecast day 1 as "future" ──
        elif sst_forecast and 1 in sst_forecast:
            fc = sst_forecast[1]
            if isinstance(fc, dict) and "sst" in fc:
                fc_sst = fc["sst"]
                if fc_sst.shape == sst_current.shape:
                    dsst = fc_sst - sst_current  # positive = will warm
                else:
                    dsst = np.zeros((ny, nx), dtype=np.float32)
            else:
                dsst = np.zeros((ny, nx), dtype=np.float32)
        else:
            # No comparison available — estimate from SST spatial gradient
            # (convergence of cold water = proxy for temporal cooling)
            gx = np.gradient(sst_current, axis=1)
            gy = np.gradient(sst_current, axis=0)
            dsst = -np.sqrt(gx**2 + gy**2)  # stronger gradient → more dynamic

        dsst = np.nan_to_num(dsst, nan=0).astype(np.float32)

        # Event classification
        upwelling = dsst < self.cooling_threshold
        strong_upwelling = dsst < self.strong_cooling
        warming = dsst > self.warming_threshold

        event_type = np.zeros((ny, nx), dtype=np.int8)
        event_type[upwelling] = -1
        event_type[strong_upwelling] = -2
        event_type[warming] = 1

        upwelling_pct = float(np.mean(upwelling) * 100)

        # Forecast trend (multi-day)
        fc_trend = None
        if sst_forecast:
            max_day = max(sst_forecast.keys())
            fc_last = sst_forecast[max_day]
            if isinstance(fc_last, dict) and "sst" in fc_last:
                fc_sst_last = fc_last["sst"]
                if fc_sst_last.shape == sst_current.shape:
                    fc_trend = (fc_sst_last - sst_current).astype(np.float32)

        # Advisory
        strong_pct = float(np.mean(strong_upwelling) * 100)
        if strong_pct > 10:
            advisory = f"強湧升流偵測: {strong_pct:.0f}% 區域急速降溫 — 3-5天後漁獲爆發"
        elif upwelling_pct > 15:
            advisory = f"湧升流活躍: {upwelling_pct:.0f}% 區域降溫中"
        elif float(np.mean(warming) * 100) > 20:
            advisory = "暖水入侵: 部分區域升溫 — 適合黃鰭/正鰹"
        else:
            advisory = "SST 穩定 — 無顯著湧升流事件"

        log.info(f"  dSST/dt: upwelling={upwelling_pct:.1f}%, "
                 f"strong={strong_pct:.1f}%, "
                 f"range=[{dsst.min():.2f}, {dsst.max():.2f}] °C/day")

        return {
            "dsst_dt": dsst,
            "upwelling_mask": upwelling,
            "strong_upwelling_mask": strong_upwelling,
            "warming_mask": warming,
            "event_type": event_type,
            "upwelling_pct": round(upwelling_pct, 1),
            "forecast_trend": fc_trend,
            "advisory": advisory,
        }
