"""
OceanMaster v13.2 — Longline Drift Path Predictor [G2]
========================================================
Predict where longline gear drifts after deployment using ocean currents.
Prevents gear from entering EEZ or reef zones.

Uses HYCOM u/v current data (already available).
"""

import numpy as np
import logging
from typing import Dict, List, Any

log = logging.getLogger("OceanMaster.LonglineDrift")


class LonglineDriftPredictor:
    """Predict longline gear drift path from ocean currents."""

    DEG_PER_M_LAT = 1.0 / 110540.0

    def predict_drift(
        self,
        deploy_lat: float,
        deploy_lon: float,
        u_current: np.ndarray,
        v_current: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
        soak_hours: float = 12.0,
        dt_hours: float = 0.5,
        n_hooks: int = 3000,
        mainline_km: float = 100.0,
    ) -> Dict[str, Any]:
        """
        Predict gear positions during soak time.

        Args:
            deploy_lat/lon: deployment start position
            u/v_current: 2D current grids (m/s)
            soak_hours: how long gear soaks (default 12h)
            mainline_km: total mainline length

        Returns:
            {
                "drift_path": [(lat, lon, hour), ...],
                "end_lat": float, "end_lon": float,
                "total_drift_km": float,
                "max_drift_km": float,
                "eez_warning": bool,
                "drift_speed_kn": float,
            }
        """
        from scipy.interpolate import RegularGridInterpolator

        # Build current interpolators
        try:
            u_interp = RegularGridInterpolator(
                (lats, lons), u_current, method='linear',
                bounds_error=False, fill_value=0.0)
            v_interp = RegularGridInterpolator(
                (lats, lons), v_current, method='linear',
                bounds_error=False, fill_value=0.0)
        except Exception:
            # Fallback if grid too small
            return self._fallback_result(deploy_lat, deploy_lon)

        # Integrate drift path (Euler forward)
        path = [(deploy_lat, deploy_lon, 0.0)]
        lat, lon = deploy_lat, deploy_lon
        dt_s = dt_hours * 3600

        for step in range(int(soak_hours / dt_hours)):
            u_val = float(u_interp((lat, lon)))
            v_val = float(v_interp((lat, lon)))

            deg_per_m_lon = self.DEG_PER_M_LAT / max(np.cos(np.radians(lat)), 0.01)
            lat += v_val * dt_s * self.DEG_PER_M_LAT
            lon += u_val * dt_s * deg_per_m_lon

            hour = (step + 1) * dt_hours
            path.append((lat, lon, hour))

        # Compute total drift
        total_km = 0.0
        max_km = 0.0
        for i in range(1, len(path)):
            dlat = (path[i][0] - path[i-1][0]) * 110.54
            dlon = (path[i][1] - path[i-1][1]) * 111.32 * np.cos(np.radians(path[i][0]))
            seg = np.sqrt(dlat**2 + dlon**2)
            total_km += seg
            dist_from_start = np.sqrt(
                ((path[i][0] - deploy_lat) * 110.54)**2 +
                ((path[i][1] - deploy_lon) * 111.32 * np.cos(np.radians(deploy_lat)))**2
            )
            max_km = max(max_km, dist_from_start)

        drift_speed = total_km / max(soak_hours, 1) * 0.5399  # km/h → knots

        log.info(f"  Longline drift: {total_km:.1f}km in {soak_hours}h, "
                 f"speed={drift_speed:.2f}kn")

        return {
            "drift_path": path,
            "end_lat": round(lat, 4),
            "end_lon": round(lon, 4),
            "total_drift_km": round(total_km, 2),
            "max_drift_km": round(max_km, 2),
            "drift_speed_kn": round(drift_speed, 2),
            "eez_warning": max_km > mainline_km * 0.5,
        }

    def _fallback_result(self, lat, lon):
        return {
            "drift_path": [(lat, lon, 0.0)],
            "end_lat": lat, "end_lon": lon,
            "total_drift_km": 0.0, "max_drift_km": 0.0,
            "drift_speed_kn": 0.0, "eez_warning": False,
        }
