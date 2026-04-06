"""
OceanMaster v13.2 — Departure Optimizer [G3]
==============================================
Recommend optimal departure time based on weather forecast windows,
transit time to fishing grounds, and required operation hours.
"""

import numpy as np
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional

log = logging.getLogger("OceanMaster.Departure")


class DepartureOptimizer:
    """Optimize port departure timing for maximum fishing efficiency."""

    def __init__(self, port_lat: float = 24.15, port_lon: float = 120.6):
        """Default: Kaohsiung (前鎮漁港)."""
        self.port_lat = port_lat
        self.port_lon = port_lon

    def optimize(
        self,
        target_lat: float,
        target_lon: float,
        vessel_speed_kn: float = 10.0,
        required_fishing_hours: float = 72.0,
        forecast_windows: Optional[List[Dict]] = None,
        n_days_ahead: int = 7,
    ) -> List[Dict[str, Any]]:
        """
        Find best departure windows in the next N days.

        Args:
            target_lat/lon: fishing ground coordinates
            vessel_speed_kn: cruising speed
            required_fishing_hours: hours needed on fishing ground
            forecast_windows: [{start, end, wind_ms, wave_m}, ...]
                If None, generates synthetic good-weather windows.

        Returns:
            List of departure options sorted by score.
        """
        # Transit distance
        dist_nm = self._haversine_nm(self.port_lat, self.port_lon, target_lat, target_lon)
        transit_hours = dist_nm / max(vessel_speed_kn, 1.0)
        total_hours = transit_hours * 2 + required_fishing_hours

        now = datetime.now(timezone.utc)

        if forecast_windows is None:
            forecast_windows = self._generate_windows(now, n_days_ahead)

        options = []
        for i, window in enumerate(forecast_windows):
            w_start = window.get("start", now + timedelta(hours=i * 12))
            if isinstance(w_start, str):
                w_start = datetime.fromisoformat(w_start.replace("Z", "+00:00"))
            w_end = window.get("end", w_start + timedelta(hours=48))
            if isinstance(w_end, str):
                w_end = datetime.fromisoformat(w_end.replace("Z", "+00:00"))

            window_hours = (w_end - w_start).total_seconds() / 3600
            wind = window.get("wind_ms", 8.0)
            wave = window.get("wave_m", 1.5)

            # Departure must allow full trip within window
            if window_hours < total_hours:
                continue

            # Score: larger window + calmer seas = better
            surplus = window_hours - total_hours
            sea_score = max(0, 1.0 - wave / 4.0) * max(0, 1.0 - wind / 20.0)
            score = 0.4 * min(surplus / 48, 1.0) + 0.6 * sea_score

            depart_time = w_start
            arrive_time = depart_time + timedelta(hours=transit_hours)
            return_time = arrive_time + timedelta(hours=required_fishing_hours + transit_hours)

            options.append({
                "depart": depart_time.strftime("%Y-%m-%d %H:%M UTC"),
                "arrive_fishing": arrive_time.strftime("%m-%d %H:%M"),
                "return_port": return_time.strftime("%m-%d %H:%M"),
                "transit_hours": round(transit_hours, 1),
                "distance_nm": round(dist_nm, 0),
                "window_surplus_h": round(surplus, 1),
                "wind_ms": wind,
                "wave_m": wave,
                "score": round(float(score), 3),
            })

        options.sort(key=lambda x: x["score"], reverse=True)
        if options:
            best = options[0]
            log.info(f"  Departure optimizer: best={best['depart']}, "
                     f"score={best['score']}")
        return options

    def _generate_windows(self, now, n_days):
        """
        [v18] Fetch real weather windows from Open-Meteo Marine API.

        Falls back to synthetic if API unavailable.
        Ref: ZeroNorth/Sofar weather-optimized departure concept.
        API: https://marine-api.open-meteo.com/v1/marine (free, no key)
        """
        try:
            import urllib.request
            import json

            lat = self.port_lat
            lon = self.port_lon
            url = (
                f"https://marine-api.open-meteo.com/v1/marine?"
                f"latitude={lat}&longitude={lon}"
                f"&hourly=wave_height,wind_wave_height"
                f"&forecast_days={min(n_days, 7)}"
                f"&timezone=UTC"
            )
            # Also get wind from weather API
            weather_url = (
                f"https://api.open-meteo.com/v1/forecast?"
                f"latitude={lat}&longitude={lon}"
                f"&hourly=wind_speed_10m,wind_gusts_10m"
                f"&forecast_days={min(n_days, 7)}"
                f"&timezone=UTC"
            )

            req = urllib.request.Request(url, headers={"User-Agent": "OceanMaster/15.3"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                marine = json.loads(resp.read())

            req2 = urllib.request.Request(weather_url, headers={"User-Agent": "OceanMaster/15.3"})
            with urllib.request.urlopen(req2, timeout=10) as resp2:
                weather = json.loads(resp2.read())

            hourly_wave = marine.get("hourly", {}).get("wave_height", [])
            hourly_wind = weather.get("hourly", {}).get("wind_speed_10m", [])
            times = marine.get("hourly", {}).get("time", [])

            if not hourly_wave or len(hourly_wave) < 24:
                raise ValueError("Insufficient marine data")

            # Build 12-hour windows from hourly data
            windows = []
            step = 12  # hours per window
            for i in range(0, min(len(hourly_wave) - step, n_days * 24), step):
                chunk_wave = hourly_wave[i:i + step]
                chunk_wind = hourly_wind[i:i + step] if hourly_wind else [8.0] * step
                avg_wave = sum(w for w in chunk_wave if w is not None) / max(len([w for w in chunk_wave if w is not None]), 1)
                avg_wind = sum(w for w in chunk_wind if w is not None) / max(len([w for w in chunk_wind if w is not None]), 1)

                from datetime import datetime, timedelta, timezone  # noqa: already at top
                start = now + timedelta(hours=i)
                end = start + timedelta(hours=48)

                windows.append({
                    "start": start, "end": end,
                    "wind_ms": round(avg_wind, 1),
                    "wave_m": round(avg_wave, 1),
                })

            log.info(f"  🌊 Open-Meteo forecast: {len(windows)} windows fetched")
            return windows if windows else self._synthetic_windows(now, n_days)

        except Exception as e:
            log.warning(f"  Open-Meteo API failed ({e}), using synthetic windows")
            return self._synthetic_windows(now, n_days)

    def _synthetic_windows(self, now, n_days):
        """Fallback: synthetic weather windows."""
        windows = []
        for d in range(n_days):
            start = now + timedelta(days=d, hours=6)
            end = start + timedelta(hours=120)
            wind = 5.0 + np.random.default_rng(d).uniform(0, 10)
            wave = 0.5 + np.random.default_rng(d + 100).uniform(0, 3)
            windows.append({"start": start, "end": end, "wind_ms": round(wind, 1), "wave_m": round(wave, 1)})
        return windows

    @staticmethod
    def _haversine_nm(lat1, lon1, lat2, lon2):
        R_nm = 3440.065
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
        return R_nm * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
