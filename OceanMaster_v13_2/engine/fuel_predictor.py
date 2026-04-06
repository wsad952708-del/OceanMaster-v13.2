"""
OceanMaster v13.2 — Fuel Consumption Predictor [G1]
=====================================================
Predict fuel consumption (L/nm) based on sea state and vessel parameters.
Uses existing wave/current/wind data — zero new APIs.

Ref: IMO MEPC.1/Circ.684 (EEXI/CII guidelines)
"""

import numpy as np
import logging
from typing import Dict, Any, Optional

log = logging.getLogger("OceanMaster.Fuel")


class FuelPredictor:
    """Predict per-nautical-mile fuel consumption from sea state."""

    # Default vessel parameters (typical tuna longliner ~500GT)
    DEFAULT_VESSEL = {
        "displacement_ton": 500,
        "base_consumption_l_per_nm": 45.0,  # L/nm at calm sea
        "max_speed_kn": 12.0,
    }

    def __init__(self, vessel_params: Optional[Dict] = None):
        self.vessel = vessel_params or self.DEFAULT_VESSEL

    def predict(
        self,
        wave_height: float = 0.0,
        wave_period: float = 8.0,
        current_speed_kn: float = 0.0,
        current_angle_deg: float = 0.0,
        wind_speed_ms: float = 0.0,
        wind_angle_deg: float = 0.0,
        vessel_speed_kn: float = 10.0,
    ) -> Dict[str, Any]:
        """
        Predict fuel consumption for a single leg.

        current_angle_deg: 0=following, 180=head current
        wind_angle_deg: 0=tailwind, 180=headwind

        Returns:
            {
                "fuel_l_per_nm": float,
                "fuel_penalty_pct": float (% increase over calm),
                "wave_penalty": float,
                "current_penalty": float,
                "wind_penalty": float,
                "advisory": str,
            }
        """
        base = self.vessel["base_consumption_l_per_nm"]

        # [v18] Enhanced wave added resistance (Kwon 2008 J. Ship Res.)
        # Includes wave steepness + encounter angle
        # Ref: ZeroNorth uses similar physics-based approach
        steepness = wave_height / max(1.56 * wave_period**2, 1.0)
        wave_factor = 1.0 + 0.015 * wave_height**2 * (1 + 10 * steepness)

        # Current effect on ground speed:
        #   0°=following → ground speed increases → less fuel per nm
        #   180°=head → ground speed decreases → more fuel per nm
        current_component = current_speed_kn * np.cos(np.radians(current_angle_deg))
        ground_speed = max(vessel_speed_kn + current_component, 1.0)
        current_factor = vessel_speed_kn / ground_speed

        # [v18] Enhanced wind resistance with direction factor
        # Convention: 0°=tailwind, 180°=headwind
        # Headwind → max resistance; tailwind → slight reduction
        # cos(180 - angle): headwind cos(0)=1.0, tailwind cos(180)=-1.0
        relative_heading = np.cos(np.radians(180.0 - wind_angle_deg))
        # +0.4 for full headwind, -0.1 for full tailwind
        direction_factor = 1.0 + 0.4 * max(0, relative_heading) - 0.1 * max(0, -relative_heading)
        wind_factor = 1.0 + 0.002 * wind_speed_ms**2 * direction_factor

        total = base * wave_factor * current_factor * wind_factor
        penalty_pct = (total / base - 1.0) * 100

        if penalty_pct > 40:
            advisory = "⛽ 油耗嚴重增加，建議改航或等待"
        elif penalty_pct > 20:
            advisory = "⛽ 油耗偏高，考慮降速或調整航向"
        elif penalty_pct > 10:
            advisory = "油耗略增，可接受"
        else:
            advisory = "油耗正常"

        return {
            "fuel_l_per_nm": round(total, 2),
            "fuel_penalty_pct": round(penalty_pct, 1),
            "wave_penalty": round((wave_factor - 1) * 100, 1),
            "current_penalty": round((current_factor - 1) * 100, 1),
            "wind_penalty": round((wind_factor - 1) * 100, 1),
            "advisory": advisory,
        }

    def estimate_trip_fuel(
        self,
        distance_nm: float,
        wave_height: float = 1.5,
        current_speed_kn: float = 0.5,
        current_angle_deg: float = 90.0,
        wind_speed_ms: float = 5.0,
        wind_angle_deg: float = 90.0,
    ) -> Dict[str, float]:
        """Estimate total trip fuel consumption."""
        per_nm = self.predict(
            wave_height, 8.0, current_speed_kn, current_angle_deg,
            wind_speed_ms, wind_angle_deg,
        )
        total = per_nm["fuel_l_per_nm"] * distance_nm
        return {
            "total_fuel_l": round(total, 0),
            "distance_nm": distance_nm,
            "fuel_l_per_nm": per_nm["fuel_l_per_nm"],
            "penalty_pct": per_nm["fuel_penalty_pct"],
        }
